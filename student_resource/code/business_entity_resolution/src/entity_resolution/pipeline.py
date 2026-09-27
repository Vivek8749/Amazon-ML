"""End-to-end orchestration: train, full (train + test), and predict modes.

Every expensive stage is checkpointed (see checkpoint.py): rerunning the same
command after a killed session skips finished stages and continues.
"""
import gc
import hashlib
import os
import pickle
import time

import numpy as np
import pandas as pd

from .blocking import build_blockers, generate_all_candidates
from .checkpoint import Checkpoint
from .config import (
    MODEL_PATH, N_WORKERS, OUTPUT_DIR, PRED_SHARD_ENTITIES, RANDOM_SEED, TEST_S1,
    TEST_S2, TEST_S3, TRAIN_GT, TRAIN_S1, TRAIN_S2, TRAIN_S3, VAL_FRACTION,
)
from .data import load_pool_sampled, load_source, parse_ground_truth, write_output
from .evaluation import evaluate
from .features import FEATURE_NAMES
from .preprocessing import PREPROCESS_VERSION
from .inference import predict_all, singleton_post_process
from .training import (
    build_training_data, find_best_threshold, hard_negative_mining, train_xgb,
)


def _save_model(model, threshold):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "threshold": threshold,
                     "preprocess_version": PREPROCESS_VERSION,
                     "feature_names": list(FEATURE_NAMES)}, f)
    print(f"[Model] Saved -> {MODEL_PATH}")


def _prepare_training_data(sample_size):
    """Steps 1-5: sample S1, load the S2+S3 pool, split train/validation."""
    # ---- 1. Load S1 + GT (fast, small-ish files) ----
    print("[Data] Loading S1 + GT...")
    t0 = time.time()
    s1_full = load_source(TRAIN_S1)
    gt_full = pd.read_csv(TRAIN_GT, sep="\t", dtype=str)
    print(f"[Data] S1: {len(s1_full):,}, GT: {len(gt_full):,} in {time.time()-t0:.1f}s")

    # ---- 2. Sample S1 ----
    rng = np.random.RandomState(RANDOM_SEED)
    if sample_size is None or sample_size <= 0 or sample_size >= len(s1_full):
        print(f"[Data] Using FULL dataset ({len(s1_full):,} entities)")
        s1s = s1_full.copy()
        gts = gt_full.copy()
        extra = 500_000
    else:
        ids = rng.choice(s1_full["entity_id"].values,
                         size=min(sample_size, len(s1_full)), replace=False)
        s1s = s1_full[s1_full["entity_id"].isin(set(ids))].copy()
        gts = gt_full[gt_full["source1_entity_id"].isin(set(ids))].copy()
        extra = min(max(sample_size * 3, 20_000), 30_000)  # cap at 30K/country to avoid OOM
        # Pool size = must_haves + 30K × n_countries ≈ 100-150K records max
        # At 150K pool: 2 threads × 500 queries × 150K × 4B = ~600MB ✔
    countries = set(s1s["country_norm"].unique())
    del s1_full, gt_full; gc.collect()

    # ---- 3. Find all S2/S3 IDs referenced in GT (must-have for pool) ----
    must_have = set()
    for _, row in gts.iterrows():
        m = row.get("matched_entity_ids", "")
        if pd.notna(m) and m:
            must_have.update(str(m).split(","))
    print(f"[Data] S1 sample: {len(s1s):,}, must-have pool IDs: {len(must_have):,}")

    # ---- 4. Smart pool loading (threaded, chunked) ----
    pool = load_pool_sampled(TRAIN_S2, TRAIN_S3, must_have, countries,
                              extra_per_country=extra)

    # ---- 5. Train/val split ----
    ids2 = s1s["entity_id"].values.copy(); rng.shuffle(ids2)
    nv = max(int(len(ids2) * VAL_FRACTION), 10)
    val_ids   = set(ids2[:nv])
    train_ids = set(ids2[nv:])
    s1_tr = s1s[s1s["entity_id"].isin(train_ids)]
    s1_va = s1s[s1s["entity_id"].isin(val_ids)]
    gt_tr = gts[gts["source1_entity_id"].isin(train_ids)]
    gt_va = gts[gts["source1_entity_id"].isin(val_ids)]
    print(f"[Split] Train: {len(s1_tr):,}, Val: {len(s1_va):,}")
    return s1_tr, s1_va, gt_tr, gt_va, pool


def run_train(sample_size, resume=None):
    print("="*72)
    print(f" TRAIN  (sample={sample_size:,}, workers={N_WORKERS})")
    print("="*72)
    t_all = time.time()
    ckpt = Checkpoint("train", resume=resume, sample_size=sample_size)
    if ckpt.has("final"):
        model, threshold, f_score = ckpt.load("final")
        print(f"[Checkpoint] ✓ training already finished — validation F_0.5 = {f_score:.4f}")
        _save_model(model, threshold)
        return model, threshold

    s1_tr, s1_va, gt_tr, gt_va, pool = ckpt.stage(
        "data", lambda: _prepare_training_data(sample_size))

    # ---- 6. Blocking ----
    def _block():
        blockers = build_blockers(pool)
        tr = generate_all_candidates(s1_tr, pool, blockers)
        va = generate_all_candidates(s1_va, pool, blockers)
        blockers.release()
        return tr, va
    tr_cands, va_cands = ckpt.stage("candidates", _block)

    # Blocking recall
    gt_va_lk = parse_ground_truth(gt_va)
    found = total = 0
    for sid, truth in gt_va_lk.items():
        if truth:
            found += len(truth & set(va_cands.get(sid,[])))
            total += len(truth)
    print(f"[Block] Val blocking recall: {found/total:.4f} ({found}/{total})" if total else "[Block] no val matches")

    # ---- 7. Features + Round 1 training (with grid search) ----
    X_tr, y_tr, X_va, y_va = ckpt.stage("features", lambda: (
        *build_training_data(s1_tr, pool, gt_tr, tr_cands),
        *build_training_data(s1_va, pool, gt_va, va_cands)))

    # ---- 8. Round 1: grid search (each config checkpointed), threshold, validation ----
    def _round1():
        m = train_xgb(X_tr, y_tr, X_va, y_va, grid_search=True, checkpoint=ckpt)
        thr = find_best_threshold(m, X_va, y_va)
        return m, thr, predict_all(m, s1_va, pool, va_cands, thr)
    model_r1, threshold_r1, va_matches_r1 = ckpt.stage("round1", _round1)

    # top features
    imp = model_r1.feature_importances_
    print("\n[Feature Importance] Top 15:")
    for name, sc in sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])[:15]:
        print(f"  {name}: {sc:.4f}")

    f_score_r1 = evaluate(va_matches_r1, gt_va)
    print(f"\n  ROUND 1 Val F_0.5 = {f_score_r1:.4f}")

    # ---- 9. Hard negative mining → Round 2 (only with enough data) ----
    # Gate: hard neg mining needs enough entities to produce meaningful hard
    # negatives. With tiny samples (<10K), there are too few FPs to mine,
    # and the retrained model overfits to noise.
    if sample_size >= 10_000:
        X_r2, y_r2 = ckpt.stage("hardneg", lambda: hard_negative_mining(
            model_r1, s1_tr, pool, gt_tr, tr_cands,
        ))
    else:
        print(f"[HardNeg] Skipped — sample_size={sample_size:,} < 10K minimum")
        X_r2, y_r2 = None, None

    if X_r2 is not None and len(X_r2) > 0:
        def _round2():
            m = train_xgb(X_r2, y_r2, X_va, y_va, grid_search=False)
            thr = find_best_threshold(m, X_va, y_va)
            return m, thr, predict_all(m, s1_va, pool, va_cands, thr)
        model_r2, threshold_r2, va_matches_r2 = ckpt.stage("round2", _round2)
        f_score_r2 = evaluate(va_matches_r2, gt_va)
        print(f"  ROUND 2 Val F_0.5 = {f_score_r2:.4f}")

        # Pick the better model
        if f_score_r2 > f_score_r1:
            print("[HardNeg] ✓ Round 2 improves F₀.₅ — using round-2 model")
            model, threshold, f_score = model_r2, threshold_r2, f_score_r2
            va_matches = va_matches_r2
        else:
            print("[HardNeg] ✗ Round 2 did NOT improve — reverting to round-1 model")
            model, threshold, f_score = model_r1, threshold_r1, f_score_r1
            va_matches = va_matches_r1
    else:
        model, threshold, f_score = model_r1, threshold_r1, f_score_r1
        va_matches = va_matches_r1

    # ---- 10. Singleton detection post-processing ----
    va_matches_pre = {k: list(v) for k, v in va_matches.items()}  # copy before singleton
    va_matches_post = singleton_post_process(
        va_matches, model, s1_va, pool, va_cands,
    )
    f_score_post = evaluate(va_matches_post, gt_va)

    if f_score_post > f_score:
        print(f"[Singleton] ✓ Singleton detection improved F₀.₅: {f_score:.4f} → {f_score_post:.4f}")
        f_score = f_score_post
    else:
        print(f"[Singleton] ✗ Singleton detection did NOT improve — reverting")
        # Restore pre-singleton matches
        for k, v in va_matches_pre.items():
            va_matches[k] = v
        # Re-evaluate to confirm
        f_score = evaluate(va_matches, gt_va)

    print(f"\n{'='*72}")
    print(f"  FINAL VALIDATION F_0.5 = {f_score:.4f}   ({time.time()-t_all:.0f}s total)")
    print(f"{'='*72}\n")

    ckpt.save("final", (model, threshold, f_score))
    _save_model(model, threshold)
    return model, threshold



def _model_digest(model):
    """Identity of a trained model, so predictions from another model are never reused."""
    return hashlib.sha1(pickle.dumps(model)).hexdigest()[:16]


def _predict_country(ckpt, tag, s1_co, pool_co, model, threshold):
    """Candidates, then matches scored shard by shard — each step checkpointed."""
    def _block():
        blockers = build_blockers(pool_co)
        c = generate_all_candidates(s1_co, pool_co, blockers)
        blockers.release()
        return c
    cands = ckpt.stage(f"{tag}/candidates", _block)

    # Predict + singleton post-processing, one checkpointed shard at a time
    matches = {}
    n = (len(s1_co) + PRED_SHARD_ENTITIES - 1) // PRED_SHARD_ENTITIES
    for i in range(n):
        shard = s1_co.iloc[i * PRED_SHARD_ENTITIES:(i + 1) * PRED_SHARD_ENTITIES]

        def _score(shard=shard):
            sc = {sid: cands.get(sid, []) for sid in shard["entity_id"].values}
            m = predict_all(model, shard, pool_co, sc, threshold)
            return singleton_post_process(m, model, shard, pool_co, sc)
        print(f"[Test/{tag}] shard {i + 1}/{n}")
        matches.update(ckpt.stage(f"{tag}/shard_{i:04d}", _score))
    return cands, matches


def predict_test_by_country(model, threshold, resume=None):
    """
    Predict on the test set country-by-country to manage memory.
    For each country: filter cached pool subset, block, predict, collect results.

    Checkpointed per country: the candidates, then the scored matches in shards
    of PRED_SHARD_ENTITIES entities. Each entity's matches depend only on its own
    candidates, so shard-by-shard scoring gives the same output as one pass.
    Test S2/S3 are only loaded if some country still has work left.
    """
    print("\n" + "="*72)
    print(" TEST PREDICTION (country-by-country)")
    print("="*72)
    t0 = time.time()
    ckpt = Checkpoint("predict", resume=resume, model=_model_digest(model),
                      threshold=float(threshold))

    # Load S1 test (smallest file, cached)
    ts1 = load_source(TEST_S1)
    countries = sorted(ts1["country_norm"].unique())
    print(f"[Test] S1: {len(ts1):,} entities, countries: {countries}")

    def tag(country):
        return country or "blank"

    def n_shards(country):
        n = int((ts1["country_norm"] == country).sum())
        return (n + PRED_SHARD_ENTITIES - 1) // PRED_SHARD_ENTITIES

    def finished(country):
        return ckpt.has(f"{tag(country)}/candidates") and all(
            ckpt.has(f"{tag(country)}/shard_{i:04d}") for i in range(n_shards(country)))

    # Preprocess full test S2 + S3 ONCE (cached on disk after first run)
    test_pool_full = None
    if not all(finished(c) for c in countries):
        print("[Test] Loading full test pool (preprocessed + cached)...")
        t_pool = time.time()
        pool_parts = []
        for path in (TEST_S2, TEST_S3):
            pool_parts.append(load_source(path, use_cache=True))
        test_pool_full = pd.concat(pool_parts, ignore_index=True)
        del pool_parts; gc.collect()
        print(f"[Test] Full pool: {len(test_pool_full):,} records in {time.time()-t_pool:.1f}s")

    all_matches = {}
    all_candidates = {}

    for country in countries:
        print(f"\n--- Processing country: {country.upper()} ---")
        tc = time.time()
        s1_co = ts1[ts1["country_norm"] == country].copy()
        print(f"[Test/{country}] S1 entities: {len(s1_co):,}")

        if finished(country):
            cands = ckpt.load(f"{tag(country)}/candidates")
            matches = {}
            for i in range(n_shards(country)):
                matches.update(ckpt.load(f"{tag(country)}/shard_{i:04d}"))
            print(f"[Checkpoint] ✓ {country.upper()} already finished — loaded")
        else:
            # Filter pool for this country (fast: in-memory boolean mask)
            pool_co = test_pool_full[test_pool_full["country_norm"] == country].copy()
            print(f"[Test/{country}] Pool: {len(pool_co):,}")
            if len(pool_co) == 0:
                cands, matches = {}, {}
            else:
                cands, matches = _predict_country(ckpt, tag(country), s1_co, pool_co,
                                                  model, threshold)
            del pool_co; gc.collect()

        # Collect
        for sid in s1_co["entity_id"].values:
            all_matches[sid] = matches.get(sid, [])
            all_candidates[sid] = cands.get(sid, [])

        cands = matches = None; gc.collect()     # free before the next country
        print(f"[Test/{country}] Done in {time.time()-tc:.0f}s")

    del test_pool_full; gc.collect()

    # Write
    write_output(all_matches, all_candidates, OUTPUT_DIR)
    print(f"\n[Test] TOTAL TIME: {time.time()-t0:.0f}s")
    return all_matches, all_candidates


def run_full(sample_size=50_000, resume=None):
    model, threshold = run_train(sample_size=sample_size, resume=resume)
    predict_test_by_country(model, threshold, resume=resume)


def load_model(path=MODEL_PATH):
    """Load (model, threshold), refusing a model built for different inputs.

    The pickle records the preprocessing version and feature list it was
    trained with; a model from before a preprocessing change (e.g. enabling
    transliteration) would silently score differently-normalised text.
    """
    with open(path, "rb") as f:
        d = pickle.load(f)
    saved_prep = d.get("preprocess_version")
    saved_feats = d.get("feature_names")
    if saved_prep != PREPROCESS_VERSION or saved_feats != list(FEATURE_NAMES):
        raise RuntimeError(
            f"Saved model at {path} does not match the current code "
            f"(preprocessing {saved_prep!r} vs {PREPROCESS_VERSION!r}, "
            f"features {'match' if saved_feats == list(FEATURE_NAMES) else 'differ'}). "
            "Retrain with --mode full (or TRAIN_MODEL = True in the notebook)."
        )
    print(f"[Model] loaded, threshold={d['threshold']:.4f}")
    return d["model"], d["threshold"]


def run_predict(resume=None):
    model, threshold = load_model()
    predict_test_by_country(model, threshold, resume=resume)
