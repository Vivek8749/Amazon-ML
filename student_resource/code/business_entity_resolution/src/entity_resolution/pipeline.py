"""End-to-end orchestration: train, full (train + test), and predict modes.

Every expensive stage is checkpointed (see checkpoint.py): rerunning the same
command after a killed session skips finished stages and continues.

Pair scores are kept and every match decision is made from them
(inference.decide_matches): threshold, one-record-one-entity rule and the
singleton rule. The threshold is tuned on per-entity macro F0.5 — the
leaderboard metric — on the validation entities.
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
    MODEL_PATH, N_MODELS, N_WORKERS, NEG_POS_RATIO, OUTPUT_DIR, PRED_SHARD_ENTITIES,
    RANDOM_SEED, SCORE_KEEP_FLOOR, SINGLETON_MAX_SCORE_THRESHOLD, TEST_S1, TEST_S2,
    TEST_S3, TRAIN_GT, TRAIN_POOL_EXTRA_PER_COUNTRY, TRAIN_S1, TRAIN_S2, TRAIN_S3,
    VAL_FRACTION,
)
from .data import load_pool_sampled, load_source, parse_ground_truth, write_output
from .evaluation import loss_breakdown
from .features import FEATURE_NAMES
from .inference import decide_matches, score_pairs, tune_threshold
from .preprocessing import PREPROCESS_VERSION
from .training import (
    build_training_data, hard_negative_mining, train_ensemble, train_xgb,
)


def _save_model(model, threshold):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "threshold": threshold,
                     "preprocess_version": PREPROCESS_VERSION,
                     "feature_names": list(FEATURE_NAMES)}, f)
    print(f"[Model] Saved -> {MODEL_PATH}")


def _pool_extra(sample_size, full_sample):
    """Random (non-matching) pool records per country for the training pool."""
    if TRAIN_POOL_EXTRA_PER_COUNTRY == "full":
        return 10**12                       # every record of the country
    if TRAIN_POOL_EXTRA_PER_COUNTRY is not None:
        return int(TRAIN_POOL_EXTRA_PER_COUNTRY)
    if full_sample:
        return 500_000
    return min(max(sample_size * 3, 20_000), 30_000)   # previous default


def _prepare_training_data(sample_size):
    """Steps 1-3 and 5: sample S1, collect the true-match IDs, split train/validation.

    The S2+S3 pool is not part of this checkpoint: it is rebuilt from the parquet
    cache on every run (deterministically), because the full pool is several GB.
    """
    # ---- 1. Load S1 + GT (fast, small-ish files) ----
    print("[Data] Loading S1 + GT...")
    t0 = time.time()
    s1_full = load_source(TRAIN_S1)
    gt_full = pd.read_csv(TRAIN_GT, sep="\t", dtype=str)
    print(f"[Data] S1: {len(s1_full):,}, GT: {len(gt_full):,} in {time.time()-t0:.1f}s")

    # ---- 2. Sample S1 ----
    rng = np.random.RandomState(RANDOM_SEED)
    full_sample = sample_size is None or sample_size <= 0 or sample_size >= len(s1_full)
    if full_sample:
        print(f"[Data] Using FULL dataset ({len(s1_full):,} entities)")
        s1s = s1_full.copy()
        gts = gt_full.copy()
    else:
        ids = rng.choice(s1_full["entity_id"].values,
                         size=min(sample_size, len(s1_full)), replace=False)
        s1s = s1_full[s1_full["entity_id"].isin(set(ids))].copy()
        gts = gt_full[gt_full["source1_entity_id"].isin(set(ids))].copy()
    extra = _pool_extra(sample_size, full_sample)
    countries = sorted(s1s["country_norm"].unique())
    del s1_full, gt_full; gc.collect()

    # ---- 3. Find all S2/S3 IDs referenced in GT (must-have for pool) ----
    must_have = set()
    for _, row in gts.iterrows():
        m = row.get("matched_entity_ids", "")
        if pd.notna(m) and m:
            must_have.update(str(m).split(","))
    print(f"[Data] S1 sample: {len(s1s):,}, must-have pool IDs: {len(must_have):,}")

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
    return s1_tr, s1_va, gt_tr, gt_va, must_have, countries, extra


def _tune(model, s1_va, pool, va_cands, truth_va, label):
    """Validation pair scores -> entity-level threshold and macro F0.5."""
    scores = score_pairs(model, s1_va, pool, va_cands)
    threshold, f_score = tune_threshold(scores, truth_va, s1_va["entity_id"].values)
    print(f"  {label}: validation macro F0.5 = {f_score:.4f} @ threshold {threshold:.3f}")
    return scores, threshold, f_score


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

    s1_tr, s1_va, gt_tr, gt_va, must_have, countries, extra = ckpt.stage(
        "data", lambda: _prepare_training_data(sample_size))

    # ---- 4. Training pool: true matches + random records (all, when "full") ----
    pool = load_pool_sampled(TRAIN_S2, TRAIN_S3, must_have, countries, extra_per_country=extra)

    # ---- 6. Blocking ----
    def _block():
        blockers = build_blockers(pool)
        tr = generate_all_candidates(s1_tr, pool, blockers)
        va = generate_all_candidates(s1_va, pool, blockers)
        blockers.release()
        return tr, va
    tr_cands, va_cands = ckpt.stage("candidates", _block)

    # Blocking recall
    truth_va = parse_ground_truth(gt_va)
    found = total = 0
    for sid, truth in truth_va.items():
        if truth:
            found += len(truth & set(va_cands.get(sid,[])))
            total += len(truth)
    print(f"[Block] Val blocking recall: {found/total:.4f} ({found}/{total})" if total else "[Block] no val matches")

    # ---- 7. Features (every blocked candidate when NEG_POS_RATIO is None) ----
    X_tr, y_tr, X_va, y_va = ckpt.stage("features", lambda: (
        *build_training_data(s1_tr, pool, gt_tr, tr_cands),
        *build_training_data(s1_va, pool, gt_va, va_cands)))

    # ---- 8. Round 1: grid search (each config checkpointed) -> seed ensemble ----
    best = ckpt.stage("round1", lambda: train_xgb(
        X_tr, y_tr, X_va, y_va, grid_search=True, checkpoint=ckpt))
    model = train_ensemble(best, X_tr, y_tr, X_va, y_va, N_MODELS, checkpoint=ckpt)

    # top features
    imp = model.feature_importances_
    print("\n[Feature Importance] Top 15:")
    for name, sc in sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])[:15]:
        print(f"  {name}: {sc:.4f}")

    scores_va, threshold, f_score = ckpt.stage(
        "val_scores", lambda: _tune(model, s1_va, pool, va_cands, truth_va, "ROUND 1"))

    # ---- 9. Hard negatives -> Round 2 (only when negatives were sampled) ----
    # With NEG_POS_RATIO = None every blocked negative is already a training pair.
    if NEG_POS_RATIO is not None and sample_size >= 10_000:
        X_r2, y_r2 = ckpt.stage("hardneg", lambda: hard_negative_mining(
            model, s1_tr, pool, gt_tr, tr_cands,
        ))
        if X_r2 is not None and len(X_r2) > 0:
            model_r2 = ckpt.stage("round2", lambda: train_xgb(
                X_r2, y_r2, X_va, y_va, grid_search=False))
            s2, t2, f2 = ckpt.stage(
                "val_scores_r2", lambda: _tune(model_r2, s1_va, pool, va_cands, truth_va, "ROUND 2"))
            if f2 > f_score:
                print("[HardNeg] ✓ Round 2 improves F₀.₅ — using round-2 model")
                model, scores_va, threshold, f_score = model_r2, s2, t2, f2
            else:
                print("[HardNeg] ✗ Round 2 did NOT improve — keeping round-1 model")
    else:
        print("[HardNeg] Skipped — all blocked negatives are already training pairs"
              if NEG_POS_RATIO is None else
              f"[HardNeg] Skipped — sample_size={sample_size:,} < 10K minimum")

    # ---- 10. Where the remaining validation loss comes from ----
    va_matches = decide_matches(scores_va, threshold, s1_va["entity_id"].values)
    loss_breakdown(va_matches, truth_va, va_cands)

    print(f"\n{'='*72}")
    print(f"  FINAL VALIDATION F_0.5 = {f_score:.4f} @ threshold {threshold:.3f}   "
          f"({time.time()-t_all:.0f}s total)")
    print(f"{'='*72}\n")

    ckpt.save("final", (model, threshold, f_score))
    _save_model(model, threshold)
    return model, threshold


def _model_digest(model):
    """Identity of a trained model, so scores from another model are never reused."""
    return hashlib.sha1(pickle.dumps(model)).hexdigest()[:16]


def _score_country(ckpt, tag, s1_co, pool_co, model):
    """Candidates, then pair scores shard by shard — each step checkpointed."""
    def _block():
        # Candidates in blocks of PRED_SHARD_ENTITIES entities, each checkpointed,
        # so a killed session keeps the finished blocks. Blockers are built once.
        blockers, merged = None, {}
        n_blocks = (len(s1_co) + PRED_SHARD_ENTITIES - 1) // PRED_SHARD_ENTITIES
        for i in range(n_blocks):
            block = s1_co.iloc[i * PRED_SHARD_ENTITIES:(i + 1) * PRED_SHARD_ENTITIES]

            def _cands(block=block):
                nonlocal blockers
                if blockers is None:
                    blockers = build_blockers(pool_co)
                return generate_all_candidates(block, pool_co, blockers)
            print(f"[Test/{tag}] candidate block {i + 1}/{n_blocks}")
            merged.update(ckpt.stage(f"{tag}/cands_{i:04d}", _cands))
        if blockers is not None:
            blockers.release()
        return merged
    cands = ckpt.stage(f"{tag}/candidates", _block)

    parts = []
    n = (len(s1_co) + PRED_SHARD_ENTITIES - 1) // PRED_SHARD_ENTITIES
    for i in range(n):
        shard = s1_co.iloc[i * PRED_SHARD_ENTITIES:(i + 1) * PRED_SHARD_ENTITIES]

        def _score(shard=shard):
            sc = {sid: cands.get(sid, []) for sid in shard["entity_id"].values}
            return score_pairs(model, shard, pool_co, sc, keep_floor=SCORE_KEEP_FLOOR)
        print(f"[Test/{tag}] shard {i + 1}/{n}")
        parts.append(ckpt.stage(f"{tag}/scores_{i:04d}", _score))
    return cands, parts


def predict_test_by_country(model, threshold, resume=None):
    """
    Predict on the test set country-by-country to manage memory.

    Checkpointed per country: the candidates, then the pair scores in shards of
    PRED_SHARD_ENTITIES entities. Matches are decided once a country is fully
    scored, because the one-record-one-entity rule compares every S1 entity
    that claims a record, across shards. Scores do not depend on the threshold,
    so a new threshold reuses them. Test S2/S3 are only loaded if some country
    still has scoring left.
    """
    print("\n" + "="*72)
    print(" TEST PREDICTION (country-by-country)")
    print("="*72)
    if not SCORE_KEEP_FLOOR <= min(threshold, SINGLETON_MAX_SCORE_THRESHOLD):
        raise ValueError("SCORE_KEEP_FLOOR must not exceed the threshold or the singleton cut-off")
    t0 = time.time()
    ckpt = Checkpoint("predict", resume=resume, model=_model_digest(model))

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
            ckpt.has(f"{tag(country)}/scores_{i:04d}") for i in range(n_shards(country)))

    # Preprocess full test S2 + S3 ONCE (cached on disk after first run)
    test_pool_full = None
    only_env = [c.strip().lower() for c in os.environ.get("ER_COUNTRIES", "").split(",") if c.strip()]
    if not all(finished(c) for c in countries if not only_env or c in only_env):
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

    # ER_COUNTRIES (env, comma-separated): only compute these countries in this
    # process; the others are loaded if finished. Output is written once all are.
    only = [c.strip().lower() for c in os.environ.get("ER_COUNTRIES", "").split(",") if c.strip()]
    for country in countries:
        if only and country not in only and not finished(country):
            print(f"\n--- Skipping {country.upper()} (ER_COUNTRIES={','.join(only)}) ---")
            continue
        print(f"\n--- Processing country: {country.upper()} ---")
        tc = time.time()
        s1_co = ts1[ts1["country_norm"] == country].copy()
        print(f"[Test/{country}] S1 entities: {len(s1_co):,}")

        if finished(country):
            cands = ckpt.load(f"{tag(country)}/candidates")
            parts = [ckpt.load(f"{tag(country)}/scores_{i:04d}") for i in range(n_shards(country))]
            print(f"[Checkpoint] ✓ {country.upper()} already scored — loaded")
        else:
            # Filter pool for this country (fast: in-memory boolean mask)
            pool_co = test_pool_full[test_pool_full["country_norm"] == country].copy()
            print(f"[Test/{country}] Pool: {len(pool_co):,}")
            if len(pool_co) == 0:
                cands, parts = {}, []
            else:
                cands, parts = _score_country(ckpt, tag(country), s1_co, pool_co, model)
            del pool_co; gc.collect()

        # Decide over the whole country: threshold, one-record-one-entity, singletons
        scores = (pd.concat(parts, ignore_index=True) if parts
                  else pd.DataFrame({"s1_id": [], "cand_id": [], "p": []}))
        matches = decide_matches(scores, threshold, s1_co["entity_id"].values)
        n_matched = sum(1 for v in matches.values() if v)
        print(f"[Test/{country}] {n_matched:,} of {len(s1_co):,} entities matched, "
              f"{sum(map(len, matches.values())):,} matches @ threshold {threshold:.3f}")

        # Collect
        for sid in s1_co["entity_id"].values:
            all_matches[sid] = matches.get(sid, [])
            all_candidates[sid] = cands.get(sid, [])

        cands = matches = scores = parts = None; gc.collect()     # free before the next country
        print(f"[Test/{country}] Done in {time.time()-tc:.0f}s")

    del test_pool_full; gc.collect()

    missing = len(ts1) - len(all_matches)
    if missing:
        print(f"\n[Test] {missing:,} entities belong to countries not finished yet — "
              f"output not written; rerun without ER_COUNTRIES once they are done")
        return all_matches, all_candidates

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


def run_predict(resume=None, threshold=None):
    model, saved_threshold = load_model()
    if threshold is not None:
        print(f"[Model] threshold override: {saved_threshold:.4f} -> {threshold:.4f}")
    predict_test_by_country(model, saved_threshold if threshold is None else threshold, resume=resume)


# Entities at or beyond this position of the seeded shuffle were never in a
# training sample: run_train samples with RandomState(RANDOM_SEED).choice, which
# takes a prefix of RandomState(RANDOM_SEED).permutation for any sample size.
TUNE_OFFSET = 1_000_000


def run_tune(n_entities=20_000, resume=None):
    """Re-tune a saved model's decision threshold without retraining.

    Scores `n_entities` training S1 entities the model never saw against the
    full training pool (as dense as the test pool), picks the threshold that
    maximises per-entity macro F0.5 with the one-record-one-entity rule, and
    saves the model with that threshold (the previous file is kept as a backup).
    """
    print("="*72)
    print(f" TUNE THRESHOLD  ({n_entities:,} unseen training entities, full pool)")
    print("="*72)
    t0 = time.time()
    model, old_threshold = load_model()
    ckpt = Checkpoint("tune", resume=resume, model=_model_digest(model), n_entities=n_entities)

    def _data():
        s1_full = load_source(TRAIN_S1)
        gt_full = pd.read_csv(TRAIN_GT, sep="\t", dtype=str)
        perm = np.random.RandomState(RANDOM_SEED).permutation(len(s1_full))
        pick = set(s1_full["entity_id"].values[perm[TUNE_OFFSET:TUNE_OFFSET + n_entities]])
        s1_t = s1_full[s1_full["entity_id"].isin(pick)].copy()
        gt_t = gt_full[gt_full["source1_entity_id"].isin(pick)].copy()
        must = set().union(*parse_ground_truth(gt_t).values())
        return s1_t, gt_t, must, sorted(s1_t["country_norm"].unique())
    s1_t, gt_t, must, countries = ckpt.stage("data", _data)
    truth = parse_ground_truth(gt_t)
    ids = s1_t["entity_id"].values

    pool = load_pool_sampled(TRAIN_S2, TRAIN_S3, must, countries,
                             extra_per_country=_pool_extra(n_entities, False))

    def _block():
        blockers = build_blockers(pool)
        c = generate_all_candidates(s1_t, pool, blockers)
        blockers.release()
        return c
    cands = ckpt.stage("candidates", _block)
    scores = ckpt.stage("scores", lambda: score_pairs(model, s1_t, pool, cands))
    pool = None; gc.collect()          # free the full pool before deciding

    old = decide_matches(scores, old_threshold, ids, one_to_one=False)
    print(f"\n### Current decision: threshold {old_threshold:.4f}, no one-to-one rule")
    old_report = loss_breakdown(old, truth, cands)
    new_threshold, new_f = tune_threshold(scores, truth, ids)
    print(f"\n### Tuned decision: threshold {new_threshold:.3f}, one-to-one rule")
    loss_breakdown(decide_matches(scores, new_threshold, ids), truth, cands)

    backup = MODEL_PATH.replace(".pkl", ".before_tune.pkl")
    if not os.path.exists(backup):
        os.replace(MODEL_PATH, backup)
        print(f"[Model] Previous model kept as {backup}")
    _save_model(model, new_threshold)
    ckpt.save("final", (old_threshold, old_report["score"], new_threshold, new_f))
    print(f"\n{'='*72}")
    print(f"  THRESHOLD {old_threshold:.4f} -> {new_threshold:.3f}:  macro F0.5 "
          f"{old_report['score']:.4f} -> {new_f:.4f}  on {len(ids):,} unseen entities  "
          f"({time.time()-t0:.0f}s)")
    print(f"{'='*72}\n")
    return new_threshold
