"""End-to-end orchestration: train, full (train + test), and predict modes."""
import gc
import os
import pickle
import time

import numpy as np
import pandas as pd

from .blocking import build_blockers, generate_all_candidates
from .config import (
    MODEL_PATH, N_WORKERS, OUTPUT_DIR, RANDOM_SEED, TEST_S1, TEST_S2, TEST_S3,
    TRAIN_GT, TRAIN_S1, TRAIN_S2, TRAIN_S3, VAL_FRACTION,
)
from .data import load_pool_sampled, load_source, parse_ground_truth, write_output
from .evaluation import evaluate
from .features import FEATURE_NAMES
from .inference import predict_all, singleton_post_process
from .training import (
    build_training_data, find_best_threshold, hard_negative_mining, train_xgb,
)


def run_train(sample_size):
    print("="*72)
    print(f" TRAIN  (sample={sample_size:,}, workers={N_WORKERS})")
    print("="*72)
    t_all = time.time()

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

    # ---- 6. Blocking ----
    blockers = build_blockers(pool)
    tr_cands  = generate_all_candidates(s1_tr, pool, blockers)
    va_cands  = generate_all_candidates(s1_va, pool, blockers)
    blockers.release()

    # Blocking recall
    gt_va_lk = parse_ground_truth(gt_va)
    found = total = 0
    for sid, truth in gt_va_lk.items():
        if truth:
            found += len(truth & set(va_cands.get(sid,[])))
            total += len(truth)
    print(f"[Block] Val blocking recall: {found/total:.4f} ({found}/{total})" if total else "[Block] no val matches")

    # ---- 7. Features + Round 1 training (with grid search) ----
    X_tr, y_tr = build_training_data(s1_tr, pool, gt_tr, tr_cands)
    X_va, y_va = build_training_data(s1_va, pool, gt_va, va_cands)
    model_r1 = train_xgb(X_tr, y_tr, X_va, y_va, grid_search=True)

    # top features
    imp = model_r1.feature_importances_
    print("\n[Feature Importance] Top 15:")
    for name, sc in sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])[:15]:
        print(f"  {name}: {sc:.4f}")

    # ---- 8. Round 1 threshold + evaluation ----
    threshold_r1 = find_best_threshold(model_r1, X_va, y_va)
    va_matches_r1 = predict_all(model_r1, s1_va, pool, va_cands, threshold_r1)
    f_score_r1 = evaluate(va_matches_r1, gt_va)
    print(f"\n  ROUND 1 Val F_0.5 = {f_score_r1:.4f}")

    # ---- 9. Hard negative mining → Round 2 (only with enough data) ----
    # Gate: hard neg mining needs enough entities to produce meaningful hard
    # negatives. With tiny samples (<10K), there are too few FPs to mine,
    # and the retrained model overfits to noise.
    if sample_size >= 10_000:
        X_r2, y_r2 = hard_negative_mining(
            model_r1, s1_tr, pool, gt_tr, tr_cands,
        )
    else:
        print(f"[HardNeg] Skipped — sample_size={sample_size:,} < 10K minimum")
        X_r2, y_r2 = None, None

    if X_r2 is not None and len(X_r2) > 0:
        model_r2 = train_xgb(X_r2, y_r2, X_va, y_va, grid_search=False)
        threshold_r2 = find_best_threshold(model_r2, X_va, y_va)
        va_matches_r2 = predict_all(model_r2, s1_va, pool, va_cands, threshold_r2)
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

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "threshold": threshold}, f)
    print(f"[Model] Saved -> {MODEL_PATH}")
    return model, threshold



def predict_test_by_country(model, threshold):
    """
    Predict on the test set country-by-country to manage memory.
    For each country: filter cached pool subset, block, predict, collect results.

    OPTIMIZATION: Test S2/S3 are preprocessed ONCE (and cached), then filtered
    per country in-memory. Previously we re-read + re-preprocessed the full
    TSVs for EACH country (3× redundant I/O + regex work).
    """
    print("\n" + "="*72)
    print(" TEST PREDICTION (country-by-country)")
    print("="*72)
    t0 = time.time()

    # Load S1 test (smallest file, cached)
    ts1 = load_source(TEST_S1)
    countries = sorted(ts1["country_norm"].unique())
    print(f"[Test] S1: {len(ts1):,} entities, countries: {countries}")

    # Preprocess full test S2 + S3 ONCE (cached on disk after first run)
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

        # Filter pool for this country (fast: in-memory boolean mask)
        pool_co = test_pool_full[test_pool_full["country_norm"] == country].copy()
        print(f"[Test/{country}] Pool: {len(pool_co):,}")

        if len(pool_co) == 0:
            for sid in s1_co["entity_id"].values:
                all_matches[sid] = []
                all_candidates[sid] = []
            continue

        # Block
        blockers = build_blockers(pool_co)
        cands = generate_all_candidates(s1_co, pool_co, blockers)
        blockers.release()

        # Predict
        matches = predict_all(model, s1_co, pool_co, cands, threshold)

        # Singleton post-processing
        matches = singleton_post_process(matches, model, s1_co, pool_co, cands)

        # Collect
        for sid in s1_co["entity_id"].values:
            all_matches[sid] = matches.get(sid, [])
            all_candidates[sid] = cands.get(sid, [])

        del pool_co, blockers, cands, matches; gc.collect()
        print(f"[Test/{country}] Done in {time.time()-tc:.0f}s")

    del test_pool_full; gc.collect()

    # Write
    write_output(all_matches, all_candidates, OUTPUT_DIR)
    print(f"\n[Test] TOTAL TIME: {time.time()-t0:.0f}s")
    return all_matches, all_candidates


def run_full(sample_size=50_000):
    model, threshold = run_train(sample_size=sample_size)
    predict_test_by_country(model, threshold)


def run_predict():
    with open(MODEL_PATH, "rb") as f:
        d = pickle.load(f)
    model, threshold = d["model"], d["threshold"]
    print(f"[Model] loaded, threshold={threshold:.4f}")
    predict_test_by_country(model, threshold)
