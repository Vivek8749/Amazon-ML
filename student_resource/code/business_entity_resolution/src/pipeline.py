#!/usr/bin/env python3
"""
Main pipeline for Business Entity Resolution.
Runs the complete workflow: data loading → blocking → feature engineering → 
model training → prediction → output generation.

Usage:
    python -m src.pipeline --mode train        # Train on a sample & evaluate
    python -m src.pipeline --mode full         # Full training + test prediction
    python -m src.pipeline --mode predict      # Predict on test set using saved model
"""
import argparse
import os
import sys
import time
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    MATCHING_RESULTS, CANDIDATE_PAIRS,
    MODEL_DIR, VAL_FRACTION, RANDOM_SEED,
    NEG_POS_RATIO, TFIDF_TOP_K,
)
from src.blocking import load_and_preprocess, generate_candidates, TFIDFBlocker
from src.matcher import (
    build_training_pairs, train_model, optimise_threshold,
    predict_matches, save_model, load_model,
)
from src.evaluation import evaluate
from src.io_utils import write_matching_results, write_candidate_pairs, load_ground_truth


def sample_data(s1_df, s2_df, s3_df, gt_df, n_s1=50_000, seed=RANDOM_SEED):
    """
    Sample a subset of data for faster iteration.
    Takes n_s1 entities from Source 1 and all their candidates from S2/S3.
    """
    rng = np.random.RandomState(seed)
    sampled_s1_ids = rng.choice(s1_df["entity_id"].values, size=min(n_s1, len(s1_df)), replace=False)
    sampled_s1_ids = set(sampled_s1_ids)

    s1_sample = s1_df[s1_df["entity_id"].isin(sampled_s1_ids)].copy()
    gt_sample = gt_df[gt_df["source1_entity_id"].isin(sampled_s1_ids)].copy()

    # Find all S2/S3 IDs referenced in ground truth
    referenced_ids = set()
    for _, row in gt_sample.iterrows():
        matched = row.get("matched_entity_ids", "")
        if pd.notna(matched) and matched != "":
            referenced_ids.update(str(matched).split(","))

    # Keep all S2/S3 records from the same countries as sampled S1
    s1_countries = set(s1_sample["country_norm"].unique())
    s2_sample = s2_df[s2_df["country_norm"].isin(s1_countries)].copy()
    s3_sample = s3_df[s3_df["country_norm"].isin(s1_countries)].copy()

    print(f"[Sample] S1: {len(s1_sample)}, S2: {len(s2_sample)}, S3: {len(s3_sample)}, GT: {len(gt_sample)}")
    return s1_sample, s2_sample, s3_sample, gt_sample


def train_and_validate(s1_df, pool_df, gt_df, n_s1_train=30_000):
    """
    Train/validation split, blocking, feature engineering, model training,
    threshold optimisation, and evaluation.
    """
    # Split S1 entities into train/val
    rng = np.random.RandomState(RANDOM_SEED)
    s1_ids = s1_df["entity_id"].values
    rng.shuffle(s1_ids)

    n_val = int(len(s1_ids) * VAL_FRACTION)
    val_s1_ids = set(s1_ids[:n_val])
    train_s1_ids = set(s1_ids[n_val:])

    s1_train = s1_df[s1_df["entity_id"].isin(train_s1_ids)]
    s1_val = s1_df[s1_df["entity_id"].isin(val_s1_ids)]
    gt_train = gt_df[gt_df["source1_entity_id"].isin(train_s1_ids)]
    gt_val = gt_df[gt_df["source1_entity_id"].isin(val_s1_ids)]

    print(f"[Split] Train S1: {len(s1_train)}, Val S1: {len(s1_val)}")

    # ─── Blocking ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 1: BLOCKING / CANDIDATE GENERATION")
    print("="*60)

    t0 = time.time()
    train_candidates, blocker = generate_candidates(s1_train, pool_df, top_k=TFIDF_TOP_K)
    val_candidates, _ = generate_candidates(s1_val, pool_df, tfidf_blocker=blocker, top_k=TFIDF_TOP_K)
    print(f"[Blocking] Done in {time.time()-t0:.1f}s")

    # Check blocking recall on validation
    gt_val_lookup = {}
    for _, row in gt_val.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        if pd.notna(matched) and matched != "":
            gt_val_lookup[s1_id] = set(str(matched).split(","))
        else:
            gt_val_lookup[s1_id] = set()

    found = 0
    total = 0
    for s1_id, true_matches in gt_val_lookup.items():
        if not true_matches:
            continue
        cands = set(val_candidates.get(s1_id, []))
        found += len(true_matches & cands)
        total += len(true_matches)

    blocking_recall = found / total if total > 0 else 0.0
    print(f"[Blocking] Validation blocking recall: {blocking_recall:.4f} ({found}/{total})")

    # ─── Feature Engineering & Training ───────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 2: FEATURE ENGINEERING & MODEL TRAINING")
    print("="*60)

    t0 = time.time()
    X_train, y_train, train_pairs = build_training_pairs(
        s1_train, pool_df, gt_train, train_candidates, neg_pos_ratio=NEG_POS_RATIO
    )
    X_val, y_val, val_pairs = build_training_pairs(
        s1_val, pool_df, gt_val, val_candidates, neg_pos_ratio=NEG_POS_RATIO
    )
    print(f"[Features] Done in {time.time()-t0:.1f}s")

    # Train model
    model = train_model(X_train, y_train, X_val, y_val)

    # Feature importance
    importances = model.feature_importances_
    from src.features import FEATURE_NAMES
    sorted_feats = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    print("\n[Features] Top 15 features:")
    for name, imp in sorted_feats[:15]:
        print(f"  {name}: {imp:.4f}")

    # ─── Threshold Optimisation ───────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 3: THRESHOLD OPTIMISATION")
    print("="*60)

    threshold = optimise_threshold(model, X_val, y_val, beta=0.5)

    # ─── Evaluation ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 4: VALIDATION EVALUATION")
    print("="*60)

    val_matches = predict_matches(model, s1_val, pool_df, val_candidates, threshold)
    eval_result = evaluate(val_matches, gt_val, beta=0.5)

    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS: F_0.5 = {eval_result['f_beta']:.4f}")
    print(f"{'='*60}\n")

    # Save model
    save_model(model, threshold)

    return model, threshold, blocker, eval_result


def predict_on_test(model, threshold, blocker=None):
    """Run predictions on the full test set."""
    print("\n" + "="*60)
    print("STAGE 5: TEST SET PREDICTION")
    print("="*60)

    # Load test data
    print("[Test] Loading test data...")
    t0 = time.time()
    test_s1 = load_and_preprocess(TEST_S1)
    test_s2 = load_and_preprocess(TEST_S2)
    test_s3 = load_and_preprocess(TEST_S3)
    print(f"[Test] Loaded in {time.time()-t0:.1f}s")
    print(f"[Test] S1: {len(test_s1)}, S2: {len(test_s2)}, S3: {len(test_s3)}")

    # Pool = S2 + S3
    test_pool = pd.concat([test_s2, test_s3], ignore_index=True)
    del test_s2, test_s3
    gc.collect()

    # Blocking
    print("[Test] Running blocking...")
    t0 = time.time()
    if blocker is not None:
        # Re-fit blocker on test pool
        test_blocker = TFIDFBlocker(top_k=TFIDF_TOP_K)
        test_blocker.fit(test_pool)
        test_candidates, _ = generate_candidates(test_s1, test_pool, tfidf_blocker=test_blocker)
    else:
        test_candidates, test_blocker = generate_candidates(test_s1, test_pool, top_k=TFIDF_TOP_K)
    print(f"[Test] Blocking done in {time.time()-t0:.1f}s")

    # Prediction
    print("[Test] Running matching model...")
    t0 = time.time()
    test_matches = predict_matches(model, test_s1, test_pool, test_candidates, threshold)
    print(f"[Test] Prediction done in {time.time()-t0:.1f}s")

    # Ensure every S1 entity is present
    for s1_id in test_s1["entity_id"].values:
        if s1_id not in test_matches:
            test_matches[s1_id] = []
        if s1_id not in test_candidates:
            test_candidates[s1_id] = []

    # Write outputs
    write_matching_results(test_matches, MATCHING_RESULTS)
    write_candidate_pairs(test_candidates, CANDIDATE_PAIRS)

    print(f"\n{'='*60}")
    print("TEST SET PREDICTION COMPLETE")
    print(f"  matching_results.tsv: {MATCHING_RESULTS}")
    print(f"  candidate_pairs.tsv: {CANDIDATE_PAIRS}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    parser.add_argument(
        "--mode", choices=["train", "full", "predict"], default="train",
        help="train=sample train+eval, full=full train+test predict, predict=load model and predict"
    )
    parser.add_argument(
        "--sample-size", type=int, default=50_000,
        help="Number of S1 entities to sample in 'train' mode"
    )
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"ENTITY RESOLUTION PIPELINE - Mode: {args.mode}")
    print(f"{'='*60}\n")

    if args.mode in ("train", "full"):
        # Load training data
        print("[Data] Loading training data...")
        t0 = time.time()

        if args.mode == "train":
            # Load all data but sample S1
            train_s1 = load_and_preprocess(TRAIN_S1)
            train_s2 = load_and_preprocess(TRAIN_S2)
            train_s3 = load_and_preprocess(TRAIN_S3)
            gt_df = load_ground_truth(TRAIN_GT)

            print(f"[Data] Loaded in {time.time()-t0:.1f}s")

            # Sample for faster iteration
            s1_df, s2_df, s3_df, gt_sample = sample_data(
                train_s1, train_s2, train_s3, gt_df, n_s1=args.sample_size
            )
            del train_s1
            gc.collect()

            pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
            del s2_df, s3_df
            gc.collect()

            model, threshold, blocker, eval_result = train_and_validate(s1_df, pool_df, gt_sample)

        else:  # full mode — sample training data then predict on test
            train_s1 = load_and_preprocess(TRAIN_S1)
            train_s2 = load_and_preprocess(TRAIN_S2)
            train_s3 = load_and_preprocess(TRAIN_S3)
            gt_df = load_ground_truth(TRAIN_GT)

            print(f"[Data] Loaded in {time.time()-t0:.1f}s")

            # Sample for tractable training (full S2+S3 concat would OOM)
            s1_df, s2_df, s3_df, gt_sample = sample_data(
                train_s1, train_s2, train_s3, gt_df, n_s1=args.sample_size
            )
            del train_s1, train_s2, train_s3
            gc.collect()

            pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
            del s2_df, s3_df
            gc.collect()

            model, threshold, blocker, eval_result = train_and_validate(s1_df, pool_df, gt_sample)

            # Now predict on test
            predict_on_test(model, threshold, blocker)

    elif args.mode == "predict":
        model, threshold = load_model()
        predict_on_test(model, threshold)


if __name__ == "__main__":
    main()
