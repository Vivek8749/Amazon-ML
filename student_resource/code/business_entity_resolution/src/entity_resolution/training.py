"""Training-pair assembly, XGBoost training/grid search, thresholding, hard-negative mining."""
import time

import numpy as np
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
from xgboost import XGBClassifier

from .config import (
    HARD_NEG_MAX_RATIO, HARD_NEG_MIX_RATIO, HARD_NEG_SCORE_FLOOR,
    N_WORKERS, NEG_POS_RATIO, RANDOM_SEED,
)
from .data import parse_ground_truth
from .features import parallel_compute_features_ordered

# ===== TRAINING DATA ==========================================================

def build_training_data(s1_df, pool_df, gt_df, candidates, neg_ratio=NEG_POS_RATIO):
    """Build (X, y) from candidate pairs + ground truth, using parallel features."""
    print("[Train] Assembling pair tuples...")
    t0 = time.time()

    gt_lookup = parse_ground_truth(gt_df)

    # Index records
    pool_name = dict(zip(pool_df["entity_id"], pool_df["name_clean"]))
    pool_addr = dict(zip(pool_df["entity_id"], pool_df["addr_clean"]))
    s1_name   = dict(zip(s1_df["entity_id"],   s1_df["name_clean"]))
    s1_addr   = dict(zip(s1_df["entity_id"],   s1_df["addr_clean"]))

    pairs_data = []   # (n1, a1, n2, a2)
    labels     = []
    rng = np.random.RandomState(RANDOM_SEED)

    for s1_id, cands in tqdm(candidates.items(), desc="Pair Assembly", unit="entity"):
        if s1_id not in s1_name or s1_id not in gt_lookup:
            continue
        n1, a1 = s1_name[s1_id], s1_addr[s1_id]
        truth  = gt_lookup[s1_id]

        pos = [c for c in cands if c in truth  and c in pool_name]
        neg = [c for c in cands if c not in truth and c in pool_name]

        max_neg = max(len(pos) * neg_ratio, 2)
        if len(neg) > max_neg:
            neg = rng.choice(neg, size=max_neg, replace=False).tolist()

        for cid in pos:
            pairs_data.append((n1, a1, pool_name[cid], pool_addr[cid]))
            labels.append(1)
        for cid in neg:
            pairs_data.append((n1, a1, pool_name[cid], pool_addr[cid]))
            labels.append(0)

    print(f"[Train] {len(labels):,} pairs assembled in {time.time()-t0:.1f}s")

    # Parallel feature computation
    X = parallel_compute_features_ordered(pairs_data, desc="Train features")
    y = np.array(labels, dtype=np.int32)
    print(f"[Train] Pos: {y.sum():,}, Neg: {(1-y).sum():,}")
    return X, y


# ===== MODEL ===================================================================

# XGBoost hyperparameter grid for tuning
XGB_GRID = [
    {"max_depth": 8,  "learning_rate": 0.05, "n_estimators": 800},
    {"max_depth": 10, "learning_rate": 0.05, "n_estimators": 800},
    {"max_depth": 10, "learning_rate": 0.03, "n_estimators": 1200},
    {"max_depth": 12, "learning_rate": 0.05, "n_estimators": 600},
    {"max_depth": 8,  "learning_rate": 0.1,  "n_estimators": 500},
    {"max_depth": 10, "learning_rate": 0.1,  "n_estimators": 500},
]


def train_xgb(X_train, y_train, X_val=None, y_val=None, grid_search=True):
    """Train XGBoost with optional grid search over depth/lr/n_estimators.

    If grid_search=True and X_val is provided, trains all configs in XGB_GRID,
    picks the one with the best validation F₀.₅. Otherwise uses the first config.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / max(n_pos, 1)

    base_params = dict(
        objective="binary:logistic", eval_metric="logloss",
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=spw,
        tree_method="hist",
        n_jobs=N_WORKERS, random_state=RANDOM_SEED,
        early_stopping_rounds=50,
    )

    if not grid_search or X_val is None:
        # Single config: use first grid entry
        cfg = XGB_GRID[0]
        params = {**base_params, **cfg}
        model = XGBClassifier(**params, device="cuda")
        print(f"[XGB] Training single config: {cfg}")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)] if X_val is not None else None, verbose=50)
        return model

    # Grid search
    print(f"[XGB] Grid search: {len(XGB_GRID)} configurations")
    best_model = None
    best_f05 = -1.0
    best_cfg = None
    results = []

    for i, cfg in enumerate(XGB_GRID):
        params = {**base_params, **cfg}
        model = XGBClassifier(**params, device="cuda")
        print(f"\n[XGB Grid {i+1}/{len(XGB_GRID)}] {cfg}")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

        # Evaluate with F₀.₅ on validation set
        proba = model.predict_proba(X_val)[:, 1]
        prec, rec, thresholds = precision_recall_curve(y_val, proba)
        fbeta = np.where(
            (0.25 * prec + rec) > 0,
            1.25 * prec * rec / (0.25 * prec + rec), 0.0)
        best_idx = np.argmax(fbeta[:-1])
        f05 = fbeta[best_idx]
        thresh = thresholds[best_idx]

        results.append((cfg, f05, thresh))
        print(f"  → F₀.₅={f05:.4f} @ thresh={thresh:.4f} "
              f"(P={prec[best_idx]:.4f} R={rec[best_idx]:.4f})")

        if f05 > best_f05:
            best_f05 = f05
            best_model = model
            best_cfg = cfg

    print(f"\n[XGB] Grid search results:")
    for cfg, f05, thresh in sorted(results, key=lambda x: -x[1]):
        marker = " ★" if cfg == best_cfg else ""
        print(f"  {cfg} → F₀.₅={f05:.4f}{marker}")
    print(f"[XGB] Best: {best_cfg} → F₀.₅={best_f05:.4f}")

    return best_model


def find_best_threshold(model, X_val, y_val, beta=0.5):
    proba = model.predict_proba(X_val)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val, proba)
    fbeta = np.where(
        (beta**2 * prec + rec) > 0,
        (1 + beta**2) * prec * rec / (beta**2 * prec + rec), 0.0)
    best = np.argmax(fbeta[:-1])
    print(f"[Thresh] Best F_{beta}: {fbeta[best]:.4f} @ {thresholds[best]:.4f}  "
          f"(P={prec[best]:.4f} R={rec[best]:.4f})")
    return float(thresholds[best])


# ===== HARD NEGATIVE MINING ====================================================

def hard_negative_mining(model, s1_df, pool_df, gt_df, candidates,
                         score_floor=HARD_NEG_SCORE_FLOOR,
                         mix_ratio=HARD_NEG_MIX_RATIO,
                         max_neg_ratio=HARD_NEG_MAX_RATIO):
    """Mine hard negatives from round-1 model's false positives.

    Protocol:
      1. Score all training candidate pairs with the round-1 model
      2. Hard negatives = candidates scored > score_floor AND NOT in ground truth
      3. Mix: mix_ratio% hard negatives + (1-mix_ratio)% random negatives
      4. Cap at max_neg_ratio:1 negative-to-positive ratio
      5. Recompute features for the mixed set

    Args:
        model: round-1 trained XGBClassifier
        s1_df, pool_df: DataFrames with name_clean, addr_clean columns
        gt_df: ground truth DataFrame
        candidates: {s1_id: [candidate_ids]} from blocking
        score_floor: only mine FPs scored above this
        mix_ratio: fraction of negatives that should be hard (0.0–1.0)
        max_neg_ratio: maximum overall neg:pos ratio

    Returns:
        (X_r2, y_r2): round-2 training data with hard negatives mixed in
    """
    print(f"\n{'='*60}")
    print("HARD NEGATIVE MINING (Round 2)")
    print(f"{'='*60}")
    t0 = time.time()

    gt_lookup = parse_ground_truth(gt_df)

    # Index records
    pool_name = dict(zip(pool_df["entity_id"], pool_df["name_clean"]))
    pool_addr = dict(zip(pool_df["entity_id"], pool_df["addr_clean"]))
    s1_name = dict(zip(s1_df["entity_id"], s1_df["name_clean"]))
    s1_addr = dict(zip(s1_df["entity_id"], s1_df["addr_clean"]))

    # Score all candidate pairs
    print("[HardNeg] Scoring all training candidates with round-1 model...")
    pairs_data = []
    pairs_meta = []  # (s1_id, cand_id, is_true_match)

    for s1_id, cands in tqdm(candidates.items(), desc="Pair collection", leave=False):
        if s1_id not in s1_name or s1_id not in gt_lookup:
            continue
        n1, a1 = s1_name[s1_id], s1_addr[s1_id]
        truth = gt_lookup[s1_id]
        for cid in cands:
            if cid in pool_name:
                pairs_data.append((n1, a1, pool_name[cid], pool_addr[cid]))
                pairs_meta.append((s1_id, cid, cid in truth))

    if not pairs_data:
        print("[HardNeg] No pairs to mine — skipping")
        return None, None

    X_all = parallel_compute_features_ordered(pairs_data, desc="HardNeg features")
    proba = model.predict_proba(X_all)[:, 1]

    # Separate: positives, hard negatives, random negatives
    positives = []
    hard_negs = []
    random_negs = []

    for i, (s1_id, cid, is_true) in enumerate(pairs_meta):
        if is_true:
            positives.append(i)
        elif proba[i] >= score_floor:
            hard_negs.append(i)  # false positive — scored high but not a match
        else:
            random_negs.append(i)

    n_pos = len(positives)
    n_total_neg = min(n_pos * max_neg_ratio, len(hard_negs) + len(random_negs))
    n_hard = min(int(n_total_neg * mix_ratio), len(hard_negs))
    n_random = min(int(n_total_neg * (1 - mix_ratio)), len(random_negs))

    print(f"[HardNeg] Positives: {n_pos:,}")
    print(f"[HardNeg] Hard negatives available: {len(hard_negs):,} (score >= {score_floor})")
    print(f"[HardNeg] Using: {n_hard:,} hard + {n_random:,} random negatives")

    # Sample
    rng = np.random.RandomState(RANDOM_SEED + 1)
    if len(hard_negs) > n_hard:
        hard_negs = rng.choice(hard_negs, size=n_hard, replace=False).tolist()
    else:
        hard_negs = hard_negs[:n_hard]
    if len(random_negs) > n_random:
        random_negs = rng.choice(random_negs, size=n_random, replace=False).tolist()
    else:
        random_negs = random_negs[:n_random]

    # Build round-2 training set
    selected = positives + hard_negs + random_negs
    X_r2 = X_all[selected]
    y_r2 = np.array([1] * len(positives) + [0] * (len(hard_negs) + len(random_negs)),
                     dtype=np.int32)

    print(f"[HardNeg] Round-2 data: {X_r2.shape[0]:,} samples "
          f"({y_r2.sum():,} pos, {(1-y_r2).sum():,} neg) in {time.time()-t0:.1f}s")
    return X_r2, y_r2
