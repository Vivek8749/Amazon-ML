"""
XGBoost-based matching model with F0.5-optimised threshold selection.
"""
import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

from .config import XGB_PARAMS, DEFAULT_MATCH_THRESHOLD, MODEL_DIR
from .features import compute_features, FEATURE_NAMES


def _row_to_dict(row: pd.Series) -> dict:
    """Convert a DataFrame row to a dict for feature computation."""
    return {
        "name_norm": row.get("name_norm", ""),
        "addr_norm": row.get("addr_norm", ""),
        "name_tokens": row.get("name_tokens", set()),
        "addr_tokens": row.get("addr_tokens", set()),
        "addr_nums": row.get("addr_nums", set()),
        "country_norm": row.get("country_norm", ""),
        "business_name": row.get("business_name", ""),
        "business_address": row.get("business_address", ""),
    }


def build_training_pairs(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    candidates: dict,
    neg_pos_ratio: int = 3,
) -> tuple:
    """
    Build training feature matrix from candidate pairs.
    
    For each S1 entity in ground_truth:
    - Positive pairs: candidates that are actual matches
    - Negative pairs: candidates that are NOT matches (sampled)
    
    Returns: (X, y, pair_ids)
    """
    print("[Training] Building training pairs...")

    # Parse ground truth into a lookup
    gt_lookup = {}
    for _, row in ground_truth.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        if pd.isna(matched) or matched == "":
            gt_lookup[s1_id] = set()
        else:
            gt_lookup[s1_id] = set(str(matched).split(","))

    # Index pool records
    pool_index = {}
    for _, row in pool_df.iterrows():
        pool_index[row["entity_id"]] = _row_to_dict(row)

    # Index S1 records
    s1_index = {}
    for _, row in s1_df.iterrows():
        s1_index[row["entity_id"]] = _row_to_dict(row)

    all_features = []
    all_labels = []
    all_pairs = []

    s1_ids_with_candidates = [
        s1_id for s1_id in candidates if s1_id in s1_index and s1_id in gt_lookup
    ]

    for s1_id in tqdm(s1_ids_with_candidates, desc="Building pairs"):
        s1_rec = s1_index[s1_id]
        true_matches = gt_lookup.get(s1_id, set())
        cands = candidates.get(s1_id, [])

        if not cands:
            continue

        # Separate positives and negatives
        pos_cands = [c for c in cands if c in true_matches and c in pool_index]
        neg_cands = [c for c in cands if c not in true_matches and c in pool_index]

        # Sample negatives
        rng = np.random.RandomState(hash(s1_id) % (2**31))
        max_neg = max(len(pos_cands) * neg_pos_ratio, 1) if pos_cands else min(len(neg_cands), 2)
        if len(neg_cands) > max_neg:
            neg_cands = rng.choice(neg_cands, size=max_neg, replace=False).tolist()

        # Compute features
        for c_id in pos_cands:
            feats = compute_features(s1_rec, pool_index[c_id])
            all_features.append([feats[fn] for fn in FEATURE_NAMES])
            all_labels.append(1)
            all_pairs.append((s1_id, c_id))

        for c_id in neg_cands:
            feats = compute_features(s1_rec, pool_index[c_id])
            all_features.append([feats[fn] for fn in FEATURE_NAMES])
            all_labels.append(0)
            all_pairs.append((s1_id, c_id))

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    print(f"[Training] Built {len(y)} pairs: {y.sum()} positives, {(1-y).sum()} negatives")
    return X, y, all_pairs


def train_model(X_train, y_train, X_val=None, y_val=None) -> XGBClassifier:
    """Train XGBoost classifier."""
    print("[Training] Training XGBoost model...")

    # Compute scale_pos_weight for class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight

    model = XGBClassifier(**params)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
    else:
        model.fit(X_train, y_train, verbose=50)

    print("[Training] Model trained successfully.")
    return model


def optimise_threshold(model: XGBClassifier, X_val, y_val, beta=0.5) -> float:
    """
    Find the threshold that maximises F_beta on the validation set.
    """
    y_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

    # Compute F_beta for each threshold
    f_scores = []
    for p, r in zip(precisions[:-1], recalls[:-1]):
        if p + r == 0:
            f_scores.append(0.0)
        else:
            fb = (1 + beta**2) * p * r / (beta**2 * p + r)
            f_scores.append(fb)

    f_scores = np.array(f_scores)
    best_idx = np.argmax(f_scores)
    best_threshold = thresholds[best_idx]
    best_f = f_scores[best_idx]

    print(f"[Threshold] Best F_{beta}: {best_f:.4f} at threshold {best_threshold:.4f}")
    print(f"[Threshold] Precision: {precisions[best_idx]:.4f}, Recall: {recalls[best_idx]:.4f}")

    return float(best_threshold)


def predict_matches(
    model: XGBClassifier,
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    candidates: dict,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> dict:
    """
    Predict matches for all S1 entities using the trained model.
    
    Returns: {s1_entity_id: [list of matched entity_ids]}
    """
    print(f"[Predict] Predicting matches for {len(candidates)} S1 entities at threshold={threshold:.4f}...")

    # Index pool records
    pool_index = {}
    for _, row in pool_df.iterrows():
        pool_index[row["entity_id"]] = _row_to_dict(row)

    # Index S1 records
    s1_index = {}
    for _, row in s1_df.iterrows():
        s1_index[row["entity_id"]] = _row_to_dict(row)

    matches = {}
    batch_features = []
    batch_pairs = []
    batch_s1_ids = []

    for s1_id in tqdm(candidates, desc="Computing features"):
        if s1_id not in s1_index:
            matches[s1_id] = []
            continue

        s1_rec = s1_index[s1_id]
        cands = candidates.get(s1_id, [])

        for c_id in cands:
            if c_id not in pool_index:
                continue
            feats = compute_features(s1_rec, pool_index[c_id])
            batch_features.append([feats[fn] for fn in FEATURE_NAMES])
            batch_pairs.append((s1_id, c_id))
            batch_s1_ids.append(s1_id)

    if not batch_features:
        return {s1_id: [] for s1_id in candidates}

    # Predict in bulk
    print(f"[Predict] Scoring {len(batch_features)} pairs...")
    X = np.array(batch_features, dtype=np.float32)
    y_proba = model.predict_proba(X)[:, 1]

    # Group by S1 entity
    for i, (s1_id, c_id) in enumerate(batch_pairs):
        if y_proba[i] >= threshold:
            if s1_id not in matches:
                matches[s1_id] = []
            matches[s1_id].append(c_id)

    # Ensure all S1 entities are present (even singletons)
    for s1_id in candidates:
        if s1_id not in matches:
            matches[s1_id] = []

    n_matched = sum(1 for v in matches.values() if v)
    n_singleton = sum(1 for v in matches.values() if not v)
    total_matches = sum(len(v) for v in matches.values())
    print(f"[Predict] {n_matched} entities with matches, {n_singleton} singletons, {total_matches} total match pairs")

    return matches


def save_model(model: XGBClassifier, threshold: float, path: str = None):
    """Save model and threshold."""
    if path is None:
        path = os.path.join(MODEL_DIR, "xgb_matcher.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "threshold": threshold}, f)
    print(f"[Model] Saved to {path}")


def load_model(path: str = None) -> tuple:
    """Load model and threshold."""
    if path is None:
        path = os.path.join(MODEL_DIR, "xgb_matcher.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["threshold"]
