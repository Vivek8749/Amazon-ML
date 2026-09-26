"""Scoring candidate pairs and singleton post-processing."""
import time

import numpy as np
from tqdm import tqdm

from .config import SINGLETON_MAX_SCORE_THRESHOLD
from .features import compute_pair_features, parallel_compute_features_ordered

# ===== SINGLETON DETECTION =====================================================

def singleton_post_process(matches, model, s1_df, pool_df, candidates,
                           max_score_thresh=SINGLETON_MAX_SCORE_THRESHOLD):
    """Post-process matches to improve singleton detection.

    For entities where the model predicted matches, check if the best candidate
    score is suspiciously low. If all candidate scores are below max_score_thresh,
    override to singleton (empty match list). This boosts precision on entities
    that are borderline.

    For F₀.₅, correctly predicting a singleton = 1.0, false merge on singleton = 0.0.
    This step trades a small amount of recall for improved precision on marginal cases.

    Args:
        matches: {s1_id: [matched_ids]} from predict_all
        model: trained XGBClassifier
        s1_df, pool_df: DataFrames
        candidates: {s1_id: [candidate_ids]}
        max_score_thresh: if best candidate < this, force singleton

    Returns:
        updated matches dict
    """
    print("[Singleton] Running singleton detection post-processing...")
    t0 = time.time()

    pool_name = dict(zip(pool_df["entity_id"], pool_df["name_clean"]))
    pool_addr = dict(zip(pool_df["entity_id"], pool_df["addr_clean"]))
    s1_name = dict(zip(s1_df["entity_id"], s1_df["name_clean"]))
    s1_addr = dict(zip(s1_df["entity_id"], s1_df["addr_clean"]))

    n_forced_singleton = 0
    n_checked = 0

    for s1_id, matched in list(matches.items()):
        if not matched:
            continue  # already singleton

        cands = candidates.get(s1_id, [])
        if not cands or s1_id not in s1_name:
            continue

        n1, a1 = s1_name[s1_id], s1_addr[s1_id]

        # Compute features for all candidates (not just matched)
        pairs = []
        for cid in cands:
            if cid in pool_name:
                pairs.append((n1, a1, pool_name[cid], pool_addr[cid]))

        if not pairs:
            continue

        X = np.array([compute_pair_features(*p) for p in pairs], dtype=np.float32)
        scores = model.predict_proba(X)[:, 1]
        best_score = scores.max()

        n_checked += 1

        # If best candidate score is below threshold, force singleton
        if best_score < max_score_thresh:
            matches[s1_id] = []
            n_forced_singleton += 1

    print(f"[Singleton] Checked {n_checked:,} entities with matches, "
          f"forced {n_forced_singleton:,} to singleton "
          f"(best_score < {max_score_thresh}) in {time.time()-t0:.1f}s")
    return matches


# ===== PREDICTION (parallelised) ==============================================

def predict_all(model, s1_df, pool_df, candidates, threshold):
    """Score all candidate pairs with parallel feature computation."""
    print(f"[Pred] Building pair tuples...")
    t0 = time.time()

    pool_name = dict(zip(pool_df["entity_id"], pool_df["name_clean"]))
    pool_addr = dict(zip(pool_df["entity_id"], pool_df["addr_clean"]))
    s1_name   = dict(zip(s1_df["entity_id"],   s1_df["name_clean"]))
    s1_addr   = dict(zip(s1_df["entity_id"],   s1_df["addr_clean"]))

    pairs_data = []
    pairs_ids  = []    # (s1_id, cand_id)

    for s1_id, cands in tqdm(candidates.items(), desc="Pred Pair Assembly", unit="entity", leave=False):
        if s1_id not in s1_name:
            continue
        n1, a1 = s1_name[s1_id], s1_addr[s1_id]
        for cid in cands:
            if cid in pool_name:
                pairs_data.append((n1, a1, pool_name[cid], pool_addr[cid]))
                pairs_ids.append((s1_id, cid))

    print(f"[Pred] {len(pairs_data):,} pairs in {time.time()-t0:.1f}s")

    # Parallel features
    X = parallel_compute_features_ordered(pairs_data, desc="Pred features")

    # Score in bulk
    print(f"[Pred] Scoring {X.shape[0]:,} pairs @ threshold={threshold:.4f}...")
    proba = model.predict_proba(X)[:, 1]

    matches = {}
    for i, (s1_id, cid) in enumerate(pairs_ids):
        if proba[i] >= threshold:
            matches.setdefault(s1_id, []).append(cid)

    # Ensure all S1 present
    for sid in s1_df["entity_id"].values:
        if sid not in matches:
            matches[sid] = []

    nm = sum(1 for v in matches.values() if v)
    tm = sum(len(v) for v in matches.values())
    print(f"[Pred] {nm:,} matched entities, {tm:,} total matches")
    return matches
