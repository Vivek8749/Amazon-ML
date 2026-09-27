"""Scoring candidate pairs and turning pair scores into final match sets.

Pairs are scored once (`score_pairs`) and every decision is made from those
scores (`decide_matches`), so thresholds and decision rules can be re-tuned
without recomputing features.
"""
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import ONE_TO_ONE, SINGLETON_MAX_SCORE_THRESHOLD
from .evaluation import macro_f05
from .features import parallel_compute_features_ordered


# ===== SCORING (parallelised) ==================================================

def score_pairs(model, s1_df, pool_df, candidates, keep_floor=0.0):
    """Model probability for every (S1, candidate) pair.

    Returns a DataFrame [s1_id, cand_id, p]. Pairs scoring below `keep_floor`
    are dropped (used at test time to keep checkpoints small; any floor at or
    below both the threshold and the singleton cut-off leaves decisions unchanged).
    """
    print("[Pred] Building pair tuples...")
    t0 = time.time()

    pool_name = dict(zip(pool_df["entity_id"], pool_df["name_clean"]))
    pool_addr = dict(zip(pool_df["entity_id"], pool_df["addr_clean"]))
    s1_name   = dict(zip(s1_df["entity_id"],   s1_df["name_clean"]))
    s1_addr   = dict(zip(s1_df["entity_id"],   s1_df["addr_clean"]))

    pairs_data = []
    s1_ids, cand_ids = [], []

    for s1_id, cands in tqdm(candidates.items(), desc="Pred Pair Assembly", unit="entity", leave=False):
        if s1_id not in s1_name:
            continue
        n1, a1 = s1_name[s1_id], s1_addr[s1_id]
        for cid in cands:
            if cid in pool_name:
                pairs_data.append((n1, a1, pool_name[cid], pool_addr[cid]))
                s1_ids.append(s1_id)
                cand_ids.append(cid)

    print(f"[Pred] {len(pairs_data):,} pairs in {time.time()-t0:.1f}s")
    X = parallel_compute_features_ordered(pairs_data, desc="Pred features")
    del pairs_data
    proba = model.predict_proba(X)[:, 1].astype(np.float32) if len(X) else np.empty(0, np.float32)

    scores = pd.DataFrame({"s1_id": s1_ids, "cand_id": cand_ids, "p": proba})
    if keep_floor > 0:
        scores = scores[scores["p"] >= keep_floor].reset_index(drop=True)
    return scores


# ===== DECISION ================================================================

def decide_matches(scores, threshold, s1_ids, one_to_one=ONE_TO_ONE,
                   min_best=SINGLETON_MAX_SCORE_THRESHOLD):
    """Final match set for every S1 entity in `s1_ids`.

    1. keep pairs with p >= threshold;
    2. one_to_one: each S2/S3 record keeps only its highest-scoring S1 entity —
       in the training ground truth no record belongs to two S1 entities, so
       every additional claim on a record is a certain false positive;
    3. singleton rule: an entity whose best candidate scores below `min_best`
       gets no matches.
    """
    kept = scores[scores["p"] >= threshold]
    if one_to_one and len(kept):
        kept = (kept.sort_values(["p", "s1_id"], ascending=[False, True], kind="stable")
                    .drop_duplicates("cand_id", keep="first"))
    if min_best > 0 and len(kept):
        best = scores.groupby("s1_id")["p"].max()
        kept = kept[kept["s1_id"].map(best) >= min_best]
    grouped = kept.groupby("s1_id")["cand_id"].agg(list).to_dict() if len(kept) else {}
    return {sid: sorted(grouped.get(sid, [])) for sid in s1_ids}


def predict_all(model, s1_df, pool_df, candidates, threshold, **decide_kwargs):
    """Score all candidate pairs and decide the final matches."""
    scores = score_pairs(model, s1_df, pool_df, candidates)
    matches = decide_matches(scores, threshold, s1_df["entity_id"].values, **decide_kwargs)
    nm = sum(1 for v in matches.values() if v)
    tm = sum(len(v) for v in matches.values())
    print(f"[Pred] {nm:,} matched entities, {tm:,} total matches @ threshold={threshold:.4f}")
    return matches


def tune_threshold(scores, truth, s1_ids, one_to_one=ONE_TO_ONE,
                   min_best=SINGLETON_MAX_SCORE_THRESHOLD,
                   grid=np.round(np.concatenate([np.arange(0.05, 0.95, 0.01),
                                                 np.arange(0.95, 0.9951, 0.005)]), 3)):
    """Threshold that maximises macro F0.5 per S1 entity — the leaderboard metric —
    with the full decision rule applied (not pair-level F0.5 on sampled pairs)."""
    results = [(t, macro_f05(decide_matches(scores, t, s1_ids, one_to_one, min_best), truth))
               for t in grid]
    best_t, best_f = max(results, key=lambda r: (r[1], -abs(r[0] - 0.5)))
    print(f"[Thresh] entity-level macro F0.5 = {best_f:.4f} @ threshold {best_t:.2f} "
          f"(one_to_one={one_to_one}, grid {grid[0]:.2f}-{grid[-1]:.2f})")
    return float(best_t), best_f
