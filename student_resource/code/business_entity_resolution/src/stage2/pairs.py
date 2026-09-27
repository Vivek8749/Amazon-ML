"""Candidate pairs + pair features for a split.

pairs frame columns: s1_id, cand_id, word_cos, word_rank, char_cos, char_rank,
then the 48 stage-1 similarity features (entity_resolution.features.FEATURE_NAMES).
"""
import gc
import os

import numpy as np
import pandas as pd

from stage2.common import P, cached_source, load, log

MISSING_RANK = 40


def pairs_from_cands(by_strategy):
    """by_strategy: {strategy: {s1_id: (ids, scores)}} -> pairs DataFrame."""
    frames = []
    for s, cd in by_strategy.items():
        sids, cids, cos, rank = [], [], [], []
        for sid, (ids, sc) in cd.items():
            n = len(ids)
            if n == 0:
                continue
            sids.append(np.full(n, sid, dtype=object)); cids.append(np.asarray(ids, dtype=object))
            cos.append(np.asarray(sc, dtype=np.float32)); rank.append(np.arange(n, dtype=np.int16))
        f = pd.DataFrame({"s1_id": np.concatenate(sids), "cand_id": np.concatenate(cids),
                          f"{s}_cos": np.concatenate(cos), f"{s}_rank": np.concatenate(rank)})
        frames.append(f.set_index(["s1_id", "cand_id"]))
    df = frames[0]
    for f in frames[1:]:
        df = df.join(f, how="outer")
    df = df.reset_index()
    for s in by_strategy:
        df[f"{s}_cos"] = df[f"{s}_cos"].fillna(0).astype(np.float32)
        df[f"{s}_rank"] = df[f"{s}_rank"].fillna(MISSING_RANK).astype(np.int16)
    return df


def add_pair_features(df, s1, pool):
    from entity_resolution.features import FEATURE_NAMES, parallel_compute_features_ordered
    s1n = dict(zip(s1["entity_id"], s1["name_clean"])); s1a = dict(zip(s1["entity_id"], s1["addr_clean"]))
    pn = dict(zip(pool["entity_id"], pool["name_clean"])); pa = dict(zip(pool["entity_id"], pool["addr_clean"]))
    data = [(s1n[a], s1a[a], pn[b], pa[b]) for a, b in zip(df["s1_id"].values, df["cand_id"].values)]
    from stage2.extra import EXTRA_NAMES, compute_extra
    X = parallel_compute_features_ordered(data, desc="pair feats")
    E = compute_extra(data)
    del data; gc.collect()
    feats = pd.DataFrame(X, columns=list(FEATURE_NAMES))
    extra = pd.DataFrame(E, columns=EXTRA_NAMES)
    return pd.concat([df.reset_index(drop=True), feats, extra], axis=1)


def label(df, gt):
    truth = {}
    for sid, m in gt[["source1_entity_id", "matched_entity_ids"]].values:
        if isinstance(m, str) and m:
            for c in m.split(","):
                truth[c] = sid
    df["y"] = (df["cand_id"].map(truth) == df["s1_id"]).astype(np.int8)
    return df
