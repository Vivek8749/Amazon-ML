"""Graph post-processing over each S1 entity's candidate cluster.

Accepted matches of one S1 entity should form one cluster of S2/S3 copies of the
same business. With the fine-tuned embeddings as S2/S3 <-> S2/S3 edge weights:

  recover: a rejected candidate with p >= lo joins if its best edge to an accepted
           match is >= s_add (A->B, B~C  =>  A->C)
  prune:   an accepted match with p < hi is dropped if its best edge to the
           other accepted matches is < s_drop (and the entity has >= 2 accepted)

    python -m stage3.graph tune [oof_parquet]   # grid on OOF, saves graph_params.pkl
"""
import itertools
import sys

import numpy as np
import pandas as pd
import torch

from stage2.common import P, cached_source, load, log, save


def one_to_one(sc, t):
    k = sc[sc["p"] >= t].sort_values(["p", "s1_id"], ascending=[False, True], kind="stable")
    return k.drop_duplicates("cand_id", keep="first")


def edges(emb, rows):
    """rows: s1_id, cand_id, accepted(bool). For every row: best cosine to an accepted
    *other* candidate of the same entity (-1 if none)."""
    rows = rows.sort_values("s1_id", kind="stable").reset_index(drop=True)
    r = emb.row["pool"].reindex(rows["cand_id"].values).values
    ok = ~np.isnan(r)
    E = torch.from_numpy(np.asarray(emb.E["pool"][np.where(ok, r, 0).astype(np.int64)])).cuda().float()
    E[~torch.from_numpy(ok).cuda()] = 0
    sid = rows["s1_id"].values
    start = np.r_[0, np.flatnonzero(sid[1:] != sid[:-1]) + 1]
    size = np.diff(np.r_[start, len(rows)])
    acc = torch.from_numpy(rows["accepted"].values).cuda()
    best = np.full(len(rows), -1.0, np.float32)
    # pad groups to a common width in size buckets
    for lo_, hi_ in ((1, 8), (9, 32), (33, 128), (129, 10_000)):
        g = np.flatnonzero((size >= lo_) & (size <= hi_))
        if len(g) == 0:
            continue
        W = int(size[g].max())
        for b in range(0, len(g), max(1, 2_000_000 // (W * W))):
            gg = g[b:b + max(1, 2_000_000 // (W * W))]
            idx = start[gg][:, None] + np.arange(W)[None, :]
            valid = np.arange(W)[None, :] < size[gg][:, None]
            idx_t = torch.from_numpy(np.where(valid, idx, 0)).cuda()
            X = E[idx_t]                                         # (G, W, d)
            S = X @ X.transpose(1, 2)                             # (G, W, W)
            a = acc[idx_t] & torch.from_numpy(valid).cuda()      # accepted columns
            S = S.masked_fill(~a[:, None, :], -1.0)
            S = S.masked_fill(torch.eye(W, dtype=torch.bool, device="cuda")[None], -1.0)
            m = S.max(-1).values.cpu().numpy()
            best[idx[valid]] = m[valid]
    rows["edge"] = best
    return rows


def apply(sc, emb, t, lo, s_add, hi, s_drop):
    """sc: s1_id, cand_id, p. Returns kept rows (s1_id, cand_id, p)."""
    base = one_to_one(sc, t)
    cand = sc[sc["p"] >= min(lo, t)][["s1_id", "cand_id", "p"]]
    key = set(zip(base["s1_id"], base["cand_id"]))
    cand = cand.assign(accepted=[(a, b) in key for a, b in zip(cand["s1_id"], cand["cand_id"])])
    cand = edges(emb, cand)
    n_acc = cand.groupby("s1_id")["accepted"].transform("sum")
    add = (~cand["accepted"]) & (cand["p"] >= lo) & (cand["edge"] >= s_add)
    drop = cand["accepted"] & (cand["p"] < hi) & (n_acc >= 2) & (cand["edge"] < s_drop)
    kept = cand[(cand["accepted"] & ~drop) | add]
    # recovered rows must still respect one-to-one against other entities' accepted rows
    kept = kept.sort_values(["accepted", "p"], ascending=[False, False], kind="stable") \
               .drop_duplicates("cand_id", keep="first")
    return kept[["s1_id", "cand_id", "p"]]


def tune(oof_path):
    from stage3.emb import EmbLookup
    from stage3.train import kept_macro, truth_for
    df = pd.read_parquet(oof_path)
    col = "p2"
    ids = load("train_ids.pkl")
    s1 = cached_source("train_source1")[["entity_id", "country_norm"]]
    keep = set(s1[s1["country_norm"].isin(df["country"].unique())]["entity_id"]) & set(ids)
    truth = truth_for(keep)
    emb = EmbLookup("train")
    sc = df[["s1_id", "cand_id", col, "y"]].rename(columns={col: "p"})
    best = None
    for t in (0.6, 0.65, 0.7, 0.75, 0.8):
        f0 = kept_macro(one_to_one(sc, t)[["s1_id", "cand_id"]], df, truth)
        log(f"t={t}: base {f0:.4f}")
        if best is None or f0 > best[0]:
            best = (f0, (t, 1.0, 2.0, 0.0, -2.0))
    t = best[1][0]
    for lo, s_add, hi, s_drop in itertools.product((0.2, 0.35, 0.5), (0.85, 0.9, 0.95, 2.0),
                                                   (0.0, 0.85, 0.95), (0.3, 0.5)):
        if s_add == 2.0 and lo != 0.5:
            continue
        if hi == 0.0 and s_drop != 0.3:
            continue
        f = kept_macro(apply(sc, emb, t, lo, s_add, hi, s_drop)[["s1_id", "cand_id"]], df, truth)
        tag = " *" if f > best[0] else ""
        log(f"t={t} lo={lo} s_add={s_add} hi={hi} s_drop={s_drop}: {f:.4f}{tag}")
        if f > best[0]:
            best = (f, (t, lo, s_add, hi, s_drop))
    log(f"best {best}")
    save(best, "graph_params.pkl")


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        tune(sys.argv[2] if len(sys.argv) > 2 else P("oof_train_e.parquet"))
