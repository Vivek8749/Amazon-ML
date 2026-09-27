"""Test prediction with the stage-1 + stage-2 models (checkpointed per country shard).

    python -m stage2.predict score [country ...]       # scores -> er_work/test_scores_<co>_<i>.parquet
    python -m stage2.predict write THRESHOLD           # decision + output TSVs
"""
import gc
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

from stage2.common import P, cached_source, exists, load, log
from stage2.model import STAGE2_FLOOR, add_block_feats, add_stage2_feats
from stage2.pairs import add_pair_features, pairs_from_cands

SHARD = 150_000


def avg_pred(boosters, X):
    d = xgb.DMatrix(X)
    return np.mean([b.predict(d, iteration_range=(0, b.best_iteration + 1)) for b in boosters],
                   axis=0).astype(np.float32)


STRATS = ("word", "char", "skel")


def feats(countries):
    """Pair + blocking features for every test shard -> test_feat_<co>_<i>.parquet"""
    s1 = cached_source("test_source1")
    pool = pd.concat([cached_source("test_source2"), cached_source("test_source3")],
                     ignore_index=True)[["entity_id", "name_clean", "addr_clean", "country_norm"]]
    for co in countries:
        s1c = s1[s1["country_norm"] == co].reset_index(drop=True)
        n = (len(s1c) + SHARD - 1) // SHARD
        if all(exists(f"test_feat_{co}_{i}.parquet") for i in range(n)):
            continue
        by = {s: load(f"cands_test_{co}_{s}.pkl") for s in STRATS}
        pc = pool[pool["country_norm"] == co]
        for i in range(n):
            out = f"test_feat_{co}_{i}.parquet"
            if exists(out):
                continue
            sh = s1c.iloc[i * SHARD:(i + 1) * SHARD]
            ids = set(sh["entity_id"])
            df = pairs_from_cands({s: {k: v for k, v in c.items() if k in ids} for s, c in by.items()})
            log(f"[{co} {i+1}/{n}] {len(df):,} pairs")
            df = add_pair_features(df, sh, pc[pc["entity_id"].isin(set(df["cand_id"]))])
            df = add_block_feats(df)
            df.to_parquet(P(out))
            log(f"[{co} {i+1}/{n}] features saved")
            del df; gc.collect()
        del by; gc.collect()


def score(countries):
    M = load("stage_models.pkl")
    s1 = cached_source("test_source1")
    pool = None
    for co in countries:
        s1c = s1[s1["country_norm"] == co].reset_index(drop=True)
        n = (len(s1c) + SHARD - 1) // SHARD
        if all(exists(f"test_scores_{co}_{i}.parquet") for i in range(n)):
            continue
        if pool is None:
            pool = pd.concat([cached_source("test_source2"), cached_source("test_source3")],
                             ignore_index=True)[["entity_id", "name_clean", "addr_clean", "country_norm"]]
        pc = pool[pool["country_norm"] == co]
        name_of = dict(zip(pc["entity_id"], pc["name_clean"])); addr_of = dict(zip(pc["entity_id"], pc["addr_clean"]))
        for i in range(n):
            out = f"test_scores_{co}_{i}.parquet"
            if exists(out):
                continue
            if not exists(f"test_feat_{co}_{i}.parquet"):
                feats([co])
            df = pd.read_parquet(P(f"test_feat_{co}_{i}.parquet"))
            log(f"[{co} {i+1}/{n}] scoring {len(df):,} pairs")
            df["p1"] = avg_pred(M["B1"], df[M["F1"]].values.astype(np.float32))
            s2 = df[df["p1"] >= STAGE2_FLOOR].copy()
            s2 = add_stage2_feats(s2, name_of, addr_of)
            s2["p2"] = avg_pred(M["B2"], s2[M["F2"]].values.astype(np.float32))
            res = df[["s1_id", "cand_id", "p1"]].merge(s2[["s1_id", "cand_id", "p2"]], how="left",
                                                      on=["s1_id", "cand_id"])
            res["p2"] = res["p2"].fillna(res["p1"]).astype(np.float32)
            res.to_parquet(P(out))
            log(f"[{co} {i+1}/{n}] saved; pairs p2>=0.5: {(res.p2 >= 0.5).sum():,}")
            del df, s2, res; gc.collect()


def write(threshold, col="p2"):
    from entity_resolution.data import write_output
    s1 = cached_source("test_source1")
    matches, cands = {}, {}
    for co in sorted(s1["country_norm"].unique()):
        s1c = s1[s1["country_norm"] == co]
        n = (len(s1c) + SHARD - 1) // SHARD
        sc = pd.concat([pd.read_parquet(P(f"test_scores_{co}_{i}.parquet")) for i in range(n)],
                       ignore_index=True)
        cands.update(sc.groupby("s1_id")["cand_id"].agg(list).to_dict())
        k = sc[sc[col] >= threshold].sort_values([col, "s1_id"], ascending=[False, True], kind="stable")
        k = k.drop_duplicates("cand_id", keep="first")
        m = k.groupby("s1_id")["cand_id"].agg(sorted).to_dict()
        matches.update(m)
        log(f"[{co}] {len(m):,}/{len(s1c):,} entities matched, {len(k):,} matches, "
            f"{len(k)/max(len(m),1):.2f} per matched entity")
    ids = s1["entity_id"].values
    out_dir = os.environ.get("ER_OUT", P("output"))
    write_output({s: matches.get(s, []) for s in ids}, {s: sorted(cands.get(s, [])) for s in ids}, out_dir)
    log(f"wrote {out_dir}")


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(sys.argv[2:] or ["france", "us", "india"])
    elif sys.argv[1] == "feats":
        feats(sys.argv[2:] or ["france", "us", "india"])
    else:
        write(float(sys.argv[2]), *(sys.argv[3:4]))
