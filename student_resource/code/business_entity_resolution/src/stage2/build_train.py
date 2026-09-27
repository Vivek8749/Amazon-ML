"""Pair features + labels + old-model score for the stage-2 training entities.

    python -m stage2.build_train [country ...]
"""
import gc
import pickle
import sys

import numpy as np
import pandas as pd

from stage2.common import P, cached_source, exists, load, log
from stage2.pairs import add_pair_features, label, pairs_from_cands

OLD_MODEL = "/teamspace/studios/this_studio/Amazon-ML/student_resource/code/business_entity_resolution/src/models/xgb_model.pkl"


def main():
    from entity_resolution.features import FEATURE_NAMES
    ids = set(load("train_ids.pkl"))
    s1 = cached_source("train_source1")
    s1 = s1[s1["entity_id"].isin(ids)].reset_index(drop=True)
    gt = pd.read_csv("/teamspace/studios/this_studio/Amazon-ML/student_resource/dataset/train/train_ground_truth.tsv",
                     sep="\t", dtype=str)
    gt = gt[gt["source1_entity_id"].isin(ids)]
    countries = sys.argv[1:] or sorted(s1["country_norm"].unique())
    old = pickle.load(open(OLD_MODEL, "rb"))["model"]
    pool = None
    for co in countries:
        out = f"feat_train_{co}.parquet"
        if exists(out):
            continue
        if pool is None:
            pool = pd.concat([cached_source("train_source2"), cached_source("train_source3")],
                             ignore_index=True)[["entity_id", "name_clean", "addr_clean", "country_norm"]]
        by = {s: load(f"cands_train_{co}_{s}.pkl") for s in ("word", "char", "skel")}
        df = pairs_from_cands(by); del by; gc.collect()
        log(f"[{co}] {len(df):,} pairs ({len(df)/max(df.s1_id.nunique(),1):.1f}/entity)")
        s1c = s1[s1["country_norm"] == co]
        pc = pool[pool["entity_id"].isin(set(df["cand_id"].unique()))]
        df = add_pair_features(df, s1c, pc)
        df = label(df, gt)
        df["p_old"] = old.predict_proba(df[list(FEATURE_NAMES)].values)[:, 1].astype(np.float32)
        df["country"] = co
        df.to_parquet(P(out))
        log(f"[{co}] saved {out}: pos rate {df.y.mean():.4f}")
        del df; gc.collect()


if __name__ == "__main__":
    main()
