"""Add the fine-tuned-embedding strategy to already-built train features.

    python -m stage3.augment [country ...]   # feat_train_<co>.parquet -> feat_train_<co>_e.parquet

Rebuilds the candidate union with the emb strategy, reuses the stored string
features for known pairs, computes them only for the new emb-only pairs, and adds
emb_cos / emb_rank / emb_sim for every pair.
"""
import gc
import pickle
import sys

import numpy as np
import pandas as pd

from stage2.build_train import OLD_MODEL
from stage2.common import P, cached_source, exists, load, log
from stage3.emb import EmbLookup
from stage2.pairs import add_pair_features, label, pairs_from_cands

STRATS = ("word", "char", "skel", "emb")


def with_emb(df_old, by, s1c, pool_c, emb, gt=None, old_model=None):
    """df_old: pair frame built from word/char/skel (features included) or None."""
    union = pairs_from_cands(by)
    if df_old is None:
        new = union
        keep = None
    else:
        key = pd.MultiIndex.from_frame(df_old[["s1_id", "cand_id"]])
        is_new = ~pd.MultiIndex.from_frame(union[["s1_id", "cand_id"]]).isin(key)
        new = union[is_new].reset_index(drop=True)
        keep = df_old.merge(union[["s1_id", "cand_id", "emb_cos", "emb_rank"]], on=["s1_id", "cand_id"], how="left")
        keep["emb_cos"] = keep["emb_cos"].fillna(0).astype(np.float32)
        keep["emb_rank"] = keep["emb_rank"].fillna(40).astype(np.int16)
    log(f"  {len(new):,} new pairs need string features")
    if len(new):
        new = add_pair_features(new, s1c, pool_c[pool_c["entity_id"].isin(set(new["cand_id"]))])
        if gt is not None:
            new = label(new, gt)
        if old_model is not None:
            from entity_resolution.features import FEATURE_NAMES
            new["p_old"] = old_model.predict_proba(new[list(FEATURE_NAMES)].values)[:, 1].astype(np.float32)
    df = new if keep is None else pd.concat([keep, new], ignore_index=True)
    df["emb_sim"] = emb.sim(df["s1_id"].values, df["cand_id"].values)
    return df


def main():
    ids = set(load("train_ids.pkl"))
    s1 = cached_source("train_source1")
    s1 = s1[s1["entity_id"].isin(ids)].reset_index(drop=True)
    gt = pd.read_csv("/teamspace/studios/this_studio/Amazon-ML/student_resource/dataset/train/train_ground_truth.tsv",
                     sep="\t", dtype=str)
    gt = gt[gt["source1_entity_id"].isin(ids)]
    old = pickle.load(open(OLD_MODEL, "rb"))["model"]
    emb = EmbLookup("train")
    pool = pd.concat([cached_source("train_source2"), cached_source("train_source3")],
                     ignore_index=True)[["entity_id", "name_clean", "addr_clean", "country_norm"]]
    for co in sys.argv[1:] or ["india", "us"]:
        out = f"feat_train_{co}_e.parquet"
        if exists(out):
            continue
        df_old = pd.read_parquet(P(f"feat_train_{co}.parquet"))
        by = {s: load(f"cands_train_{co}_{s}.pkl") for s in STRATS}
        df = with_emb(df_old, by, s1[s1["country_norm"] == co], pool[pool["country_norm"] == co], emb, gt, old)
        df["country"] = co
        del df_old, by; gc.collect()
        df.to_parquet(P(out))
        log(f"[{co}] saved {out}: {len(df):,} pairs, pos {int(df.y.sum()):,}")
        del df; gc.collect()


if __name__ == "__main__":
    main()
