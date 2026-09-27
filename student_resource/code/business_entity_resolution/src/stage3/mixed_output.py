"""Interim submission: stage-3 decisions for the countries already scored, the old model's
saved test scores (the 0.978 leaderboard run, threshold 0.92) for the rest.

    python -m stage3.mixed_output france,us    # -> er_work/output_mixed
"""
import glob
import os
import pickle
import sys

import pandas as pd

from stage2.common import P, PRED_CKPT, cached_source, exists, load, log
from stage3.predict import SHARD

OLD_THR = 0.92


def one_to_one(sc, t):
    k = sc[sc["p"] >= t].sort_values(["p", "s1_id"], ascending=[False, True], kind="stable")
    return k.drop_duplicates("cand_id", keep="first")


def main():
    new = sys.argv[1].split(",")
    from entity_resolution.data import write_output
    s1 = cached_source("test_source1")
    G = load("graph_params.pkl")
    f, (t, lo, s_add, hi, s_drop) = G
    emb = None
    matches, cands = {}, {}
    for co in sorted(s1["country_norm"].unique()):
        if co in new:
            n = (len(s1[s1["country_norm"] == co]) + SHARD - 1) // SHARD
            sc = pd.concat([pd.read_parquet(P(f"test_scores_{co}_{i}_e.parquet")) for i in range(n)],
                           ignore_index=True)[["s1_id", "cand_id", "p2"]].rename(columns={"p2": "p"})
            if emb is None and not (s_add > 1 and hi <= 0):
                from stage3.emb import EmbLookup
                emb = EmbLookup("test")
            if s_add > 1 and hi <= 0:     # graph tuning chose the plain threshold
                k = one_to_one(sc, t)
                src = f"stage-3 (thr {t})"
            else:
                from stage3.graph import apply
                k = apply(sc, emb, t, lo, s_add, hi, s_drop)
                src = f"stage-3 (thr {t}, graph)"
        else:
            sc = pd.concat([pickle.load(open(x, "rb")) for x in
                            sorted(glob.glob(os.path.join(PRED_CKPT, f"{co}_scores_*.pkl")))], ignore_index=True)
            k = one_to_one(sc, OLD_THR)
            src = f"old model (thr {OLD_THR})"
        if co in new:
            cands.update(sc.groupby("s1_id")["cand_id"].agg(list).to_dict())
        else:
            cands.update(pickle.load(open(os.path.join(PRED_CKPT, f"{co}_candidates.pkl"), "rb")))
        m = k.groupby("s1_id")["cand_id"].agg(sorted).to_dict()
        matches.update(m)
        log(f"[{co}] {src}: {len(m):,} entities matched, {len(k):,} matches")
    ids = s1["entity_id"].values
    out = P("output_mixed")
    write_output({s: matches.get(s, []) for s in ids}, {s: sorted(cands.get(s, [])) for s in ids}, out)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
