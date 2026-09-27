"""Safety submission: old model's saved test scores re-decided at a stricter threshold.

    python -m stage2.safe_output 0.99
"""
import glob
import os
import pickle
import sys

import pandas as pd

from stage2.common import P, PRED_CKPT, cached_source, log


def main():
    t = float(sys.argv[1])
    from entity_resolution.data import write_output
    s1 = cached_source("test_source1")
    matches, cands = {}, {}
    for co in ("france", "india", "us"):
        cands.update(pickle.load(open(os.path.join(PRED_CKPT, f"{co}_candidates.pkl"), "rb")))
        sc = pd.concat([pickle.load(open(f, "rb")) for f in
                        sorted(glob.glob(os.path.join(PRED_CKPT, f"{co}_scores_*.pkl")))], ignore_index=True)
        k = sc[sc["p"] >= t].sort_values(["p", "s1_id"], ascending=[False, True], kind="stable")
        k = k.drop_duplicates("cand_id", keep="first")
        m = k.groupby("s1_id")["cand_id"].agg(sorted).to_dict()
        matches.update(m)
        log(f"{co}: {len(m):,} entities matched, {len(k):,} matches")
    ids = s1["entity_id"].values
    out = P(f"output_safe_{t}")
    write_output({s: matches.get(s, []) for s in ids}, {s: sorted(cands.get(s, [])) for s in ids}, out)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
