"""Query expansion: re-query the pool with the text of each entity's most confident
match, so records that resemble the matched copy (same native script, same short
address) but not the S1 record are retrieved.

    python -m stage2.expand train|test COUNTRY SCORES_PARQUET [col] [min_p]
Output: cands_<split>_<co>_exp.pkl  {s1_id: (ids, scores)}
"""
import gc
import sys
import time

import numpy as np
import pandas as pd

from stage2.block import TOP_K, query_scored
from stage2.common import P, cached_source, exists, log, save
from entity_resolution.blocking import tfidf as T


def main():
    split, co, scores_path = sys.argv[1:4]
    col = sys.argv[4] if len(sys.argv) > 4 else "p1"
    min_p = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    out = f"cands_{split}_{co}_exp.pkl"
    if exists(out):
        return
    sc = pd.read_parquet(scores_path, columns=["s1_id", "cand_id", col])
    sc = sc[sc[col] >= min_p].sort_values(col, ascending=False).drop_duplicates("s1_id")
    pre = "train" if split == "train" else "test"
    pool = pd.concat([cached_source(f"{pre}_source2"), cached_source(f"{pre}_source3")], ignore_index=True)
    pool = pool[pool["country_norm"] == co].reset_index(drop=True)
    sc = sc[sc["cand_id"].isin(set(pool["entity_id"]))]
    log(f"[{split}/{co}] expansion queries {len(sc):,}, pool {len(pool):,}")
    t0 = time.time()
    idx = T.build_tfidf_blocker(pool)
    text = dict(zip(pool["entity_id"], pool["combined"]))
    q = np.array([text[c] for c in sc["cand_id"].values], dtype=object)
    res = query_scored(idx, q, np.full(len(q), co, dtype=object), TOP_K)
    idx.release(); gc.collect()
    save(dict(zip(sc["s1_id"].values, res)), out)
    log(f"[{split}/{co}] expansion done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
