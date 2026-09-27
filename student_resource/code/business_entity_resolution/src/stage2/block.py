"""Per-country char + word TF-IDF top-K blocking that also returns cosine scores.

Usage:
    python -m stage2.block train N      # N unseen training S1 entities vs the full training pool
    python -m stage2.block test us,india  # test S1 vs the test pool (char TF-IDF only)

Output per (split, country, strategy): {s1_id: (cand_ids ndarray, scores ndarray)} best first.
"""
import gc
import re
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from stage2.common import cached_source, exists, log, save
from entity_resolution.blocking import tfidf as T

TOP_K = 30
TUNE_OFFSET = 1_000_000


def query_scored(index, texts, countries, top_k):
    xp, xsp, is_gpu = T._backend()
    texts, countries = np.asarray(texts), np.asarray(countries)
    out = [None] * len(texts)
    for co, (pool_mat, pool_ids, n_real) in index.parts.items():
        q_idx = np.flatnonzero(countries == co)
        if len(q_idx) == 0:
            continue
        k = min(top_k, n_real)
        q_mat = T._parallel_transform(index.vec, texts[q_idx]).astype(np.float32)
        chunk = index._chunk_size(pool_mat.shape[0])
        for start in tqdm(range(0, len(q_idx), chunk), desc=f"{index.name} [{co}]", mininterval=30):
            qc = q_mat[start:start + chunk].tocoo()
            c = qc.shape[0]
            dense_q = xp.zeros((index.n_features, c), dtype=xp.float32)
            dense_q[xp.asarray(qc.col), xp.asarray(qc.row)] = xp.asarray(qc.data)
            scores = xp.around(pool_mat @ dense_q, 5)
            rows, vals = T._block_topk(xp, scores, k)
            rows, vals = T._to_host(xp, rows), T._to_host(xp, vals)
            del scores, dense_q
            order = np.lexsort((rows, -vals), axis=0)
            rows = np.take_along_axis(rows, order, axis=0)
            vals = np.take_along_axis(vals, order, axis=0)
            for j in range(c):
                m = (vals[:, j] > 0) & (rows[:, j] < n_real)
                out[q_idx[start + j]] = (pool_ids[rows[m, j]], vals[m, j].astype(np.float32))
    if is_gpu:
        xp.get_default_memory_pool().free_all_blocks()
    empty = (np.array([], dtype=object), np.array([], dtype=np.float32))
    return [o if o is not None else empty for o in out]


_LEGAL_RE = re.compile(r"\b(pvt|private|ltd|limited|llp|inc|llc|corp|corporation|co|company|"
                       r"praivet|piraivet|praibhet|limitet|pra|li)\b\.?")
_DIGRAPH = [("ph", "f"), ("sh", "s"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("bh", "b"),
            ("dh", "d"), ("ch", "c"), ("c", "k"), ("q", "k"), ("x", "ks"), ("z", "j"), ("w", "v"), ("d", "t"), ("g", "k"), ("b", "p"), ("f", "p")]


def skeleton(name, addr):
    """Phonetic consonant skeleton of the name + the address numbers: robust to
    transliterated native-script names ("royl proprtijh" ~ "royal properties")
    and to empty / truncated addresses."""
    n = _LEGAL_RE.sub(" ", (name or "").lower())
    n = re.sub(r"[^a-z ]", "", n)
    for a, b in _DIGRAPH:
        n = n.replace(a, b)
    n = " ".join(re.sub(r"(.)\1+", r"\1", re.sub(r"[aeiouy]", "", w)) or w[:1] for w in n.split())
    nums = " ".join(re.findall(r"\d+", addr or "")[:2])
    return f"{n} {nums}".strip()


def skeleton_text(df):
    return np.array([skeleton(a, b) for a, b in zip(df["name_clean"].values, df["addr_clean"].values)],
                    dtype=object)


def build_skel_blocker(pool_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return T.TfidfIndex("skel", TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), max_df=0.05, max_features=150_000,
        sublinear_tf=True, dtype=np.float32), pool_df, "skel")


def run(split, s1, pool, countries, strategies=("word", "char")):
    for co in countries:
        s1_co = s1[s1["country_norm"] == co]
        need = [s for s in strategies if not exists(f"cands_{split}_{co}_{s}.pkl")]
        if not need or len(s1_co) == 0:
            continue
        pool_co = pool[pool["country_norm"] == co].reset_index(drop=True)
        log(f"[{split}/{co}] S1 {len(s1_co):,}  pool {len(pool_co):,}  strategies {need}")
        for s in need:
            t0 = time.time()
            if s == "skel":
                pool_co["skel"] = skeleton_text(pool_co)
                idx = build_skel_blocker(pool_co)
                q = skeleton_text(s1_co)
            else:
                idx = (T.build_word_tfidf_blocker if s == "word" else T.build_tfidf_blocker)(pool_co)
                q = s1_co["combined"].values
            res = query_scored(idx, q, s1_co["country_norm"].values, TOP_K)
            idx.release(); del idx; gc.collect()
            save(dict(zip(s1_co["entity_id"].values, res)), f"cands_{split}_{co}_{s}.pkl")
            log(f"[{split}/{co}/{s}] done in {time.time()-t0:.0f}s")
        del pool_co; gc.collect()


def main():
    split = sys.argv[1]
    if split == "train":
        n = int(sys.argv[2])
        s1 = cached_source("train_source1")
        perm = np.random.RandomState(42).permutation(len(s1))
        s1 = s1.iloc[np.sort(perm[TUNE_OFFSET:TUNE_OFFSET + n])].reset_index(drop=True)
        save(s1["entity_id"].values, "train_ids.pkl")
        pool = pd.concat([cached_source("train_source2"), cached_source("train_source3")],
                         ignore_index=True)
        countries = sys.argv[3].split(",") if len(sys.argv) > 3 else sorted(s1["country_norm"].unique())
        strategies = sys.argv[4].split(",") if len(sys.argv) > 4 else ("word", "char")
        run("train", s1, pool, countries, strategies)
    else:
        countries = sys.argv[2].split(",")
        strategies = sys.argv[3].split(",") if len(sys.argv) > 3 else ("char",)
        s1 = cached_source("test_source1")
        pool = pd.concat([cached_source("test_source2"), cached_source("test_source3")],
                         ignore_index=True)
        run("test", s1, pool, countries, strategies)


if __name__ == "__main__":
    main()
