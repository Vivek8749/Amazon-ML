"""TF-IDF blocking: exact per-country top-K cosine retrieval, on GPU when available.

Why not sparse x sparse? Common character n-grams (" s", "a ") occur in almost
every record, so a query-vs-pool similarity row is ~100% dense: the old scipy
path produced 4.7M non-zeros per query row on the India test pool (~0.6 s per
query, ~128 h for India). Here each chunk of queries is densified and scored
with one sparse(pool) x dense(queries) product, followed by an exact two-stage
block top-K. The same code runs on CuPy (GPU) or NumPy/SciPy (CPU fallback).
"""
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from ..config import (
    N_WORKERS, NAME_TFIDF_MAX_FEATURES, TFIDF_CPU_CHUNK, TFIDF_FIT_SAMPLE, TFIDF_DEVICE, TFIDF_GPU_MEM_FRACTION,
    TFIDF_MAX_CHUNK, TFIDF_MAX_DF, TFIDF_MAX_FEATURES, WORD_TFIDF_MAX,
)

_BLOCK = 256          # rows per block in the two-stage top-K
_BACKEND = None       # resolved once: (xp, sparse module, is_gpu)


# ===== BACKEND =================================================================

def _backend():
    """(array module, sparse module, is_gpu) according to TFIDF_DEVICE."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = (np, sp, False)
        if TFIDF_DEVICE in ("auto", "cuda"):
            try:
                import cupy as cp
                import cupyx.scipy.sparse as csp
                # Tiny SpMM: fails fast if cuSPARSE / nvJitLink can't be loaded.
                probe = csp.csr_matrix(sp.identity(4, dtype=np.float32, format="csr"))
                (probe @ cp.ones((4, 2), dtype=cp.float32)).sum()
                _BACKEND = (cp, csp, True)
            except Exception as exc:
                if TFIDF_DEVICE == "cuda":
                    raise RuntimeError("TFIDF_DEVICE='cuda' but CuPy sparse is unusable") from exc
                print(f"[TF-IDF] GPU unavailable ({type(exc).__name__}) — using CPU")
        print(f"[TF-IDF] Retrieval backend: {'GPU (CuPy)' if _BACKEND[2] else 'CPU (SciPy)'}")
    return _BACKEND


def _to_host(xp, a):
    return a.get() if hasattr(a, "get") else a


# ===== EXACT TOP-K =============================================================

def _block_topk(xp, s, k):
    """Top-k rows per column of a dense (n_pool, n_query) score matrix.

    Stage 1 keeps the k blocks with the highest block maxima; every top-k
    element lies in one of them, so stage 2 (top-k inside those blocks) is
    exact. Much faster than a full-axis argpartition, which CuPy implements
    with a full sort. Returns (rows, scores), each (k, n_query), unordered.
    """
    n, c = s.shape
    if k >= n:
        return xp.broadcast_to(xp.arange(n)[:, None], (n, c)), s
    nb = n // _BLOCK
    if nb < k or n % _BLOCK:
        rows = xp.argpartition(-s, k - 1, axis=0)[:k]
        return rows, xp.take_along_axis(s, rows, axis=0)
    s3 = s.reshape(nb, _BLOCK, c)
    blk = xp.argpartition(-s3.max(axis=1), k - 1, axis=0)[:k]                  # (k, c)
    vals = xp.take_along_axis(
        s3, xp.broadcast_to(blk[:, None, :], (k, _BLOCK, c)), axis=0,
    ).reshape(k * _BLOCK, c)
    loc = xp.argpartition(-vals, k - 1, axis=0)[:k]                             # (k, c)
    rows = blk[loc // _BLOCK, xp.arange(c)[None, :]] * _BLOCK + loc % _BLOCK
    return rows, xp.take_along_axis(vals, loc, axis=0)


# ===== PARALLEL VECTORISATION ==================================================

def _parallel_transform(vec, texts):
    """vec.transform on all CPU cores (sklearn's transform is single-threaded)."""
    texts = np.asarray(texts, dtype=object)
    if len(texts) < 200_000:
        return vec.transform(texts).tocsr()
    parts = np.array_split(np.arange(len(texts)), N_WORKERS * 2)
    mats = Parallel(n_jobs=N_WORKERS, backend="loky")(delayed(vec.transform)(texts[p]) for p in parts)
    return sp.vstack(mats, format="csr")


def _fit_transform(vec, texts):
    """Fit vocabulary + IDF on a sample of at most TFIDF_FIT_SAMPLE records, then
    transform every record in parallel."""
    texts = np.asarray(texts, dtype=object)
    if len(texts) > TFIDF_FIT_SAMPLE:
        idx = np.random.RandomState(0).choice(len(texts), TFIDF_FIT_SAMPLE, replace=False)
        vec.fit(texts[np.sort(idx)])
    else:
        vec.fit(texts)
    return _parallel_transform(vec, texts)


# ===== INDEX ===================================================================

class TfidfIndex:
    """A fitted TF-IDF vectoriser plus one pool matrix per country."""

    def __init__(self, name, vectorizer, pool_df, text_col):
        self.name, self.vec, self.text_col = name, vectorizer, text_col
        xp, xsp, is_gpu = _backend()
        print(f"[Block] Fitting {name} TF-IDF on '{text_col}'...")
        t0 = time.time()
        mat = _fit_transform(vectorizer, pool_df[text_col].values)
        mat.indices = mat.indices.astype(np.int32)
        mat.indptr = mat.indptr.astype(np.int32)
        self.n_features = mat.shape[1]

        ids = pool_df["entity_id"].values
        countries = pool_df["country_norm"].values
        self.parts = {}   # country -> (device CSR padded to a multiple of _BLOCK, ids, n_real)
        for co in pd.unique(countries):
            rows = np.flatnonzero(countries == co)
            sub = mat[rows]
            pad = (-sub.shape[0]) % _BLOCK
            if pad:
                sub = sp.vstack([sub, sp.csr_matrix((pad, sub.shape[1]), dtype=np.float32)],
                                format="csr")
            dev = xsp.csr_matrix(sub) if is_gpu else sub
            self.parts[co] = (dev, ids[rows], len(rows))
        print(f"[Block] {name} TF-IDF {mat.shape}, nnz={mat.nnz:,} "
              f"({mat.nnz / max(mat.shape[0], 1):.0f}/record) in {time.time()-t0:.1f}s")

    def _chunk_size(self, n_pool):
        xp, _, is_gpu = _backend()
        if not is_gpu:
            return TFIDF_CPU_CHUNK
        free = xp.cuda.Device().mem_info[0]
        # scores (n_pool) + dense query column (n_features) + top-K temporaries
        per_query = 4 * (n_pool + self.n_features) * 1.3
        return int(max(1, min(TFIDF_MAX_CHUNK, free * TFIDF_GPU_MEM_FRACTION // per_query)))

    def query(self, texts, countries, top_k):
        """Top-k pool entity_ids per query (same country, score > 0, best first)."""
        xp, xsp, is_gpu = _backend()
        # ER_TFIDF_SPGEMM=1 (env): sparse x sparse product on the GPU
        spgemm = is_gpu and os.environ.get("ER_TFIDF_SPGEMM") == "1"
        texts, countries = np.asarray(texts), np.asarray(countries)
        results = [[] for _ in range(len(texts))]
        for co, (pool_mat, pool_ids, n_real) in self.parts.items():
            q_idx = np.flatnonzero(countries == co)
            if len(q_idx) == 0:
                continue
            k = min(top_k, n_real)
            q_mat = _parallel_transform(self.vec, texts[q_idx]).astype(np.float32)
            chunk = self._chunk_size(pool_mat.shape[0])
            if spgemm:
                chunk = min(chunk, int(os.environ.get("ER_TFIDF_SPGEMM_CHUNK", "512")))
            starts = range(0, len(q_idx), chunk)
            for start in tqdm(starts, desc=f"{self.name} TF-IDF [{co}]", unit="chunk",
                              leave=False, disable=len(starts) < 2):
                if spgemm:
                    # sparse(pool) x sparse(queries^T): work only where n-grams overlap
                    qT = q_mat[start:start + chunk].T.tocsr()
                    c = qT.shape[1]
                    scores = (pool_mat @ xsp.csr_matrix(qT)).toarray()  # (n_pool_padded, c)
                    dense_q = None
                else:
                    qc = q_mat[start:start + chunk].tocoo()
                    c = qc.shape[0]
                    dense_q = xp.zeros((self.n_features, c), dtype=xp.float32)
                    dense_q[xp.asarray(qc.col), xp.asarray(qc.row)] = xp.asarray(qc.data)
                    scores = pool_mat @ dense_q                          # (n_pool_padded, c)
                # GPU sums in varying order, so near-equal scores flip between runs;
                # rounding turns them into exact ties, broken by pool row below.
                scores = xp.around(scores, 5)
                rows, vals = _block_topk(xp, scores, k)
                rows, vals = _to_host(xp, rows), _to_host(xp, vals)
                del scores, dense_q
                order = np.lexsort((rows, -vals), axis=0)   # best first; ties by pool row
                rows = np.take_along_axis(rows, order, axis=0)
                vals = np.take_along_axis(vals, order, axis=0)
                for j in range(c):
                    keep = rows[:, j][(vals[:, j] > 0) & (rows[:, j] < n_real)]
                    results[q_idx[start + j]] = pool_ids[keep].tolist()
        if is_gpu:
            xp.get_default_memory_pool().free_all_blocks()
        return results

    def release(self):
        """Free device memory held by the pool matrices."""
        self.parts = {}
        xp, _, is_gpu = _backend()
        if is_gpu:
            xp.get_default_memory_pool().free_all_blocks()


# ===== BUILDERS ================================================================

def build_tfidf_blocker(pool_df: pd.DataFrame) -> TfidfIndex:
    """Char 2-4-gram TF-IDF over name + address (primary blocker)."""
    return TfidfIndex("char", TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), max_df=TFIDF_MAX_DF,
        max_features=TFIDF_MAX_FEATURES, sublinear_tf=True, dtype=np.float32,
    ), pool_df, "combined")


def build_word_tfidf_blocker(pool_df: pd.DataFrame) -> TfidfIndex:
    """Word 1-2-gram TF-IDF over name + address (complementary pass)."""
    return TfidfIndex("word", TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_df=TFIDF_MAX_DF,
        max_features=WORD_TFIDF_MAX, sublinear_tf=True, dtype=np.float32,
    ), pool_df, "combined")


def build_name_tfidf_blocker(pool_df: pd.DataFrame) -> TfidfIndex:
    """Char 2-4-gram TF-IDF over the name only, so shared long addresses
    (many businesses in one building) can't drown out the name signal."""
    return TfidfIndex("name", TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), max_df=TFIDF_MAX_DF,
        max_features=NAME_TFIDF_MAX_FEATURES, sublinear_tf=True, dtype=np.float32,
    ), pool_df, "name_clean")
