#!/usr/bin/env python3
"""
Scalable Entity Resolution Pipeline — Parallelised with Workers & Threads.

Key parallelism:
 - ThreadPoolExecutor for concurrent I/O (loading 3 source files in parallel)
 - ProcessPoolExecutor for CPU-heavy work (feature engineering, blocking scoring)
 - Vectorised pandas ops (no row-by-row apply)
 - Batched sparse-matrix operations

Usage:
    python run_pipeline.py --mode train --sample-size 5000
    python run_pipeline.py --mode full
    python run_pipeline.py --mode predict
"""
import argparse
import hashlib
import os
import sys
import time
import gc
import re
import pickle
import warnings
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
from tqdm import tqdm
from difflib import SequenceMatcher

# ---- GPU (CuPy) with graceful CPU fallback -----------------------------------
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    _GPU_OK = cp.cuda.is_available()
except Exception:
    _GPU_OK = False
if _GPU_OK:
    print("[GPU] CuPy detected — TF-IDF matmul will run on GPU")
else:
    print("[GPU] CuPy not available — running TF-IDF matmul on CPU")

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ===== CONFIGURATION ==========================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TRAIN_S1 = os.path.join(BASE_DIR, "dataset", "train", "train_source1.tsv")
TRAIN_S2 = os.path.join(BASE_DIR, "dataset", "train", "train_source2.tsv")
TRAIN_S3 = os.path.join(BASE_DIR, "dataset", "train", "train_source3.tsv")
TRAIN_GT = os.path.join(BASE_DIR, "dataset", "train", "train_ground_truth.tsv")
TEST_S1  = os.path.join(BASE_DIR, "dataset", "test", "test_source1.tsv")
TEST_S2  = os.path.join(BASE_DIR, "dataset", "test", "test_source2.tsv")
TEST_S3  = os.path.join(BASE_DIR, "dataset", "test", "test_source3.tsv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "xgb_model.pkl")
CACHE_DIR  = os.path.join(BASE_DIR, ".cache")       # parquet cache for preprocessed data

TFIDF_TOP_K       = 100             # high blocking recall
TFIDF_MAX_FEATURES= 200_000         # 200K vocab — plenty of RAM
NEG_POS_RATIO     = 3               # balanced negatives
RANDOM_SEED       = 42
VAL_FRACTION      = 0.1
BATCH_SIZE        = 2_000           # smaller batches to cap peak RAM
FEAT_CHUNK        = 25_000          # feature-computation chunk for workers
N_WORKERS         = min(mp.cpu_count(), 16)   # use all 16 cores
BLOCK_THREADS     = 2               # 2 concurrent blocking threads to limit peak RAM
PRED_BATCH_SIZE   = 500_000         # large CPU predict batch
WORD_TFIDF_MAX    = 100_000         # bigger word vocab for better recall
WORD_TFIDF_TOP_K  = 30              # word-level top-K

# config print moved to main() to avoid worker spam

# ===== DYNAMIC PARQUET CACHE ==================================================
# Key idea: preprocessing regex is the bottleneck (~800s for large files).
# After first run, save preprocessed DataFrames as .parquet files keyed on
# (filename, file_size, mtime). Subsequent runs load parquet in ~10-15s.

def _file_fingerprint(path: str) -> str:
    """Fast fingerprint: basename + size + mtime. No content hashing needed
    because the source TSVs never change between runs."""
    st = os.stat(path)
    raw = f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_path(path: str, suffix: str = "") -> str:
    """Return the parquet cache file path for a given source TSV."""
    fp = _file_fingerprint(path)
    tag = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(CACHE_DIR, f"{tag}_{fp}{suffix}.parquet")


def _cache_load(path: str, suffix: str = "") -> pd.DataFrame | None:
    """Try loading preprocessed data from parquet cache. Returns None on miss."""
    cp = _cache_path(path, suffix)
    if os.path.exists(cp):
        t0 = time.time()
        df = pd.read_parquet(cp)
        # Backward compat: ensure new columns exist from older caches
        if "name_sorted_3tok" not in df.columns and "name_clean" in df.columns:
            df["name_sorted_3tok"] = (df["name_clean"]
                .str.split()
                .apply(lambda xs: " ".join(sorted(xs)[:3])
                       if isinstance(xs, list) and len(xs) >= 2 else ""))
        print(f"  [CACHE HIT] {os.path.basename(path)}{suffix} "
              f"({len(df):,} rows in {time.time()-t0:.1f}s)")
        return df
    return None


def _cache_save(path: str, df: pd.DataFrame, suffix: str = "") -> None:
    """Save preprocessed DataFrame to parquet cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = _cache_path(path, suffix)
    try:
        df.to_parquet(cp, engine="pyarrow", compression="snappy", index=False)
        sz_mb = os.path.getsize(cp) / (1024 * 1024)
        print(f"  [CACHE SAVE] {os.path.basename(cp)} ({sz_mb:.1f} MB)")
    except Exception as e:
        print(f"  [CACHE WARN] Could not save cache: {e}")


def _cache_clear():
    """Remove all cached parquet files."""
    if os.path.isdir(CACHE_DIR):
        import shutil
        shutil.rmtree(CACHE_DIR)
        print("[Cache] Cleared all cached data.")

# ===== PRE-COMPILED MEGA-REGEX (compiled ONCE at import time) =================
# Instead of 40+ separate .str.replace() calls (each re-iterating the Series),
# we compile 4 mega-patterns and do 4 single-pass substitutions.

# --- Name: dotted legal abbreviations (order: longest first) ---
_NAME_DOTTED_MAP = {
    "s.a.r.l.": "sarl", "s.a.r.l": "sarl",
    "s.a.s.": "sas",   "s.a.s": "sas",
    "s.c.i.": "sci",   "s.c.i": "sci",
    "l.l.c.": "llc",   "l.l.c": "llc",
    "l.l.p.": "llp",   "l.l.p": "llp",
    "p.l.c.": "plc",   "p.l.c": "plc",
    "l.p.": "lp",      "l.p": "lp",
    "n.a.": "na",      "n.a": "na",
}
_RE_NAME_DOTTED = re.compile(
    "|".join(re.escape(k) for k in sorted(_NAME_DOTTED_MAP, key=len, reverse=True))
)
def _repl_name_dotted(m): return _NAME_DOTTED_MAP[m.group()]

# --- Name: full words + trailing-dot suffixes (single pass) ---
_NAME_WORD_MAP = {
    "incorporated": "inc", "corporation": "corp", "limited": "ltd",
    "company": "co", "private": "pvt",
    "inc.": "inc", "corp.": "corp", "ltd.": "ltd", "pvt.": "pvt",
}
_RE_NAME_WORDS = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_NAME_WORD_MAP, key=len, reverse=True)) + r")\b"
)
def _repl_name_words(m): return _NAME_WORD_MAP[m.group()]

# --- Address: full words + trailing-dot abbreviations (single pass) ---
_ADDR_MAP = {
    # Full words
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "highway": "hwy", "parkway": "pkwy",
    "terrace": "ter", "apartment": "apt", "suite": "ste", "building": "bldg",
    "floor": "fl", "district": "dist", "nagar": "ngr", "sector": "sec",
    "colony": "col",
    # Trailing-dot abbreviations
    "st.": "st", "rd.": "rd", "ave.": "ave", "blvd.": "blvd",
    "dr.": "dr", "apt.": "apt", "ste.": "ste", "bldg.": "bldg",
    "fl.": "fl", "no.": "no",
}
_RE_ADDR = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ADDR_MAP, key=len, reverse=True)) + r")\b"
)
def _repl_addr(m): return _ADDR_MAP[m.group()]

# Common patterns compiled once
_RE_BRACKETS  = re.compile(r"[\[\](){}]")
_RE_DOMAIN    = re.compile(r"\.(com|org|net|in|fr|co\.in)$")
_RE_CO_DOT    = re.compile(r"\bco\.(?=\s|$)")
_RE_MULTI_WS  = re.compile(r"\s+")
_RE_DIGITS    = re.compile(r"\d+")


# ===== FAST VECTORISED PREPROCESSING ==========================================

def fast_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised string cleaning with entity-resolution normalisation.

    Uses pre-compiled mega-regex patterns for single-pass substitutions
    instead of 40+ chained .str.replace() calls.

    Applies:
      1. Basic cleaning (lowercase, bracket removal, &/+ expansion)
      2. Dotted legal abbreviation normalisation (L.L.C. → llc, S.A.R.L. → sarl)
      3. Full-word legal suffix normalisation (Corporation → corp, Private → pvt)
      4. Address abbreviation normalisation (Street → st, Boulevard → blvd)
      5. Whitespace collapse + strip
    """
    df = df.copy()

    # --- Name cleaning (4 passes instead of 22) ---
    name = (df["business_name"]
            .fillna("")
            .str.lower()
            .str.replace(_RE_BRACKETS, " ", regex=True)
            .str.replace("&", " and ", regex=False)
            .str.replace("+", " and ", regex=False))
    # Pass 1: dotted abbreviations (s.a.r.l. → sarl, l.l.c. → llc, etc.)
    name = name.str.replace(_RE_NAME_DOTTED, _repl_name_dotted, regex=True)
    # Pass 2: full-word + trailing-dot suffixes (corporation → corp, inc. → inc)
    name = name.str.replace(_RE_NAME_WORDS, _repl_name_words, regex=True)
    # Pass 3: domain suffixes, co. edge case
    name = name.str.replace(_RE_DOMAIN, "", regex=True)
    name = name.str.replace(_RE_CO_DOT, "co", regex=True)
    df["name_clean"] = name.str.replace(_RE_MULTI_WS, " ", regex=True).str.strip()

    # --- Address cleaning (2 passes instead of 27) ---
    addr = (df["business_address"]
            .fillna("")
            .str.lower()
            .str.replace(_RE_BRACKETS, " ", regex=True)
            .str.replace("&", " and ", regex=False)
            .str.replace("+", " and ", regex=False))
    # Single pass: full words + trailing-dot abbreviations
    addr = addr.str.replace(_RE_ADDR, _repl_addr, regex=True)
    df["addr_clean"] = addr.str.replace(_RE_MULTI_WS, " ", regex=True).str.strip()

    df["country_norm"] = df["country"].fillna("").str.lower().str.strip()
    df["combined"]     = df["name_clean"] + " " + df["addr_clean"]
    # Numeric tokens from address (for blocking key) — vectorised extraction
    df["addr_nums_str"] = (df["addr_clean"]
                           .str.findall(_RE_DIGITS)
                           .apply(lambda xs: " ".join(sorted(set(xs))[:5])
                                  if isinstance(xs, list) else ""))
    # Sorted-token blocking key: sort name tokens, take first 3
    df["name_sorted_3tok"] = (df["name_clean"]
                              .str.split()
                              .apply(lambda xs: " ".join(sorted(xs)[:3])
                                     if isinstance(xs, list) and len(xs) >= 2 else ""))
    return df


def _load_one_source(path: str, use_cache: bool = True) -> pd.DataFrame:
    """Load + preprocess one TSV, with parquet disk caching.

    First call: reads TSV → preprocesses → saves .parquet cache.
    Subsequent calls: loads .parquet directly (~10-20x faster).
    """
    tag = os.path.basename(path)
    t0 = time.time()

    # --- Try cache first ---
    if use_cache:
        cached = _cache_load(path)
        if cached is not None:
            return cached

    # --- Cache miss: load from TSV + preprocess ---
    print(f"  [CACHE MISS] {tag} — loading from TSV...")
    fsize = os.path.getsize(path) if os.path.exists(path) else 0
    if fsize > 20 * 1024 * 1024:
        chunks = []
        for chunk in tqdm(pd.read_csv(path, sep="\t", dtype=str, chunksize=250_000),
                          desc=f"Loading {tag}", unit="chunk"):
            chunks.append(fast_preprocess(chunk))
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(path, sep="\t", dtype=str)
        df = fast_preprocess(df)
    dt = time.time() - t0
    print(f"  [{tag}] {len(df):,} rows in {dt:.1f}s")

    # --- Save to cache for next run ---
    if use_cache:
        _cache_save(path, df)

    return df


def load_sources_parallel(*paths, use_cache: bool = True) -> list:
    """Load multiple source files in parallel using threads (I/O-bound).
    Each file is individually cached as parquet."""
    print(f"[IO] Loading {len(paths)} files in parallel threads...")
    t0 = time.time()
    results = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        futures = {pool.submit(_load_one_source, p, use_cache): i
                   for i, p in enumerate(paths)}
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
    print(f"[IO] All loaded in {time.time()-t0:.1f}s")
    return results


# ===== BLOCKING (TF-IDF + supplementary keys) =================================

def build_tfidf_blocker(pool_df: pd.DataFrame):
    """Fit TF-IDF vectoriser on pool texts."""
    print("[Block] Fitting TF-IDF...")
    t0 = time.time()
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4),
        max_features=TFIDF_MAX_FEATURES, sublinear_tf=True, dtype=np.float32,
    )
    pool_mat = vec.fit_transform(pool_df["combined"].values)
    print(f"[Block] Matrix {pool_mat.shape} in {time.time()-t0:.1f}s")
    return vec, pool_mat


def build_word_tfidf_blocker(pool_df: pd.DataFrame):
    """Word-level TF-IDF for complementary blocking pass."""
    print("[Block] Fitting word-level TF-IDF...")
    t0 = time.time()
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=WORD_TFIDF_MAX, sublinear_tf=True, dtype=np.float32,
    )
    pool_mat = vec.fit_transform(pool_df["combined"].values)
    print(f"[Block] Word matrix {pool_mat.shape} in {time.time()-t0:.1f}s")
    return vec, pool_mat


def _tfidf_query_chunk(args):
    """Score one chunk of S1 queries against the full pool. (worker fn)"""
    q_texts, q_countries, vec_path, pool_mat_path, pool_ids, pool_countries, top_k = args
    with open(vec_path, "rb") as f:
        vec = pickle.load(f)
    with open(pool_mat_path, "rb") as f:
        pool_mat = pickle.load(f)

    q_mat = vec.transform(q_texts)
    scores = q_mat.dot(pool_mat.T)
    results = []
    for i in range(scores.shape[0]):
        row = scores.getrow(i)
        if row.nnz == 0:
            results.append([])
            continue
        idx = row.indices
        dat = row.data
        mask = pool_countries[idx] == q_countries[i]
        fi, fd = idx[mask], dat[mask]
        if len(fd) == 0:
            results.append([])
            continue
        if len(fd) > top_k:
            top = np.argpartition(fd, -top_k)[-top_k:]
            results.append(pool_ids[fi[top]].tolist())
        else:
            results.append(pool_ids[fi].tolist())
    return results


def _matmul_topk(q_sub_mat, pool_sub_mat, pool_sub_ids, effective_k):
    """Sparse matmul + per-row top-K — never materialises a dense matrix.
    RAM-safe: only touches the non-zero entries of each result row.
    """
    sim = q_sub_mat.dot(pool_sub_mat.T)          # sparse × sparse.T → sparse
    if hasattr(sim, 'toarray'):
        # Ensure CSR format for efficient row slicing
        if not isinstance(sim, csr_matrix):
            sim = csr_matrix(sim)
    else:
        # Already dense (shouldn't happen, but handle gracefully)
        sim = csr_matrix(sim)

    results = []
    for i in range(sim.shape[0]):
        row = sim.getrow(i)
        if row.nnz == 0:
            results.append([])
            continue
        idx = row.indices
        dat = row.data
        if len(dat) <= effective_k:
            # Fewer non-zeros than top-K: return all with score > 0
            valid = idx[dat > 0]
            results.append(pool_sub_ids[valid].tolist())
        else:
            # argpartition on the small non-zero array (not the full pool)
            top = np.argpartition(dat, -effective_k)[-effective_k:]
            valid = top[dat[top] > 0]
            results.append(pool_sub_ids[idx[valid]].tolist())
    return results


def tfidf_block_batch(q_texts, q_countries, vec, pool_mat,
                      pool_ids, pool_countries, top_k=TFIDF_TOP_K):
    """TF-IDF blocking — bulk matmul + vectorised top-K per country."""
    q_mat = vec.transform(q_texts)
    n_queries = q_mat.shape[0]
    results = [None] * n_queries

    country_groups = defaultdict(list)
    for i in range(n_queries):
        country_groups[q_countries[i]].append(i)

    CHUNK = 500   # 500 queries per chunk — keeps RAM low (sparse matmul)
    for country, q_indices in country_groups.items():
        pool_mask = pool_countries == country
        if not pool_mask.any():
            for qi in q_indices:
                results[qi] = []
            continue

        pool_sub_mat = pool_mat[pool_mask]
        pool_sub_ids = pool_ids[pool_mask]
        n_pool       = pool_sub_mat.shape[0]
        effective_k  = min(top_k, n_pool)
        q_idx_arr    = np.array(q_indices)
        q_sub_mat    = q_mat[q_idx_arr]

        for chunk_start in range(0, len(q_indices), CHUNK):
            chunk_end     = min(chunk_start + CHUNK, len(q_indices))
            chunk_indices = q_indices[chunk_start:chunk_end]
            chunk_q_mat   = q_sub_mat[chunk_start:chunk_end]

            chunk_results = _matmul_topk(
                chunk_q_mat, pool_sub_mat, pool_sub_ids, effective_k
            )
            for local_i, qi in enumerate(chunk_indices):
                results[qi] = chunk_results[local_i]

    return results


def generate_all_candidates(s1_df, pool_df, vec, pool_mat, top_k=TFIDF_TOP_K,
                            word_vec=None, word_pmat=None):
    """Multi-strategy blocking: char TF-IDF + word TF-IDF + name-key + sorted-token-key + addr-num."""
    pool_ids       = pool_df["entity_id"].values
    pool_countries = pool_df["country_norm"].values

    # ---- supplementary inverted indexes (vectorised via groupby) ----
    print("[Block] Building inverted indexes...")
    t0 = time.time()

    # Name-key index: group by (name_clean[:5], country_norm)
    _pk = pool_df[["entity_id", "name_clean", "country_norm"]].copy()
    _pk["nk"] = _pk["name_clean"].str[:5]
    _pk = _pk[_pk["nk"].str.len() >= 3]
    name_key_idx = _pk.groupby(["nk", "country_norm"])["entity_id"].apply(list).to_dict()

    # Name prefix-4 index (shorter prefix catches more fuzzy matches)
    _p4 = pool_df[["entity_id", "name_clean", "country_norm"]].copy()
    _p4["nk4"] = _p4["name_clean"].str[:4]
    _p4 = _p4[_p4["nk4"].str.len() >= 3]
    name_key4_idx = _p4.groupby(["nk4", "country_norm"])["entity_id"].apply(list).to_dict()

    # Sorted-token key index (catches word reorderings)
    _st = pool_df[["entity_id", "name_sorted_3tok", "country_norm"]].copy()
    _st = _st[_st["name_sorted_3tok"].str.len() > 0]
    sorted_tok_idx_raw = _st.groupby(["name_sorted_3tok", "country_norm"])["entity_id"].apply(list).to_dict()
    sorted_tok_idx = {k: v[:200] for k, v in sorted_tok_idx_raw.items()}

    # Address-numbers index: group by (addr_nums_str, country_norm)
    _an = pool_df[["entity_id", "addr_nums_str", "country_norm"]].copy()
    _an = _an[_an["addr_nums_str"].str.len() > 0]
    addr_num_idx_raw = _an.groupby(["addr_nums_str", "country_norm"])["entity_id"].apply(list).to_dict()
    addr_num_idx = {k: v[:200] for k, v in addr_num_idx_raw.items()}

    print(f"[Block] Indexes built in {time.time()-t0:.1f}s "
          f"(name_keys5={len(name_key_idx):,}, name_keys4={len(name_key4_idx):,}, "
          f"sorted_tok={len(sorted_tok_idx):,}, addr_nums={len(addr_num_idx):,})")

    # ---- TF-IDF blocking in batches (threaded for batch-level parallelism) ----
    s1_ids         = s1_df["entity_id"].values
    s1_texts       = s1_df["combined"].values
    s1_countries   = s1_df["country_norm"].values
    s1_names       = s1_df["name_clean"].values
    s1_addr_nums   = s1_df["addr_nums_str"].values
    s1_sorted_toks = s1_df["name_sorted_3tok"].values

    n = len(s1_df)
    candidates = {}

    eff_batch = min(BATCH_SIZE, n)
    batch_ranges = [(i, min(i + eff_batch, n)) for i in range(0, n, eff_batch)]
    n_threads = min(BLOCK_THREADS, len(batch_ranges))
    print(f"[Block] {len(batch_ranges)} batches, {n:,} queries, "
          f"using {n_threads} threads...")

    def _process_batch(rng):
        bs, be = rng
        cands = tfidf_block_batch(
            s1_texts[bs:be], s1_countries[bs:be],
            vec, pool_mat, pool_ids, pool_countries, top_k,
        )
        return bs, be, cands

    t0_block = time.time()
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(_process_batch, rng) for rng in batch_ranges]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="TF-IDF blocking"):
            bs, be, cands_batch = fut.result()
            for j, s1_idx in enumerate(range(bs, be)):
                sid  = s1_ids[s1_idx]
                cset = set(cands_batch[j])
                co   = s1_countries[s1_idx]
                # supplement: name prefix-5 key
                nm = s1_names[s1_idx]
                if len(nm) >= 3 and (nm[:5], co) in name_key_idx:
                    cset.update(name_key_idx[(nm[:5], co)][:100])
                # supplement: name prefix-4 key (broader)
                if len(nm) >= 3 and (nm[:4], co) in name_key4_idx:
                    cset.update(name_key4_idx[(nm[:4], co)][:100])
                # supplement: sorted-token name key (catches word reorderings)
                stk = s1_sorted_toks[s1_idx]
                if stk and (stk, co) in sorted_tok_idx:
                    cset.update(sorted_tok_idx[(stk, co)][:100])
                # supplement: address numbers
                an = s1_addr_nums[s1_idx]
                if an and (an, co) in addr_num_idx:
                    cset.update(addr_num_idx[(an, co)][:100])
                candidates[sid] = list(cset)

    # ---- Word-level TF-IDF pass (complementary to char n-gram) ----
    if word_vec is not None and word_pmat is not None:
        print("[Block] Word-level TF-IDF pass...")
        tw = time.time()
        word_added = 0

        def _process_word_batch(rng):
            bs, be = rng
            return bs, be, tfidf_block_batch(
                s1_texts[bs:be], s1_countries[bs:be],
                word_vec, word_pmat, pool_ids, pool_countries, WORD_TFIDF_TOP_K,
            )

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(_process_word_batch, rng) for rng in batch_ranges]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Word TF-IDF"):
                bs, be, cands_batch = fut.result()
                for j, s1_idx in enumerate(range(bs, be)):
                    sid = s1_ids[s1_idx]
                    before = len(candidates.get(sid, []))
                    cset = set(candidates.get(sid, []))
                    cset.update(cands_batch[j])
                    candidates[sid] = list(cset)
                    word_added += len(candidates[sid]) - before
        print(f"[Block] Word TF-IDF added {word_added:,} new candidates in {time.time()-tw:.1f}s")

    total = sum(len(v) for v in candidates.values())
    avg   = total / max(len(candidates), 1)
    print(f"[Block] {total:,} candidates ({avg:.1f}/entity) in {time.time()-t0_block:.1f}s")
    gc.collect()   # free TF-IDF intermediates before feature engineering
    return candidates


# ===== FEATURE ENGINEERING =====================================================

def _tokens(t):  return set(t.split()) if t else set()
def _nums(t):    return set(re.findall(r"\d+", t)) if t else set()
def _jac(a, b):
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)
def _ovl(a, b):
    if not a or not b: return 0.0
    return len(a & b) / min(len(a), len(b))
def _dice(a, b):
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return 2*len(a & b) / (len(a)+len(b))
def _lr(a, b):
    la, lb = len(a), len(b)
    if la==0 and lb==0: return 1.0
    if la==0 or  lb==0: return 0.0
    return min(la,lb)/max(la,lb)
def _char_ngrams(s, n=3):
    """Character n-grams as a set."""
    if not s or len(s) < n: return set()
    return {s[i:i+n] for i in range(len(s)-n+1)}
def _containment(s1, s2):
    """Max substring containment ratio."""
    if not s1 and not s2: return 1.0
    if not s1 or not s2:  return 0.0
    if s1 in s2: return len(s1) / len(s2)
    if s2 in s1: return len(s2) / len(s1)
    return 0.0


def compute_pair_features(n1, a1, n2, a2):
    """40 similarity features for one (S1, candidate) pair."""
    nt1, nt2 = _tokens(n1), _tokens(n2)
    at1, at2 = _tokens(a1), _tokens(a2)
    an1, an2 = _nums(a1),   _nums(a2)
    safe_n = (n1 and n2)
    safe_a = (a1 and a2)
    c1, c2 = f"{n1} {a1}", f"{n2} {a2}"
    return [
        # ---- name (19) ----
        1.0 - Levenshtein.normalized_distance(n1, n2) if safe_n else (1.0 if not n1 and not n2 else 0.0),
        JaroWinkler.similarity(n1, n2)                if safe_n else (1.0 if not n1 and not n2 else 0.0),
        fuzz.token_sort_ratio(n1, n2) / 100.0,
        fuzz.token_set_ratio(n1, n2)  / 100.0,
        fuzz.partial_ratio(n1, n2)    / 100.0,
        fuzz.ratio(n1, n2)            / 100.0,
        _jac(nt1, nt2),
        _ovl(nt1, nt2),
        _dice(nt1, nt2),
        len(nt1 & nt2) / max(len(nt1), 1) if nt1 else 0.0,
        len(nt1 & nt2) / max(len(nt2), 1) if nt2 else 0.0,
        1.0 if (nt1 and nt2 and min(nt1) == min(nt2)) else 0.0,
        _lr(n1, n2),
        _containment(n1, n2),
        _jac(_char_ngrams(n1, 3), _char_ngrams(n2, 3)),
        SequenceMatcher(None, n1, n2).ratio() if safe_n else (1.0 if not n1 and not n2 else 0.0),
        1.0 if (nt1 and nt2 and sorted(nt1)[0] == sorted(nt2)[0]) else 0.0,
        min(len(nt1), len(nt2)) / max(len(nt1), len(nt2), 1),
        1.0 if (safe_n and len(n1) >= 3 and len(n2) >= 3 and n1[:3] == n2[:3]) else 0.0,
        # ---- address (11) ----
        1.0 - Levenshtein.normalized_distance(a1, a2) if safe_a else (1.0 if not a1 and not a2 else 0.0),
        JaroWinkler.similarity(a1, a2)                if safe_a else (1.0 if not a1 and not a2 else 0.0),
        fuzz.token_sort_ratio(a1, a2) / 100.0,
        fuzz.token_set_ratio(a1, a2)  / 100.0,
        fuzz.partial_ratio(a1, a2)    / 100.0,
        _jac(at1, at2),
        _ovl(at1, at2),
        _dice(at1, at2),
        _lr(a1, a2),
        _jac(at1 - an1, at2 - an2),
        _containment(a1, a2),
        # ---- address numbers (5) ----
        _jac(an1, an2),
        _ovl(an1, an2),
        len(an1 & an2) / max(len(an1), 1) if an1 else (1.0 if not an2 else 0.5),
        1.0 if (an1 and an2 and sorted(an1)[0] == sorted(an2)[0]) else (1.0 if not an1 and not an2 else 0.0),
        abs(len(an1) - len(an2)),
        # ---- cross (5) ----
        fuzz.token_sort_ratio(c1, c2) / 100.0,
        fuzz.token_set_ratio(c1, c2)  / 100.0,
        _jac(nt1 | at1, nt2 | at2),
        abs(len(nt1) - len(nt2)),
        1.0 - Levenshtein.normalized_distance(c1, c2) if (c1.strip() and c2.strip()) else (1.0 if not c1.strip() and not c2.strip() else 0.0),
    ]

N_FEATURES = 40
FEATURE_NAMES = [
    "name_lev","name_jw","name_tsort","name_tset","name_partial","name_ratio",
    "name_jac","name_ovl","name_dice","name_cont12","name_cont21",
    "name_first","name_lr",
    "name_contain","name_char3_jac","name_lcs","name_first_sorted","name_tok_ratio",
    "name_prefix3",
    "addr_lev","addr_jw","addr_tsort","addr_tset","addr_partial",
    "addr_jac","addr_ovl","addr_dice","addr_lr",
    "addr_nonum_jac","addr_contain",
    "anum_jac","anum_ovl","anum_match12","anum_first","anum_cnt_diff",
    "comb_tsort","comb_tset","comb_jac","name_tok_diff","comb_lev",
]


# ---- parallel feature workers ------------------------------------------------

def _compute_features_chunk(chunk):
    """Worker: compute features for a list of (n1,a1,n2,a2) tuples.
       Returns np.array of shape (len(chunk), N_FEATURES)."""
    out = np.empty((len(chunk), N_FEATURES), dtype=np.float32)
    for i, (n1, a1, n2, a2) in enumerate(chunk):
        out[i] = compute_pair_features(n1, a1, n2, a2)
    return out


def parallel_compute_features(pairs_data: list, desc="Features") -> np.ndarray:
    """
    Compute features for many pairs using ProcessPoolExecutor.
    pairs_data: list of (name1, addr1, name2, addr2) tuples
    """
    n = len(pairs_data)
    if n == 0:
        return np.empty((0, N_FEATURES), dtype=np.float32)

    # Split into chunks for workers
    chunk_size = max(n // N_WORKERS, 500)
    chunks = [pairs_data[i:i+chunk_size] for i in range(0, n, chunk_size)]

    print(f"[Feat] {n:,} pairs -> {len(chunks)} chunks across {N_WORKERS} workers")
    t0 = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(_compute_features_chunk, c) for c in chunks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            results.append(fut.result())

    # Reassemble in order (as_completed may return out of order)
    # Actually we need to keep the original order - let's use map instead
    # But for speed, let's just resubmit with order tracking
    # Simple fix: use indexed approach
    X = np.vstack(results) if results else np.empty((0, N_FEATURES), dtype=np.float32)
    print(f"[Feat] Done in {time.time()-t0:.1f}s, shape={X.shape}")
    return X


def parallel_compute_features_ordered(pairs_data: list, desc="Features") -> np.ndarray:
    """Same as above but preserves order with granular progress reporting."""
    n = len(pairs_data)
    if n == 0:
        return np.empty((0, N_FEATURES), dtype=np.float32)

    chunk_size = max(min(2_500, n // (N_WORKERS * 2)), 500)
    chunks = [(i, pairs_data[i:i+chunk_size]) for i in range(0, n, chunk_size)]

    print(f"[Feat] {n:,} pairs -> {len(chunks)} chunks across {N_WORKERS} workers")
    t0 = time.time()

    ordered_results = [None] * len(chunks)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        future_to_idx = {}
        for ci, (start, chunk) in enumerate(chunks):
            fut = pool.submit(_compute_features_chunk, chunk)
            future_to_idx[fut] = ci
        for fut in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc=desc, unit="chunk"):
            ci = future_to_idx[fut]
            ordered_results[ci] = fut.result()

    X = np.vstack(ordered_results) if ordered_results else np.empty((0, N_FEATURES), dtype=np.float32)
    print(f"[Feat] {X.shape} in {time.time()-t0:.1f}s")
    return X


# ===== TRAINING DATA ==========================================================

def build_training_data(s1_df, pool_df, gt_df, candidates, neg_ratio=NEG_POS_RATIO):
    """Build (X, y) from candidate pairs + ground truth, using parallel features."""
    print("[Train] Assembling pair tuples...")
    t0 = time.time()

    # Parse ground truth
    gt_lookup = {}
    for _, row in gt_df.iterrows():
        sid = row["source1_entity_id"]
        m   = row.get("matched_entity_ids", "")
        gt_lookup[sid] = set(str(m).split(",")) if pd.notna(m) and m else set()

    # Index records
    pool_name = dict(zip(pool_df["entity_id"], pool_df["name_clean"]))
    pool_addr = dict(zip(pool_df["entity_id"], pool_df["addr_clean"]))
    s1_name   = dict(zip(s1_df["entity_id"],   s1_df["name_clean"]))
    s1_addr   = dict(zip(s1_df["entity_id"],   s1_df["addr_clean"]))

    pairs_data = []   # (n1, a1, n2, a2)
    labels     = []
    rng = np.random.RandomState(RANDOM_SEED)

    for s1_id, cands in tqdm(candidates.items(), desc="Pair Assembly", unit="entity"):
        if s1_id not in s1_name or s1_id not in gt_lookup:
            continue
        n1, a1 = s1_name[s1_id], s1_addr[s1_id]
        truth  = gt_lookup[s1_id]

        pos = [c for c in cands if c in truth  and c in pool_name]
        neg = [c for c in cands if c not in truth and c in pool_name]

        max_neg = max(len(pos) * neg_ratio, 2)
        if len(neg) > max_neg:
            neg = rng.choice(neg, size=max_neg, replace=False).tolist()

        for cid in pos:
            pairs_data.append((n1, a1, pool_name[cid], pool_addr[cid]))
            labels.append(1)
        for cid in neg:
            pairs_data.append((n1, a1, pool_name[cid], pool_addr[cid]))
            labels.append(0)

    print(f"[Train] {len(labels):,} pairs assembled in {time.time()-t0:.1f}s")

    # Parallel feature computation
    X = parallel_compute_features_ordered(pairs_data, desc="Train features")
    y = np.array(labels, dtype=np.int32)
    print(f"[Train] Pos: {y.sum():,}, Neg: {(1-y).sum():,}")
    return X, y


# ===== MODEL ===================================================================

def train_xgb(X_train, y_train, X_val=None, y_val=None):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    params = dict(
        objective="binary:logistic", eval_metric="logloss",
        max_depth=10, learning_rate=0.05, n_estimators=800,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=n_neg / max(n_pos, 1),
        tree_method="hist",
        n_jobs=16, random_state=RANDOM_SEED,
        early_stopping_rounds=50,
    )
    model = XGBClassifier(**params, device="cpu")
    print(f"[XGB] Training on CPU (16 cores) with {X_train.shape[0]:,} samples...")
    if X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    else:
        model.fit(X_train, y_train, verbose=50)
    return model


def find_best_threshold(model, X_val, y_val, beta=0.5):
    proba = model.predict_proba(X_val)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val, proba)
    fbeta = np.where(
        (beta**2 * prec + rec) > 0,
        (1 + beta**2) * prec * rec / (beta**2 * prec + rec), 0.0)
    best = np.argmax(fbeta[:-1])
    print(f"[Thresh] Best F_{beta}: {fbeta[best]:.4f} @ {thresholds[best]:.4f}  "
          f"(P={prec[best]:.4f} R={rec[best]:.4f})")
    return float(thresholds[best])


# ===== PREDICTION (parallelised) ==============================================

def predict_all(model, s1_df, pool_df, candidates, threshold):
    """Score all candidate pairs with parallel feature computation."""
    print(f"[Pred] Building pair tuples...")
    t0 = time.time()

    pool_name = dict(zip(pool_df["entity_id"], pool_df["name_clean"]))
    pool_addr = dict(zip(pool_df["entity_id"], pool_df["addr_clean"]))
    s1_name   = dict(zip(s1_df["entity_id"],   s1_df["name_clean"]))
    s1_addr   = dict(zip(s1_df["entity_id"],   s1_df["addr_clean"]))

    pairs_data = []
    pairs_ids  = []    # (s1_id, cand_id)

    for s1_id, cands in tqdm(candidates.items(), desc="Pred Pair Assembly", unit="entity", leave=False):
        if s1_id not in s1_name:
            continue
        n1, a1 = s1_name[s1_id], s1_addr[s1_id]
        for cid in cands:
            if cid in pool_name:
                pairs_data.append((n1, a1, pool_name[cid], pool_addr[cid]))
                pairs_ids.append((s1_id, cid))

    print(f"[Pred] {len(pairs_data):,} pairs in {time.time()-t0:.1f}s")

    # Parallel features
    X = parallel_compute_features_ordered(pairs_data, desc="Pred features")

    # Score in bulk
    print(f"[Pred] Scoring {X.shape[0]:,} pairs @ threshold={threshold:.4f}...")
    proba = model.predict_proba(X)[:, 1]

    matches = {}
    for i, (s1_id, cid) in enumerate(pairs_ids):
        if proba[i] >= threshold:
            matches.setdefault(s1_id, []).append(cid)

    # Ensure all S1 present
    for sid in s1_df["entity_id"].values:
        if sid not in matches:
            matches[sid] = []

    nm = sum(1 for v in matches.values() if v)
    tm = sum(len(v) for v in matches.values())
    print(f"[Pred] {nm:,} matched entities, {tm:,} total matches")
    return matches


# ===== EVALUATION ==============================================================

def _f05(pred, truth, beta=0.5):
    if not truth and not pred: return 1.0
    if not truth and pred:     return 0.0
    if truth and not pred:     return 0.0
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    if p+r == 0: return 0.0
    return (1+beta**2)*p*r / (beta**2*p + r)


def evaluate(predictions, gt_df):
    gt = {}
    for _, row in gt_df.iterrows():
        sid = row["source1_entity_id"]
        m = row.get("matched_entity_ids", "")
        gt[sid] = set(str(m).split(",")) if pd.notna(m) and m else set()
    scores = [_f05(set(predictions.get(sid, [])), truth) for sid, truth in gt.items()]
    macro = np.mean(scores) if scores else 0.0
    print(f"[Eval] Macro F_0.5 = {macro:.4f}  ({len(scores):,} entities)")
    return macro


# ===== OUTPUT ==================================================================

def write_output(matches, candidates, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mr = os.path.join(output_dir, "matching_results.tsv")
    cp = os.path.join(output_dir, "candidate_pairs.tsv")
    with open(mr, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in tqdm(sorted(matches), desc="Writing matching_results.tsv", unit="row"):
            m = ",".join(sorted(set(matches[sid]))) if matches[sid] else ""
            f.write(f"{sid}\t{m}\n")
    with open(cp, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in tqdm(sorted(candidates), desc="Writing candidate_pairs.tsv", unit="row"):
            c = ",".join(sorted(set(candidates[sid]))) if candidates[sid] else ""
            f.write(f"{sid}\t{c}\n")
    print(f"[Out] {mr}\n[Out] {cp}")


# ===== MAIN PIPELINE ==========================================================

def _load_pool_sampled(s2_path, s3_path, must_have_ids, countries,
                       extra_per_country=50_000, seed=RANDOM_SEED):
    """
    Load a manageable subset of S2+S3 that includes:
      1) ALL records whose entity_id is in must_have_ids (ground-truth matches)
      2) A random sample of extra_per_country per country (noise / negatives)

    Leverages parquet cache: loads full preprocessed source files from cache
    (fast), then samples in-memory instead of re-preprocessing from TSV.
    """
    print(f"[Pool] Loading sampled pool (must_have={len(must_have_ids):,}, "
          f"extra/country={extra_per_country:,})...")
    t0 = time.time()
    must = set(must_have_ids)
    frames_must = []
    frames_rand = {c: [] for c in countries}
    rand_counts = {c: 0 for c in countries}
    rng_pool = np.random.RandomState(seed)

    for path in (s2_path, s3_path):
        tag = os.path.basename(path)

        # --- Try loading full preprocessed file from cache ---
        cached = _cache_load(path)
        if cached is not None:
            # Fast path: sample from cached full DataFrame
            mask_must = cached["entity_id"].isin(must)
            if mask_must.any():
                frames_must.append(cached[mask_must])
            rest = cached[~mask_must]
            for co in countries:
                if rand_counts[co] >= extra_per_country:
                    continue
                co_rows = rest[rest["country_norm"] == co]
                need = extra_per_country - rand_counts[co]
                if len(co_rows) > need:
                    co_rows = co_rows.sample(n=need, random_state=rng_pool)
                if len(co_rows) > 0:
                    frames_rand[co].append(co_rows)
                    rand_counts[co] += len(co_rows)
            del cached, rest; gc.collect()
            print(f"  [{tag}] sampled from cache")
            continue

        # --- Cache miss: chunked reading + preprocess + build cache ---
        print(f"  [CACHE MISS] {tag} — scanning chunks from TSV...")
        all_chunks = []  # collect all preprocessed chunks for caching
        for chunk in tqdm(pd.read_csv(path, sep="\t", dtype=str, chunksize=200_000),
                          desc=f"Scanning {tag}", unit="chunk"):
            chunk = fast_preprocess(chunk)
            all_chunks.append(chunk)
            # Must-have rows
            mask_must = chunk["entity_id"].isin(must)
            if mask_must.any():
                frames_must.append(chunk[mask_must])
            # Random sample rows (country-filtered, excluding must-haves)
            rest = chunk[~mask_must]
            for co in countries:
                if rand_counts[co] >= extra_per_country:
                    continue
                co_rows = rest[rest["country_norm"] == co]
                need = extra_per_country - rand_counts[co]
                if len(co_rows) > need:
                    co_rows = co_rows.sample(n=need, random_state=rng_pool)
                if len(co_rows) > 0:
                    frames_rand[co].append(co_rows)
                    rand_counts[co] += len(co_rows)

        # Save full preprocessed file to cache for next run
        full_df = pd.concat(all_chunks, ignore_index=True)
        _cache_save(path, full_df)
        del all_chunks, full_df; gc.collect()
        print(f"  [{tag}] scanned + cached")

    all_frames = frames_must
    for co in countries:
        all_frames.extend(frames_rand[co])
    pool = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset="entity_id")
    print(f"[Pool] {len(pool):,} records in {time.time()-t0:.1f}s")
    return pool


def run_train(sample_size):
    print("="*72)
    print(f" TRAIN  (sample={sample_size:,}, workers={N_WORKERS})")
    print("="*72)
    t_all = time.time()

    # ---- 1. Load S1 + GT (fast, small-ish files) ----
    print("[Data] Loading S1 + GT...")
    t0 = time.time()
    s1_full = _load_one_source(TRAIN_S1)
    gt_full = pd.read_csv(TRAIN_GT, sep="\t", dtype=str)
    print(f"[Data] S1: {len(s1_full):,}, GT: {len(gt_full):,} in {time.time()-t0:.1f}s")

    # ---- 2. Sample S1 ----
    rng = np.random.RandomState(RANDOM_SEED)
    if sample_size is None or sample_size <= 0 or sample_size >= len(s1_full):
        print(f"[Data] Using FULL dataset ({len(s1_full):,} entities)")
        s1s = s1_full.copy()
        gts = gt_full.copy()
        extra = 500_000
    else:
        ids = rng.choice(s1_full["entity_id"].values,
                         size=min(sample_size, len(s1_full)), replace=False)
        s1s = s1_full[s1_full["entity_id"].isin(set(ids))].copy()
        gts = gt_full[gt_full["source1_entity_id"].isin(set(ids))].copy()
        extra = 30_000  # 30K extra/country — keeps pool manageable
        # Pool size = must_haves + 30K × n_countries ≈ 100-150K records max
        # At 150K pool: 2 threads × 500 queries × 150K × 4B = ~600MB ✔
    countries = set(s1s["country_norm"].unique())
    del s1_full, gt_full; gc.collect()

    # ---- 3. Find all S2/S3 IDs referenced in GT (must-have for pool) ----
    must_have = set()
    for _, row in gts.iterrows():
        m = row.get("matched_entity_ids", "")
        if pd.notna(m) and m:
            must_have.update(str(m).split(","))
    print(f"[Data] S1 sample: {len(s1s):,}, must-have pool IDs: {len(must_have):,}")

    # ---- 4. Smart pool loading (threaded, chunked) ----
    pool = _load_pool_sampled(TRAIN_S2, TRAIN_S3, must_have, countries,
                              extra_per_country=extra)

    # ---- 5. Train/val split ----
    ids2 = s1s["entity_id"].values.copy(); rng.shuffle(ids2)
    nv = max(int(len(ids2) * VAL_FRACTION), 10)
    val_ids   = set(ids2[:nv])
    train_ids = set(ids2[nv:])
    s1_tr = s1s[s1s["entity_id"].isin(train_ids)]
    s1_va = s1s[s1s["entity_id"].isin(val_ids)]
    gt_tr = gts[gts["source1_entity_id"].isin(train_ids)]
    gt_va = gts[gts["source1_entity_id"].isin(val_ids)]
    print(f"[Split] Train: {len(s1_tr):,}, Val: {len(s1_va):,}")

    # ---- 6. Blocking ----
    vec, pmat = build_tfidf_blocker(pool)
    word_vec, word_pmat = build_word_tfidf_blocker(pool)
    tr_cands  = generate_all_candidates(s1_tr, pool, vec, pmat, word_vec=word_vec, word_pmat=word_pmat)
    va_cands  = generate_all_candidates(s1_va, pool, vec, pmat, word_vec=word_vec, word_pmat=word_pmat)

    # Blocking recall
    gt_va_lk = {}
    for _, r in gt_va.iterrows():
        sid = r["source1_entity_id"]; m = r.get("matched_entity_ids","")
        gt_va_lk[sid] = set(str(m).split(",")) if pd.notna(m) and m else set()
    found = total = 0
    for sid, truth in gt_va_lk.items():
        if truth:
            found += len(truth & set(va_cands.get(sid,[])))
            total += len(truth)
    print(f"[Block] Val blocking recall: {found/total:.4f} ({found}/{total})" if total else "[Block] no val matches")

    # ---- 7. Features + training ----
    X_tr, y_tr = build_training_data(s1_tr, pool, gt_tr, tr_cands)
    X_va, y_va = build_training_data(s1_va, pool, gt_va, va_cands)
    model = train_xgb(X_tr, y_tr, X_va, y_va)

    # top features
    imp = model.feature_importances_
    for name, sc in sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])[:10]:
        print(f"  {name}: {sc:.4f}")

    # ---- 8. Threshold ----
    threshold = find_best_threshold(model, X_va, y_va)

    # ---- 9. Evaluate ----
    va_matches = predict_all(model, s1_va, pool, va_cands, threshold)
    f_score = evaluate(va_matches, gt_va)

    print(f"\n{'='*72}")
    print(f"  VALIDATION F_0.5 = {f_score:.4f}   ({time.time()-t_all:.0f}s total)")
    print(f"{'='*72}\n")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "threshold": threshold}, f)
    print(f"[Model] Saved -> {MODEL_PATH}")
    return model, threshold



def _predict_test_by_country(model, threshold):
    """
    Predict on the test set country-by-country to manage memory.
    For each country: filter cached pool subset, block, predict, collect results.

    OPTIMIZATION: Test S2/S3 are preprocessed ONCE (and cached), then filtered
    per country in-memory. Previously we re-read + re-preprocessed the full
    TSVs for EACH country (3× redundant I/O + regex work).
    """
    print("\n" + "="*72)
    print(" TEST PREDICTION (country-by-country)")
    print("="*72)
    t0 = time.time()

    # Load S1 test (smallest file, cached)
    ts1 = _load_one_source(TEST_S1)
    countries = sorted(ts1["country_norm"].unique())
    print(f"[Test] S1: {len(ts1):,} entities, countries: {countries}")

    # Preprocess full test S2 + S3 ONCE (cached on disk after first run)
    print("[Test] Loading full test pool (preprocessed + cached)...")
    t_pool = time.time()
    pool_parts = []
    for path in (TEST_S2, TEST_S3):
        pool_parts.append(_load_one_source(path, use_cache=True))
    test_pool_full = pd.concat(pool_parts, ignore_index=True)
    del pool_parts; gc.collect()
    print(f"[Test] Full pool: {len(test_pool_full):,} records in {time.time()-t_pool:.1f}s")

    all_matches = {}
    all_candidates = {}

    for country in countries:
        print(f"\n--- Processing country: {country.upper()} ---")
        tc = time.time()
        s1_co = ts1[ts1["country_norm"] == country].copy()
        print(f"[Test/{country}] S1 entities: {len(s1_co):,}")

        # Filter pool for this country (fast: in-memory boolean mask)
        pool_co = test_pool_full[test_pool_full["country_norm"] == country].copy()
        print(f"[Test/{country}] Pool: {len(pool_co):,}")

        if len(pool_co) == 0:
            for sid in s1_co["entity_id"].values:
                all_matches[sid] = []
                all_candidates[sid] = []
            continue

        # Block
        vec, pmat = build_tfidf_blocker(pool_co)
        word_vec, word_pmat = build_word_tfidf_blocker(pool_co)
        cands = generate_all_candidates(s1_co, pool_co, vec, pmat, word_vec=word_vec, word_pmat=word_pmat)

        # Predict
        matches = predict_all(model, s1_co, pool_co, cands, threshold)

        # Collect
        for sid in s1_co["entity_id"].values:
            all_matches[sid] = matches.get(sid, [])
            all_candidates[sid] = cands.get(sid, [])

        del pool_co, vec, pmat, cands, matches; gc.collect()
        print(f"[Test/{country}] Done in {time.time()-tc:.0f}s")

    del test_pool_full; gc.collect()

    # Write
    write_output(all_matches, all_candidates, OUTPUT_DIR)
    print(f"\n[Test] TOTAL TIME: {time.time()-t0:.0f}s")
    return all_matches, all_candidates


def run_full(sample_size=50_000):
    model, threshold = run_train(sample_size=sample_size)
    _predict_test_by_country(model, threshold)


def run_predict():
    with open(MODEL_PATH, "rb") as f:
        d = pickle.load(f)
    model, threshold = d["model"], d["threshold"]
    print(f"[Model] loaded, threshold={threshold:.4f}")
    _predict_test_by_country(model, threshold)


def main():
    print(f"[Config] Workers: {N_WORKERS}, CPUs: {mp.cpu_count()}, "
          f"Cache: {CACHE_DIR}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","full","predict"], default="train")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete all cached .parquet files before running")
    args = parser.parse_args()
    if args.clear_cache:
        _cache_clear()
    {"train": lambda: run_train(args.sample_size),
     "full":  lambda: run_full(args.sample_size),
     "predict": run_predict}[args.mode]()


if __name__ == "__main__":
    mp.freeze_support()       # needed on Windows
    main()
