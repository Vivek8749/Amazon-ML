#!/usr/bin/env python3
"""
Scalable Entity Resolution Pipeline ΓÇö Parallelised with Workers & Threads.

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
import hashlib

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ===== CONFIGURATION ==========================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_S1 = os.path.join(BASE_DIR, "student_resource", "dataset", "train", "train_source1.tsv")
TRAIN_S2 = os.path.join(BASE_DIR, "student_resource", "dataset", "train", "train_source2.tsv")
TRAIN_S3 = os.path.join(BASE_DIR, "student_resource", "dataset", "train", "train_source3.tsv")
TRAIN_GT = os.path.join(BASE_DIR, "student_resource", "dataset", "train", "train_ground_truth.tsv")
TEST_S1  = os.path.join(BASE_DIR, "student_resource", "dataset", "test", "test_source1.tsv")
TEST_S2  = os.path.join(BASE_DIR, "student_resource", "dataset", "test", "test_source2.tsv")
TEST_S3  = os.path.join(BASE_DIR, "student_resource", "dataset", "test", "test_source3.tsv")
OUTPUT_DIR = os.path.join(BASE_DIR, "student_resource", "output")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "xgb_model.pkl")

TFIDF_TOP_K       = 20
TFIDF_MAX_FEATURES= 200_000
NEG_POS_RATIO     = 3
RANDOM_SEED       = 42
VAL_FRACTION      = 0.1
BATCH_SIZE        = 1_000       # TF-IDF query batch (fine-grained for smooth progress)
FEAT_CHUNK        = 25_000      # feature-computation chunk for workers
N_WORKERS         = min(mp.cpu_count(), 16)   # cap so we don't OOM
CACHE_DIR         = os.path.join(BASE_DIR, ".cache")  # parquet cache for preprocessed data

# config print moved to main() to avoid worker spam


# ===== PARQUET DISK CACHE =====================================================
# Preprocessing regex is expensive (~minutes for large files).
# After first run, save preprocessed DataFrames as .parquet files.
# Subsequent runs load parquet in ~5-15s instead of re-processing.

def _file_fingerprint(path: str) -> str:
    """Fast fingerprint: basename + size + mtime."""
    st = os.stat(path)
    raw = f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_path(path: str) -> str:
    fp = _file_fingerprint(path)
    tag = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(CACHE_DIR, f"{tag}_{fp}.parquet")


def _cache_load(path: str) -> pd.DataFrame | None:
    """Try loading preprocessed data from parquet cache."""
    cp = _cache_path(path)
    if os.path.exists(cp):
        t0 = time.time()
        df = pd.read_parquet(cp)
        print(f"  [CACHE HIT] {os.path.basename(path)} "
              f"({len(df):,} rows in {time.time()-t0:.1f}s)")
        return df
    return None


def _cache_save(path: str, df: pd.DataFrame) -> None:
    """Save preprocessed DataFrame to parquet cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = _cache_path(path)
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


# ===== FAST VECTORISED PREPROCESSING ==========================================

def fast_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised string cleaning with entity-resolution normalisation.

    Applies:
      1. Basic cleaning (lowercase, bracket removal, &/+ expansion)
      2. Dotted legal abbreviation normalisation (L.L.C. ΓåÆ llc, S.A.R.L. ΓåÆ sarl)
      3. Full-word legal suffix normalisation (Corporation ΓåÆ corp, Private ΓåÆ pvt)
      4. Address abbreviation normalisation (Street ΓåÆ st, Boulevard ΓåÆ blvd)
      5. Whitespace collapse + strip
    """
    df = df.copy()

    # --- Name cleaning ---
    name = (df["business_name"]
            .fillna("")
            .str.lower()
            .str.replace(r"[\[\]\(\)\{\}]", " ", regex=True)
            .str.replace("&", " and ", regex=False)
            .str.replace("+", " and ", regex=False))
    # Dotted legal abbreviations (order matters: longer patterns first)
    name = (name
            .str.replace(r"s\.a\.r\.l\.?", "sarl", regex=True)
            .str.replace(r"s\.a\.s\.?",   "sas",  regex=True)
            .str.replace(r"s\.c\.i\.?",   "sci",  regex=True)
            .str.replace(r"l\.l\.c\.?",   "llc",  regex=True)
            .str.replace(r"l\.l\.p\.?",   "llp",  regex=True)
            .str.replace(r"p\.l\.c\.?",   "plc",  regex=True)
            .str.replace(r"l\.p\.?",       "lp",   regex=True)
            .str.replace(r"n\.a\.?",       "na",   regex=True))
    # Full-word legal suffixes
    name = (name
            .str.replace(r"\bincorporated\b", "inc",  regex=True)
            .str.replace(r"\bcorporation\b",  "corp", regex=True)
            .str.replace(r"\blimited\b",      "ltd",  regex=True)
            .str.replace(r"\bcompany\b",      "co",   regex=True)
            .str.replace(r"\bprivate\b",      "pvt",  regex=True))
    # Trailing-dot abbreviated suffixes (Inc. ΓåÆ inc, Corp. ΓåÆ corp, etc.)
    name = (name
            .str.replace(r"\binc\.",  "inc",  regex=True)
            .str.replace(r"\bcorp\.", "corp", regex=True)
            .str.replace(r"\bltd\.",  "ltd",  regex=True)
            .str.replace(r"\bpvt\.",  "pvt",  regex=True)
            .str.replace(r"\bco\.(?=\s|$)",  "co",   regex=True))
    # Remove .com/.org/.net/.in/.fr domain suffixes from names
    name = name.str.replace(r"\.(com|org|net|in|fr|co\.in)$", "", regex=True)
    df["name_clean"] = name.str.replace(r"\s+", " ", regex=True).str.strip()

    # --- Address cleaning ---
    addr = (df["business_address"]
            .fillna("")
            .str.lower()
            .str.replace(r"[\[\]\(\)\{\}]", " ", regex=True)
            .str.replace("&", " and ", regex=False)
            .str.replace("+", " and ", regex=False))
    # Address word-level abbreviation normalisation
    addr = (addr
            .str.replace(r"\bstreet\b",    "st",   regex=True)
            .str.replace(r"\broad\b",      "rd",   regex=True)
            .str.replace(r"\bavenue\b",    "ave",  regex=True)
            .str.replace(r"\bboulevard\b", "blvd", regex=True)
            .str.replace(r"\bdrive\b",     "dr",   regex=True)
            .str.replace(r"\blane\b",      "ln",   regex=True)
            .str.replace(r"\bhighway\b",   "hwy",  regex=True)
            .str.replace(r"\bparkway\b",   "pkwy", regex=True)
            .str.replace(r"\bterrace\b",   "ter",  regex=True)
            .str.replace(r"\bapartment\b", "apt",  regex=True)
            .str.replace(r"\bsuite\b",     "ste",  regex=True)
            .str.replace(r"\bbuilding\b",  "bldg", regex=True)
            .str.replace(r"\bfloor\b",     "fl",   regex=True)
            .str.replace(r"\bdistrict\b",  "dist", regex=True)
            .str.replace(r"\bnagar\b",     "ngr",  regex=True)
            .str.replace(r"\bsector\b",    "sec",  regex=True)
            .str.replace(r"\bcolony\b",    "col",  regex=True))
    # Trailing-dot address abbreviations
    addr = (addr
            .str.replace(r"\bst\.",   "st",   regex=True)
            .str.replace(r"\brd\.",   "rd",   regex=True)
            .str.replace(r"\bave\.",  "ave",  regex=True)
            .str.replace(r"\bblvd\.", "blvd", regex=True)
            .str.replace(r"\bdr\.",   "dr",   regex=True)
            .str.replace(r"\bapt\.",  "apt",  regex=True)
            .str.replace(r"\bste\.",  "ste",  regex=True)
            .str.replace(r"\bbldg\.", "bldg", regex=True)
            .str.replace(r"\bfl\.",   "fl",   regex=True)
            .str.replace(r"\bno\.",   "no",   regex=True))
    df["addr_clean"] = addr.str.replace(r"\s+", " ", regex=True).str.strip()

    df["country_norm"] = df["country"].fillna("").str.lower().str.strip()
    df["combined"]     = df["name_clean"] + " " + df["addr_clean"]
    # Numeric tokens from address (for blocking key)
    df["addr_nums_str"] = (df["addr_clean"]
                           .str.findall(r"\d+")
                           .apply(lambda xs: " ".join(sorted(set(xs))[:5])
                                  if isinstance(xs, list) else ""))
    return df


def _load_one_source(path: str, use_cache: bool = True) -> pd.DataFrame:
    """Load + preprocess one TSV, with parquet disk caching."""
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


def load_sources_parallel(*paths) -> list:
    """Load multiple source files in parallel using threads (I/O-bound)."""
    print(f"[IO] Loading {len(paths)} files in parallel threads...")
    t0 = time.time()
    results = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        futures = {pool.submit(_load_one_source, p): i for i, p in enumerate(paths)}
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
    # Ensure int32 indices to save memory (scipy default can be int64 on large data)
    pool_mat.indptr  = pool_mat.indptr.astype(np.int32)
    pool_mat.indices = pool_mat.indices.astype(np.int32)
    print(f"[Block] Matrix {pool_mat.shape} in {time.time()-t0:.1f}s")
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


def tfidf_block_batch(q_texts, q_countries, vec, pool_mat,
                      pool_ids, pool_countries, top_k=TFIDF_TOP_K):
    """TF-IDF blocking ΓÇö row-by-row sparse queries to avoid OOM on large pools."""
    q_mat = vec.transform(q_texts)
    results = []
    chunk_size = 200  # mini-batch for fast C++ sparse BLAS without OOM
    n_queries = q_mat.shape[0]

    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        sub_sim = q_mat[start:end].dot(pool_mat.T)  # shape (chunk_size, pool_size)
        
        for local_i in range(end - start):
            global_i = start + local_i
            row = sub_sim.getrow(local_i)
            if row.nnz == 0:
                results.append([])
                continue
            idx, dat = row.indices, row.data
            mask = pool_countries[idx] == q_countries[global_i]
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


def generate_all_candidates(s1_df, pool_df, vec, pool_mat, top_k=TFIDF_TOP_K, n_workers=None):
    """Multi-strategy blocking with TF-IDF + name-key + addr-num indexes."""
    pool_ids       = pool_df["entity_id"].values
    pool_countries = pool_df["country_norm"].values

    # ---- supplementary inverted indexes (built on main thread) ----
    print("[Block] Building inverted indexes...")
    t0 = time.time()
    name_key_idx = defaultdict(list)
    addr_num_idx = defaultdict(list)
    for row in tqdm(pool_df.itertuples(index=False), total=len(pool_df),
                    desc="Indexing Pool Keys", unit="rec"):
        nm = row.name_clean
        co = row.country_norm
        if len(nm) >= 3:
            name_key_idx[(nm[:5], co)].append(row.entity_id)
        an = row.addr_nums_str
        if an:
            if len(addr_num_idx[(an, co)]) < 100:
                addr_num_idx[(an, co)].append(row.entity_id)
    print(f"[Block] Indexes built in {time.time()-t0:.1f}s")

    # ---- TF-IDF blocking in batches (threaded for batch-level parallelism) ----
    s1_ids       = s1_df["entity_id"].values
    s1_texts     = s1_df["combined"].values
    s1_countries = s1_df["country_norm"].values
    s1_names     = s1_df["name_clean"].values
    s1_addr_nums = s1_df["addr_nums_str"].values

    n = len(s1_df)
    candidates = {}

    # We use ThreadPoolExecutor for batch-level concurrency on TF-IDF queries
    # (each batch involves scipy sparse ops that release the GIL internally)
    if n_workers is None:
        n_workers = N_WORKERS
    batch_ranges = [(i, min(i + BATCH_SIZE, n)) for i in range(0, n, BATCH_SIZE)]
    print(f"[Block] {len(batch_ranges)} batches, {n:,} queries, "
          f"using {min(n_workers, len(batch_ranges))} threads...")

    def _process_batch(rng):
        bs, be = rng
        cands = tfidf_block_batch(
            s1_texts[bs:be], s1_countries[bs:be],
            vec, pool_mat, pool_ids, pool_countries, top_k,
        )
        return bs, be, cands

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(n_workers, len(batch_ranges))) as pool:
        futures = [pool.submit(_process_batch, rng) for rng in batch_ranges]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="TF-IDF blocking"):
            bs, be, cands_batch = fut.result()
            for j, s1_idx in enumerate(range(bs, be)):
                sid  = s1_ids[s1_idx]
                cset = set(cands_batch[j])
                # supplement: name key
                nm = s1_names[s1_idx]
                co = s1_countries[s1_idx]
                if len(nm) >= 3 and (nm[:5], co) in name_key_idx:
                    cset.update(name_key_idx[(nm[:5], co)][:50])
                # supplement: address numbers
                an = s1_addr_nums[s1_idx]
                if an and (an, co) in addr_num_idx:
                    cset.update(addr_num_idx[(an, co)][:50])
                candidates[sid] = list(cset)

    total = sum(len(v) for v in candidates.values())
    avg   = total / max(len(candidates), 1)
    print(f"[Block] {total:,} candidates ({avg:.1f}/entity) in {time.time()-t0:.1f}s")
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


def compute_pair_features(n1, a1, n2, a2):
    """29 similarity features for one (S1, candidate) pair."""
    nt1, nt2 = _tokens(n1), _tokens(n2)
    at1, at2 = _tokens(a1), _tokens(a2)
    an1, an2 = _nums(a1),   _nums(a2)
    safe_n = (n1 and n2)
    safe_a = (a1 and a2)
    return [
        # ---- name (13) ----
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
        # ---- address (9) ----
        1.0 - Levenshtein.normalized_distance(a1, a2) if safe_a else (1.0 if not a1 and not a2 else 0.0),
        JaroWinkler.similarity(a1, a2)                if safe_a else (1.0 if not a1 and not a2 else 0.0),
        fuzz.token_sort_ratio(a1, a2) / 100.0,
        fuzz.token_set_ratio(a1, a2)  / 100.0,
        fuzz.partial_ratio(a1, a2)    / 100.0,
        _jac(at1, at2),
        _ovl(at1, at2),
        _dice(at1, at2),
        _lr(a1, a2),
        # ---- address numbers (3) ----
        _jac(an1, an2),
        _ovl(an1, an2),
        len(an1 & an2) / max(len(an1), 1) if an1 else (1.0 if not an2 else 0.5),
        # ---- cross (4) ----
        fuzz.token_sort_ratio(f"{n1} {a1}", f"{n2} {a2}") / 100.0,
        fuzz.token_set_ratio(f"{n1} {a1}", f"{n2} {a2}")  / 100.0,
        _jac(nt1 | at1, nt2 | at2),
        abs(len(nt1) - len(nt2)),
    ]

N_FEATURES = 29
FEATURE_NAMES = [
    "name_lev","name_jw","name_tsort","name_tset","name_partial","name_ratio",
    "name_jac","name_ovl","name_dice","name_cont12","name_cont21",
    "name_first","name_lr",
    "addr_lev","addr_jw","addr_tsort","addr_tset","addr_partial",
    "addr_jac","addr_ovl","addr_dice","addr_lr",
    "anum_jac","anum_ovl","anum_match12",
    "comb_tsort","comb_tset","comb_jac","name_tok_diff",
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
    model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        max_depth=8, learning_rate=0.1, n_estimators=500,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=n_neg / max(n_pos, 1),
        device="cuda", tree_method="hist",    # GPU-accelerated training
        n_jobs=-1, random_state=RANDOM_SEED,
        early_stopping_rounds=30,
    )
    print(f"[XGB] Training on GPU (CUDA) with {X_train.shape[0]:,} samples...")
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
    Uses chunked reading so we never hold all 5M rows at once.
    """
    print(f"[Pool] Loading sampled pool (must_have={len(must_have_ids):,}, "
          f"extra/country={extra_per_country:,})...")
    t0 = time.time()
    must = set(must_have_ids)
    frames_must = []
    frames_rand = {c: [] for c in countries}
    rand_counts = {c: 0 for c in countries}

    for path in (s2_path, s3_path):
        tag = os.path.basename(path)
        for chunk in tqdm(pd.read_csv(path, sep="\t", dtype=str, chunksize=200_000),
                          desc=f"Scanning {tag}", unit="chunk"):
            chunk = fast_preprocess(chunk)
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
                    co_rows = co_rows.sample(n=need, random_state=seed)
                if len(co_rows) > 0:
                    frames_rand[co].append(co_rows)
                    rand_counts[co] += len(co_rows)
        print(f"  [{tag}] scanned")

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
    ids = rng.choice(s1_full["entity_id"].values,
                     size=min(sample_size, len(s1_full)), replace=False)
    s1s = s1_full[s1_full["entity_id"].isin(set(ids))].copy()
    gts = gt_full[gt_full["source1_entity_id"].isin(set(ids))].copy()
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
    # Scale extra noise records with sample size
    extra = max(sample_size * 10, 50_000)
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
    tr_cands  = generate_all_candidates(s1_tr, pool, vec, pmat)
    va_cands  = generate_all_candidates(s1_va, pool, vec, pmat)

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
    For each country: load relevant pool subset, block, predict, collect results.
    Saves per-country checkpoints so a killed run can resume.
    """
    print("\n" + "="*72)
    print(" TEST PREDICTION (country-by-country)")
    print("="*72)
    t0 = time.time()

    # Checkpoint directory
    ckpt_dir = os.path.join(OUTPUT_DIR, "_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load S1 test (smallest file)
    ts1 = _load_one_source(TEST_S1)
    countries = sorted(ts1["country_norm"].unique())
    print(f"[Test] S1: {len(ts1):,} entities, countries: {countries}")

    all_matches = {}
    all_candidates = {}

    # Load any previously completed country checkpoints
    for country in countries:
        ckpt_path = os.path.join(ckpt_dir, f"{country}.pkl")
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "rb") as f:
                saved = pickle.load(f)
            all_matches.update(saved["matches"])
            all_candidates.update(saved["candidates"])
            print(f"[CHECKPOINT] {country.upper()} loaded from cache "
                  f"({len(saved['matches']):,} entities)")

    for country in countries:
        # Skip if already checkpointed
        ckpt_path = os.path.join(ckpt_dir, f"{country}.pkl")
        if os.path.exists(ckpt_path):
            print(f"\n--- Skipping country: {country.upper()} (checkpoint exists) ---")
            continue

        print(f"\n--- Processing country: {country.upper()} ---")
        tc = time.time()
        s1_co = ts1[ts1["country_norm"] == country].copy()
        print(f"[Test/{country}] S1 entities: {len(s1_co):,}")

        # Load pool for this country (use cache for fast restarts)
        pool_frames = []
        for path in (TEST_S2, TEST_S3):
            full_df = _load_one_source(path, use_cache=True)
            co_chunk = full_df[full_df["country_norm"] == country]
            if len(co_chunk) > 0:
                pool_frames.append(co_chunk)
            del full_df; gc.collect()

        pool_co = pd.concat(pool_frames, ignore_index=True) if pool_frames else pd.DataFrame()
        del pool_frames; gc.collect()
        print(f"[Test/{country}] Pool: {len(pool_co):,}")

        country_matches = {}
        country_candidates = {}

        if len(pool_co) == 0:
            for sid in s1_co["entity_id"].values:
                country_matches[sid] = []
                country_candidates[sid] = []
        else:
            # Block
            vec, pmat = build_tfidf_blocker(pool_co)
            cands = generate_all_candidates(s1_co, pool_co, vec, pmat)

            # Predict
            matches = predict_all(model, s1_co, pool_co, cands, threshold)

            # Collect
            for sid in s1_co["entity_id"].values:
                country_matches[sid] = matches.get(sid, [])
                country_candidates[sid] = cands.get(sid, [])

            del pool_co, vec, pmat, cands, matches; gc.collect()

        # Save checkpoint for this country
        with open(ckpt_path, "wb") as f:
            pickle.dump({"matches": country_matches,
                         "candidates": country_candidates}, f)
        print(f"[CHECKPOINT] {country.upper()} saved ({len(country_matches):,} entities)")

        all_matches.update(country_matches)
        all_candidates.update(country_candidates)
        del country_matches, country_candidates; gc.collect()
        print(f"[Test/{country}] Done in {time.time()-tc:.0f}s")

    # Write final output
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
