"""Disk caches: preprocessed-source parquet files and sentence embeddings."""
import hashlib
import os
import time

import numpy as np
import pandas as pd

from .config import CACHE_DIR, EMBED_CACHE_DIR
from .preprocessing import PREPROCESS_VERSION

# ===== DYNAMIC PARQUET CACHE ==================================================
# Key idea: preprocessing regex is the bottleneck (~800s for large files).
# After first run, save preprocessed DataFrames as .parquet files keyed on
# (filename, file_size, mtime, preprocessing version). Subsequent runs load
# parquet in ~10-15s.

def _file_fingerprint(path: str) -> str:
    """Fast fingerprint: basename + size + mtime (+ preprocessing version).
    No content hashing needed because the source TSVs never change between runs."""
    st = os.stat(path)
    raw = f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}"
    if PREPROCESS_VERSION:
        raw += f":{PREPROCESS_VERSION}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_path(path: str, suffix: str = "") -> str:
    """Return the parquet cache file path for a given source TSV."""
    fp = _file_fingerprint(path)
    tag = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(CACHE_DIR, f"{tag}_{fp}{suffix}.parquet")


def cache_load(path: str, suffix: str = "") -> pd.DataFrame | None:
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


def cache_save(path: str, df: pd.DataFrame, suffix: str = "") -> None:
    """Save preprocessed DataFrame to parquet cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = _cache_path(path, suffix)
    tmp = f"{cp}.tmp{os.getpid()}"
    try:
        # Write-then-rename: a session killed mid-write leaves only a .tmp file,
        # never a truncated parquet that later runs would fail to read.
        df.to_parquet(tmp, engine="pyarrow", compression="snappy", index=False)
        os.replace(tmp, cp)
        sz_mb = os.path.getsize(cp) / (1024 * 1024)
        print(f"  [CACHE SAVE] {os.path.basename(cp)} ({sz_mb:.1f} MB)")
    except Exception as e:
        print(f"  [CACHE WARN] Could not save cache: {e}")


def clear_cache():
    """Remove all cached parquet files and embedding caches."""
    if os.path.isdir(CACHE_DIR):
        import shutil
        shutil.rmtree(CACHE_DIR)
        print("[Cache] Cleared all cached data (parquet + embeddings).")


# ===== EMBEDDING DISK CACHE ===================================================
# Saves sentence-transformer embeddings as .npy files keyed on a hash of the
# input texts.  Turns 14-minute encoding steps into ~1s cache loads on repeat
# runs with the same pool data.

def embed_cache_key(texts) -> str:
    """Create a deterministic hash key from the texts being encoded.
    Uses a sample-based hash for speed: first 50, last 50, and length."""
    text_list = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
    n = len(text_list)
    # Sample: first 50 + last 50 + total count for a fast fingerprint
    sample = text_list[:50] + text_list[-50:] if n > 100 else text_list
    raw = f"{n}:" + "|".join(sample)
    return hashlib.md5(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


def embed_cache_load(cache_key: str) -> np.ndarray | None:
    """Try loading cached embeddings from .npy file. Returns None on miss."""
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    path = os.path.join(EMBED_CACHE_DIR, f"emb_{cache_key}.npy")
    if os.path.exists(path):
        t0 = time.time()
        arr = np.load(path)
        sz_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [EMBED CACHE HIT] {cache_key} "
              f"({arr.shape[0]:,} vectors, {sz_mb:.1f} MB in {time.time()-t0:.1f}s)")
        return arr
    return None


def embed_cache_save(cache_key: str, embeddings: np.ndarray) -> None:
    """Save embeddings to .npy cache file."""
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    path = os.path.join(EMBED_CACHE_DIR, f"emb_{cache_key}.npy")
    tmp = f"{path}.tmp{os.getpid()}.npy"
    try:
        np.save(tmp, embeddings)
        os.replace(tmp, path)          # atomic: never a half-written .npy
        sz_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [EMBED CACHE SAVE] {cache_key} ({sz_mb:.1f} MB)")
    except Exception as e:
        print(f"  [EMBED CACHE WARN] Could not save: {e}")
