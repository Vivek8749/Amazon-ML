"""Data loading (with caching), training-pool sampling, ground truth and output writing."""
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from .cache import cache_load, cache_save
from .config import RANDOM_SEED
from .preprocessing import fast_preprocess


# ===== LOADING ================================================================

def load_source(path: str, use_cache: bool = True) -> pd.DataFrame:
    """Load + preprocess one TSV, with parquet disk caching.

    First call: reads TSV → preprocesses → saves .parquet cache.
    Subsequent calls: loads .parquet directly (~10-20x faster).
    """
    tag = os.path.basename(path)
    t0 = time.time()

    # --- Try cache first ---
    if use_cache:
        cached = cache_load(path)
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
        cache_save(path, df)

    return df


def load_sources_parallel(*paths, use_cache: bool = True) -> list:
    """Load multiple source files in parallel using threads (I/O-bound).
    Each file is individually cached as parquet."""
    print(f"[IO] Loading {len(paths)} files in parallel threads...")
    t0 = time.time()
    results = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        futures = {pool.submit(load_source, p, use_cache): i
                   for i, p in enumerate(paths)}
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
    print(f"[IO] All loaded in {time.time()-t0:.1f}s")
    return results


# ===== OUTPUT ==================================================================

def write_output(matches, candidates, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mr = os.path.join(output_dir, "matching_results.tsv")
    cp = os.path.join(output_dir, "candidate_pairs.tsv")
    # newline="\n": plain LF line endings even on Windows (no stray \r in the last column)
    with open(mr, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in tqdm(sorted(matches), desc="Writing matching_results.tsv", unit="row"):
            m = ",".join(sorted(set(matches[sid]))) if matches[sid] else ""
            f.write(f"{sid}\t{m}\n")
    with open(cp, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in tqdm(sorted(candidates), desc="Writing candidate_pairs.tsv", unit="row"):
            c = ",".join(sorted(set(candidates[sid]))) if candidates[sid] else ""
            f.write(f"{sid}\t{c}\n")
    print(f"[Out] {mr}\n[Out] {cp}")


# ===== GROUND TRUTH ============================================================

def parse_ground_truth(gt_df):
    """{source1_entity_id: set of matched S2/S3 ids} (empty set for singletons)."""
    gt_lookup = {}
    for _, row in gt_df.iterrows():
        sid = row["source1_entity_id"]
        m   = row.get("matched_entity_ids", "")
        gt_lookup[sid] = set(str(m).split(",")) if pd.notna(m) and m else set()
    return gt_lookup


# ===== TRAINING POOL ===========================================================

def load_pool_sampled(s2_path, s3_path, must_have_ids, countries,
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
        cached = cache_load(path)
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
        cache_save(path, full_df)
        del all_chunks, full_df; gc.collect()
        print(f"  [{tag}] scanned + cached")

    all_frames = frames_must
    for co in countries:
        all_frames.extend(frames_rand[co])
    pool = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset="entity_id")
    print(f"[Pool] {len(pool):,} records in {time.time()-t0:.1f}s")
    return pool
