"""Multi-strategy candidate generation: build every blocker once per pool, run
each one, union their candidates (in a fixed order), then pre-filter."""
import gc
import time
from dataclasses import dataclass, field

import pandas as pd

from ..config import (
    HNSW_TOP_K, NAME_TFIDF_TOP_K, PREFILTER_MAX_CANDIDATES, PREFILTER_MIN_SCORE,
    TFIDF_TOP_K, USE_CHAR_TFIDF, USE_HNSW, USE_KEY_INDEXES, USE_LSH, USE_WORD_TFIDF,
    WORD_TFIDF_TOP_K,
)
from .dense import HNSW_AVAILABLE, build_hnsw_index, hnsw_block_batch, load_sbert_model
from .minhash import LSH_AVAILABLE, build_minhash_lsh, minhash_block_batch
from .prefilter import quick_prefilter
from .tfidf import build_name_tfidf_blocker, build_tfidf_blocker, build_word_tfidf_blocker

# Per-key cap on bucket members returned (buckets keep pool order).
KEY_CAP = 100


# ===== BLOCKER BUNDLE ==========================================================

@dataclass
class Blockers:
    """Every blocker built over one pool. Unused ones stay None / empty."""
    tfidf: dict = field(default_factory=dict)   # strategy -> (TfidfIndex, top_k)
    keys: dict | None = None                    # strategy -> {(key, country): [ids]}
    hnsw: tuple | None = None                   # (index, pool_ids, pool_countries, sbert_model)
    lsh: tuple | None = None                    # (lsh_dict, key_to_id)

    def release(self):
        for index, _ in self.tfidf.values():
            index.release()
        self.tfidf, self.keys, self.hnsw, self.lsh = {}, None, None, None
        gc.collect()


def build_key_indexes(pool_df):
    """Exact-key inverted indexes: {(key, country): [entity_ids in pool order]}."""
    print("[Block] Building inverted indexes...")
    t0 = time.time()
    names = pool_df["name_clean"]
    frames = {
        "key_prefix5": names.str[:5],
        "key_prefix4": names.str[:4],
        "key_sorted3tok": pool_df["name_sorted_3tok"],
        "key_addr_nums": pool_df["addr_nums_str"],
    }
    min_len = {"key_prefix5": 3, "key_prefix4": 3, "key_sorted3tok": 1, "key_addr_nums": 1}
    keys = {}
    for name, key in frames.items():
        df = pd.DataFrame({"entity_id": pool_df["entity_id"].values, "k": key.values,
                           "country_norm": pool_df["country_norm"].values})
        df = df[df["k"].str.len() >= min_len[name]]
        keys[name] = {k: v[:KEY_CAP] for k, v in
                      df.groupby(["k", "country_norm"])["entity_id"].apply(list).items()}
    print(f"[Block] Indexes built in {time.time()-t0:.1f}s ("
          + ", ".join(f"{k}={len(v):,}" for k, v in keys.items()) + ")")
    return keys


def build_blockers(pool_df, sbert_model=None) -> Blockers:
    """Build every enabled blocker over `pool_df` (see USE_* in config)."""
    b = Blockers()
    if USE_CHAR_TFIDF:
        b.tfidf["char_tfidf"] = (build_tfidf_blocker(pool_df), TFIDF_TOP_K)
    if USE_WORD_TFIDF:
        b.tfidf["word_tfidf"] = (build_word_tfidf_blocker(pool_df), WORD_TFIDF_TOP_K)
    if NAME_TFIDF_TOP_K > 0:
        b.tfidf["name_tfidf"] = (build_name_tfidf_blocker(pool_df), NAME_TFIDF_TOP_K)
    if USE_KEY_INDEXES:
        b.keys = build_key_indexes(pool_df)
    if USE_HNSW and HNSW_AVAILABLE:
        index, pids, pcos, model = build_hnsw_index(pool_df, sbert_model or load_sbert_model())
        if index is not None:
            b.hnsw = (index, pids, pcos, model)
    if USE_LSH and LSH_AVAILABLE:
        lsh_dict, key_to_id = build_minhash_lsh(pool_df)
        if lsh_dict is not None:
            b.lsh = (lsh_dict, key_to_id)
    return b


# ===== CANDIDATES ==============================================================

def _key_lookup(s1_df, keys):
    """Per-key-strategy candidates for each S1 record."""
    names = s1_df["name_clean"].values
    query_keys = {
        "key_prefix5": [n[:5] if len(n) >= 3 else "" for n in names],
        "key_prefix4": [n[:4] if len(n) >= 3 else "" for n in names],
        "key_sorted3tok": s1_df["name_sorted_3tok"].values,
        "key_addr_nums": s1_df["addr_nums_str"].values,
    }
    ids, countries = s1_df["entity_id"].values, s1_df["country_norm"].values
    out = {}
    for name, qk in query_keys.items():
        index = keys[name]
        out[name] = {sid: index.get((k, co), []) if k else []
                     for sid, k, co in zip(ids, qk, countries)}
    return out


def candidates_by_strategy(s1_df, blockers: Blockers):
    """Run each blocker separately.

    Returns ({strategy: {s1_id: [candidate ids]}}, {strategy: seconds}),
    strategies in the order their candidates enter the union.
    """
    ids = s1_df["entity_id"].values
    countries = s1_df["country_norm"].values
    texts = s1_df["combined"].values
    by_strategy, timings = {}, {}

    def run(name, fn):
        t0 = time.time()
        by_strategy[name] = fn()
        timings[name] = time.time() - t0
        n = sum(len(v) for v in by_strategy[name].values())
        print(f"[Block] {name}: {n:,} candidates ({n / max(len(ids), 1):.1f}/entity) "
              f"in {timings[name]:.1f}s")

    if "char_tfidf" in blockers.tfidf:
        index, k = blockers.tfidf["char_tfidf"]
        run("char_tfidf", lambda: dict(zip(ids, index.query(s1_df[index.text_col].values, countries, k))))
    if blockers.keys is not None:
        t0 = time.time()
        for name, cands in _key_lookup(s1_df, blockers.keys).items():
            by_strategy[name] = cands
            timings[name] = (time.time() - t0) / 4
            n = sum(len(v) for v in cands.values())
            print(f"[Block] {name}: {n:,} candidates ({n / max(len(ids), 1):.1f}/entity)")
    for name in ("word_tfidf", "name_tfidf"):
        if name in blockers.tfidf:
            index, k = blockers.tfidf[name]
            run(name, lambda: dict(zip(ids, index.query(s1_df[index.text_col].values, countries, k))))
    if blockers.hnsw is not None:
        index, pids, pcos, model = blockers.hnsw
        run("hnsw", lambda: dict(zip(ids, hnsw_block_batch(
            texts, countries, index, pids, pcos, model, top_k=HNSW_TOP_K))))
    if blockers.lsh is not None:
        lsh_dict, key_to_id = blockers.lsh
        run("lsh", lambda: dict(zip(ids, minhash_block_batch(texts, countries, lsh_dict, key_to_id))))
    return by_strategy, timings


def union_candidates(s1_ids, by_strategy):
    """Union of every strategy's candidates, sorted by ID so the order (and the
    training-row order derived from it) is identical across runs."""
    out = {}
    for sid in s1_ids:
        seen = set()
        for cands in by_strategy.values():
            seen.update(cands.get(sid, ()))
        out[sid] = sorted(seen)
    return out


def generate_all_candidates(s1_df, pool_df, blockers: Blockers, return_details=False):
    """All blockers -> union -> lightweight pre-filter.

    With return_details=True also returns the per-strategy candidates and
    timings (used by eval_blocking.py).
    """
    t0 = time.time()
    by_strategy, timings = candidates_by_strategy(s1_df, blockers)
    candidates = union_candidates(s1_df["entity_id"].values, by_strategy)
    total = sum(len(v) for v in candidates.values())
    print(f"[Block] TOTAL after all strategies: {total:,} candidates "
          f"({total / max(len(candidates), 1):.1f}/entity) in {time.time()-t0:.1f}s")

    # ---- Lightweight pre-filter to reduce candidates before full feature computation ----
    print("[Block] Running quick pre-filter...")
    tp = time.time()
    candidates = quick_prefilter(
        candidates, s1_df["name_clean"].values, s1_df["entity_id"].values,
        dict(zip(pool_df["entity_id"], pool_df["name_clean"])),
        dict(zip(pool_df["entity_id"], pool_df["addr_clean"])),
        dict(zip(s1_df["entity_id"], s1_df["addr_clean"])),
        max_candidates=PREFILTER_MAX_CANDIDATES,
        min_score=PREFILTER_MIN_SCORE,
    )
    total_after = sum(len(v) for v in candidates.values())
    print(f"[Block] After pre-filter: {total_after:,} candidates "
          f"({total_after / max(len(candidates), 1):.1f}/entity) "
          f"[{100*(1 - total_after/max(total,1)):.1f}% reduction] in {time.time()-tp:.1f}s")

    gc.collect()
    if return_details:
        return candidates, by_strategy, timings
    return candidates
