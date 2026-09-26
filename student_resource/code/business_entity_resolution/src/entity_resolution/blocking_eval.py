"""Blocking evaluation at realistic pool density.

The training pipeline blocks against true matches + <=30K random records per
country, where key buckets are tiny and full of true matches, so blocking
recall looks better than it will be on the test pool (~3.8-4.7M records per
country). This evaluates S1 training queries against the full (or a
Bernoulli-sampled) S2+S3 pool of one country and reports, per strategy:
recall, candidates per entity, time, and pairs found by no other strategy.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from .blocking import build_blockers, generate_all_candidates
from .blocking.prefilter import quick_prefilter
from .config import PREFILTER_MIN_SCORE, TRAIN_GT, TRAIN_S1, TRAIN_S2, TRAIN_S3
from .data import parse_ground_truth
from .preprocessing import fast_preprocess

PREFILTER_CAPS = (10, 25, 50, 75, 100, 150, 200)


# ===== DATA ====================================================================

def _read_country(path, country, keep=None, preprocess=True, chunksize=500_000):
    """Stream a raw TSV, keep one country's rows (optionally filtered further
    by `keep(chunk) -> bool mask`), and preprocess only what is kept."""
    parts = []
    for chunk in tqdm(pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize),
                      desc=f"Reading {os.path.basename(path)}", unit="chunk"):
        chunk = chunk[chunk["country"].fillna("").str.lower().str.strip() == country]
        if keep is not None:
            chunk = chunk[keep(chunk)]
        if len(chunk):
            parts.append(fast_preprocess(chunk) if preprocess else chunk)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_eval_data(country, n_queries, pool_fraction, seed):
    """S1 query sample, its ground truth, and the S2+S3 pool for one country.

    The pool keeps every true match of the sampled queries plus a Bernoulli
    sample of the rest at `pool_fraction` (1.0 = the full, realistic pool).
    """
    rng = np.random.RandomState(seed)
    s1 = _read_country(TRAIN_S1, country, preprocess=False)
    s1 = s1.iloc[rng.choice(len(s1), size=min(n_queries, len(s1)), replace=False)]
    s1 = fast_preprocess(s1.reset_index(drop=True))

    gt = pd.read_csv(TRAIN_GT, sep="\t", dtype=str)
    gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))]
    truth = parse_ground_truth(gt)
    must = set().union(*truth.values()) if truth else set()

    def keep(chunk):
        if pool_fraction >= 1.0:
            return np.ones(len(chunk), dtype=bool)
        return chunk["entity_id"].isin(must).values | (rng.random_sample(len(chunk)) < pool_fraction)

    pool = pd.concat([_read_country(p, country, keep) for p in (TRAIN_S2, TRAIN_S3)],
                     ignore_index=True).drop_duplicates("entity_id")
    return s1, truth, pool


# ===== METRICS =================================================================

def _recall(cands, truth):
    """(pair recall, share of matched entities with every match found)."""
    found = total = complete = n_ent = 0
    for sid, t in truth.items():
        if not t:
            continue
        got = len(t & set(cands.get(sid, ())))
        found, total = found + got, total + len(t)
        complete += got == len(t)
        n_ent += 1
    return found / max(total, 1), complete / max(n_ent, 1)


def _sizes(cands, ids):
    n = np.array([len(cands.get(s, ())) for s in ids])
    return float(n.mean()), float(np.percentile(n, 95))


def evaluate_blocking(s1, truth, pool):
    ids = s1["entity_id"].values
    t0 = time.time()
    blockers = build_blockers(pool)
    build_s = time.time() - t0
    final, by_strategy, timings = generate_all_candidates(s1, pool, blockers, return_details=True)
    blockers.release()

    # Pairs each strategy finds that no other strategy finds
    true_pairs = {(s, c) for s, t in truth.items() for c in t}
    found_by = {name: {(s, c) for s, cs in cands.items() for c in cs if (s, c) in true_pairs}
                for name, cands in by_strategy.items()}
    unique = {}
    for name, pairs in found_by.items():
        others = set().union(*(p for n, p in found_by.items() if n != name))
        unique[name] = len(pairs - others) / max(len(true_pairs), 1)

    rows = []
    for name, cands in by_strategy.items():
        pr, er = _recall(cands, truth)
        mean, p95 = _sizes(cands, ids)
        rows.append(dict(strategy=name, pair_recall=pr, entity_recall=er, unique_recall=unique[name],
                         cands_mean=mean, cands_p95=p95,
                         ms_per_query=1000 * timings[name] / max(len(ids), 1)))

    union = {}
    for s in ids:
        seen = {}
        for cands in by_strategy.values():
            seen.update(dict.fromkeys(cands.get(s, ())))
        union[s] = list(seen)
    for name, cands in (("UNION", union), ("FINAL (pre-filtered)", final)):
        pr, er = _recall(cands, truth)
        mean, p95 = _sizes(cands, ids)
        rows.append(dict(strategy=name, pair_recall=pr, entity_recall=er, unique_recall=None,
                         cands_mean=mean, cands_p95=p95, ms_per_query=None))

    # Recall if the pre-filter cap were K
    name_lk = dict(zip(pool["entity_id"], pool["name_clean"]))
    addr_lk = dict(zip(pool["entity_id"], pool["addr_clean"]))
    s1_addr = dict(zip(s1["entity_id"], s1["addr_clean"]))
    cap_curve = []
    for cap in PREFILTER_CAPS:
        capped = quick_prefilter(union, s1["name_clean"].values, ids, name_lk, addr_lk, s1_addr,
                                 max_candidates=cap, min_score=PREFILTER_MIN_SCORE)
        pr, er = _recall(capped, truth)
        cap_curve.append(dict(cap=cap, pair_recall=pr, entity_recall=er,
                              cands_mean=_sizes(capped, ids)[0]))
    return dict(strategies=rows, prefilter_caps=cap_curve, build_seconds=build_s)


def print_report(report, meta):
    print("\n" + "=" * 100)
    print(f" BLOCKING EVALUATION  country={meta['country']}  queries={meta['n_queries']:,} "
          f"(with matches: {meta['n_with_matches']:,})  pool={meta['pool_size']:,} "
          f"(fraction {meta['pool_fraction']})")
    print("=" * 100)
    print(f"{'strategy':22s} {'pair rec':>9s} {'entity rec':>11s} {'unique':>8s} "
          f"{'cands/ent':>10s} {'p95':>6s} {'ms/query':>9s}")
    for r in report["strategies"]:
        uniq = f"{r['unique_recall']:8.2%}" if r["unique_recall"] is not None else f"{'':8s}"
        ms = f"{r['ms_per_query']:9.2f}" if r["ms_per_query"] is not None else f"{'':9s}"
        print(f"{r['strategy']:22s} {r['pair_recall']:9.2%} {r['entity_recall']:11.2%} {uniq} "
              f"{r['cands_mean']:10.1f} {r['cands_p95']:6.0f} {ms}")
    print(f"\n  Build time for all blockers: {report['build_seconds']:.1f}s")
    print("\n  Pre-filter cap -> recall (union ranked by the cheap pre-filter score)")
    for c in report["prefilter_caps"]:
        print(f"    cap {c['cap']:>4d}: pair recall {c['pair_recall']:7.2%}  entity recall "
              f"{c['entity_recall']:7.2%}  cands/entity {c['cands_mean']:6.1f}")


def run(country, n_queries, pool_fraction, seed, out=None):
    country = country.lower().strip()
    s1, truth, pool = load_eval_data(country, n_queries, pool_fraction, seed)
    meta = dict(country=country, n_queries=len(s1), pool_size=len(pool),
                pool_fraction=pool_fraction, n_with_matches=sum(1 for t in truth.values() if t))
    report = evaluate_blocking(s1, truth, pool)
    print_report(report, meta)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, **report}, f, indent=2)
        print(f"\n[Out] {out}")
    return report
