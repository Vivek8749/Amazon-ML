"""Stage-1 / stage-2 features, models and the decision rule (shared by train and test)."""
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from entity_resolution.features import FEATURE_NAMES

STAGE2_FLOOR = 0.002      # rows with p1 below this skip stage 2 (p2 = p1)


def add_block_feats(df):
    """Blocking-context features: per-entity normalised cosines, source, group size."""
    df["is_s3"] = df["cand_id"].str.startswith("S3").astype(np.int8)
    g = df.groupby("s1_id", sort=False)
    for s in ("word", "char", "skel", "emb"):
        if f"{s}_cos" not in df:
            continue
        mx = g[f"{s}_cos"].transform("max")
        df[f"{s}_cos_max"] = mx.astype(np.float32)
        df[f"{s}_cos_gap"] = (mx - df[f"{s}_cos"]).astype(np.float32)
    if "emb_sim" in df:
        mx = g["emb_sim"].transform("max")
        df["emb_sim_max"] = mx.astype(np.float32)
        df["emb_sim_gap"] = (mx - df["emb_sim"]).astype(np.float32)
    df["n_cands"] = g["cand_id"].transform("size").astype(np.int16)
    return df


from stage2.extra import EXTRA_NAMES

F1 = list(FEATURE_NAMES) + EXTRA_NAMES + ["word_cos", "word_rank", "char_cos", "char_rank", "skel_cos", "skel_rank", "skel_cos_max", "skel_cos_gap", "is_s3",
                            "word_cos_max", "word_cos_gap", "char_cos_max", "char_cos_gap", "n_cands"]
EMB_FEATS = ["emb_cos", "emb_rank", "emb_cos_max", "emb_cos_gap", "emb_sim", "emb_sim_max", "emb_sim_gap"]
F1E = F1 + EMB_FEATS


def _num_set(s):
    import re
    return set(re.findall(r"\d+", s)) if s else set()


def _coh_chunk(args):
    a_n, a_a, b_n, b_a = args
    out = np.zeros((len(a_n), 6), dtype=np.float32)
    for i in range(len(a_n)):
        n1, n2, x1, x2 = a_n[i], b_n[i], a_a[i], b_a[i]
        if n2 is None:
            out[i] = -1
            continue
        out[i, 0] = fuzz.token_set_ratio(n1, n2) / 100.0
        out[i, 1] = JaroWinkler.similarity(n1, n2) if n1 and n2 else 0.0
        out[i, 2] = fuzz.token_set_ratio(x1, x2) / 100.0
        out[i, 3] = JaroWinkler.similarity(x1, x2) if x1 and x2 else 0.0
        u, v = _num_set(x1), _num_set(x2)
        out[i, 4] = len(u & v) / len(u | v) if (u or v) else 1.0
        out[i, 5] = fuzz.token_sort_ratio(f"{n1} {x1}", f"{n2} {x2}") / 100.0
    return out


def _pairwise(a_n, a_a, b_n, b_a, workers=12):
    n = len(a_n)
    if n == 0:
        return np.zeros((0, 6), np.float32)
    step = max(20_000, n // (workers * 4) + 1)
    chunks = [(a_n[i:i+step], a_a[i:i+step], b_n[i:i+step], b_a[i:i+step]) for i in range(0, n, step)]
    with ProcessPoolExecutor(workers) as ex:
        return np.vstack(list(ex.map(_coh_chunk, chunks)))


COH = ["name_tset", "name_jw", "addr_tset", "addr_jw", "num_jac", "comb_tsort"]


def add_stage2_feats(df, name_of, addr_of):
    """Context features from stage-1 score p1. df must be restricted to rows that
    enter stage 2 but carry per-entity stats computed over *all* rows (p1 stats
    are computed here over the rows passed in, which include every row >= floor)."""
    df = df.sort_values(["s1_id", "p1"], ascending=[True, False], kind="stable").reset_index(drop=True)
    g = df.groupby("s1_id", sort=False)
    df["p1_rank"] = g.cumcount().astype(np.int16)
    df["p1_max"] = g["p1"].transform("max")
    df["p1_gap"] = df["p1_max"] - df["p1"]
    df["p1_sum"] = g["p1"].transform("sum")
    sid_s = df["s1_id"]
    df["p1_n50"] = (df["p1"] > 0.5).groupby(sid_s, sort=False).transform("sum").astype(np.int16)
    df["p1_n10"] = (df["p1"] > 0.1).groupby(sid_s, sort=False).transform("sum").astype(np.int16)
    df["p1_2nd"] = df["p1"].where(df["p1_rank"] == 1).groupby(sid_s, sort=False).transform("max").fillna(0.0)
    df["p1_rel"] = df["p1"] / df["p1_max"].clip(lower=1e-6)
    # references: top-1, top-2, top-3 other candidates of the same entity (by p1)
    sid = df["s1_id"].values
    cand = df["cand_id"].values
    rank = df["p1_rank"].values
    start = np.r_[0, np.flatnonzero(sid[1:] != sid[:-1]) + 1]
    size = np.diff(np.r_[start, len(df)])
    gstart = np.repeat(start, size)
    gsize = np.repeat(size, size)
    names = np.array([name_of.get(c, "") for c in cand], dtype=object)
    addrs = np.array([addr_of.get(c, "") for c in cand], dtype=object)
    for k, tag in ((0, "r1"), (1, "r2"), (2, "r3")):
        # k-th best *other* candidate: skip self
        ref = np.where(rank <= k, k + 1, k)
        ok = ref < gsize
        idx = np.where(ok, gstart + ref, 0)
        rp = np.where(ok, df["p1"].values[idx], 0.0).astype(np.float32)
        bn = np.where(ok, names[idx], None)
        ba = np.where(ok, addrs[idx], None)
        sims = _pairwise(names, addrs, bn, ba)
        for j, c in enumerate(COH):
            df[f"coh_{tag}_{c}"] = sims[:, j]
        df[f"coh_{tag}_p"] = rp
    # p-weighted coherence against the top-3 others
    w = np.stack([df[f"coh_r{i}_p"].values for i in (1, 2, 3)], 1)
    for c in ("name_tset", "addr_tset", "comb_tsort"):
        s = np.stack([df[f"coh_r{i}_{c}"].values for i in (1, 2, 3)], 1)
        s = np.where(s < 0, 0, s)
        df[f"coh_w_{c}"] = ((w * s).sum(1) / np.clip(w.sum(1), 1e-6, None)).astype(np.float32)
    return df


P1_STATS = ["p1", "p1_rank", "p1_max", "p1_gap", "p1_sum", "p1_n50", "p1_n10", "p1_2nd", "p1_rel"]
COH_FEATS = ([f"coh_r{i}_{c}" for i in (1, 2, 3) for c in COH] + [f"coh_r{i}_p" for i in (1, 2, 3)]
             + [f"coh_w_{c}" for c in ("name_tset", "addr_tset", "comb_tsort")])
F2 = F1 + P1_STATS + COH_FEATS
F2E = F1E + P1_STATS + COH_FEATS


# ===== decision =================================================================

def decide(scores, threshold, ids, one_to_one=True, min_best=0.0):
    """scores: DataFrame s1_id, cand_id, p -> {s1_id: set(cand)}"""
    kept = scores[scores["p"] >= threshold]
    if one_to_one and len(kept):
        kept = kept.sort_values(["p", "s1_id"], ascending=[False, True], kind="stable") \
                   .drop_duplicates("cand_id", keep="first")
    if min_best > 0 and len(kept):
        best = scores.groupby("s1_id")["p"].max()
        kept = kept[kept["s1_id"].map(best) >= min_best]
    grouped = kept.groupby("s1_id")["cand_id"].agg(set).to_dict() if len(kept) else {}
    return {s: grouped.get(s, set()) for s in ids}


def f05(pred, truth):
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro(pred, truth):
    return float(np.mean([f05(pred.get(s, set()), t) for s, t in truth.items()]))


def breakdown(pred, truth):
    cats = {"singleton_given": 0.0, "left_empty": 0.0, "wrong_extra": 0.0, "missing_only": 0.0}
    for s, t in truth.items():
        p = pred.get(s, set()); l = 1 - f05(p, t)
        if l == 0: continue
        k = "singleton_given" if not t else "left_empty" if not p else "wrong_extra" if p - t else "missing_only"
        cats[k] += l
    n = len(truth)
    return {k: round(v / n, 4) for k, v in cats.items()}


# ===== expected-F0.5 decision ===================================================

def ef_decide(sc, col="p", floor=0.05, kmax=12, samples=2048, one_to_one=True, seed=0,
              temp=1.0, extra_true=0.0):
    """Per entity, keep the top-k candidates (k = 0..kmax) maximising expected F0.5
    under independent Bernoulli(p) labels. Returns the kept rows (s1_id, cand_id, p).

    F0.5 = 1.25 tp / (0.25 n_true + k); k = 0 scores 1 iff n_true == 0.
    `temp` sharpens/flattens probabilities in logit space (calibration knob);
    `extra_true` adds expected true matches that blocking missed.
    """
    import torch
    s = sc.loc[sc[col] >= floor, ["s1_id", "cand_id", col]].rename(columns={col: "p"})
    s = s.sort_values(["p", "s1_id"], ascending=[False, True], kind="stable")
    if one_to_one:
        s = s.drop_duplicates("cand_id", keep="first")
    s = s.sort_values(["s1_id", "p"], ascending=[True, False], kind="stable").reset_index(drop=True)
    if temp != 1.0:
        p = s["p"].clip(1e-6, 1 - 1e-6).values
        lg = np.log(p / (1 - p)) / temp
        s["p"] = 1 / (1 + np.exp(-lg))
    sid = s["s1_id"].values
    start = np.r_[0, np.flatnonzero(sid[1:] != sid[:-1]) + 1] if len(s) else np.array([], int)
    size = np.diff(np.r_[start, len(s)])
    rank = np.arange(len(s)) - np.repeat(start, size)
    s = s[rank < kmax].reset_index(drop=True)
    rank = rank[rank < kmax]
    ent = np.repeat(np.arange(len(start)), np.minimum(size, kmax))
    E = len(start)
    Pm = np.zeros((E, kmax), np.float32)
    Pm[ent, rank] = s["p"].values
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev).manual_seed(seed)
    best_k = np.zeros(E, np.int64)
    ks = torch.arange(kmax + 1, device=dev, dtype=torch.float32)
    B = max(1, 2_000_000_000 // (samples * kmax * 4 * 4))
    for a in range(0, E, B):
        pm = torch.from_numpy(Pm[a:a + B]).to(dev)
        lab = (torch.rand((pm.shape[0], samples, kmax), device=dev, generator=g) < pm[:, None, :]).float()
        ntrue = lab.sum(-1, keepdim=True) + extra_true                       # (b, S, 1)
        tp = torch.cat([torch.zeros_like(lab[..., :1]), lab.cumsum(-1)], -1)  # (b, S, K+1)
        f = 1.25 * tp / (0.25 * ntrue + ks)
        f[..., 0] = (ntrue[..., 0] == 0).float()
        best_k[a:a + B] = f.mean(1).argmax(-1).cpu().numpy()
        del pm, lab, ntrue, tp, f
    return s[rank < best_k[ent]][["s1_id", "cand_id", "p"]]
