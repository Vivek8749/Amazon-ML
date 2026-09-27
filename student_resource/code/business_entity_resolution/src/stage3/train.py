"""Cross-validated stage-1 + stage-2 training on the stage-2 training entities.

    python -m stage3.train [--folds 5] [--final]

Reports leaderboard-style macro F0.5 (all sampled entities, incl. singletons and
blocking misses) for: old model on word-only / union candidates, new stage 1,
new stage 2. --final also fits the full-data models used on the test set.
"""
import argparse
import gc
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

from stage2.common import P, cached_source, exists, load, log, save
from stage3.model import (F1, F2, F1E, F2E, STAGE2_FLOOR, add_block_feats, add_stage2_feats,
                          breakdown, decide, macro)

GT = "/teamspace/studios/this_studio/Amazon-ML/student_resource/dataset/train/train_ground_truth.tsv"

PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda",
              max_depth=9, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=2, reg_lambda=2.0, max_bin=256)


def fit(X, y, Xv=None, yv=None, rounds=3000, params=PARAMS):
    d = xgb.DMatrix(X, label=y)
    ev = [(xgb.DMatrix(Xv, label=yv), "val")] if Xv is not None else []
    b = xgb.train(params, d, rounds, evals=ev, early_stopping_rounds=100 if ev else None,
                  verbose_eval=500)
    return b


def fold_fit(ck, X, y, tr, va):
    """One CV fold, checkpointed: (booster, OOF predictions for va) survive a reset."""
    if exists(ck):
        b, p = load(ck)
        if len(p) == int(va.sum()):
            return b, p
    b = fit(X[tr], y[tr], X[va], y[va])
    p = pred(b, X[va])
    save((b, p), ck)
    return b, p


def pred(b, X):
    it = (0, b.best_iteration + 1) if getattr(b, "best_iteration", None) is not None else (0, 0)
    return b.predict(xgb.DMatrix(X), iteration_range=it).astype(np.float32)


def truth_for(ids):
    gt = pd.read_csv(GT, sep="\t", dtype=str)
    gt = gt[gt["source1_entity_id"].isin(set(ids))]
    return {s: set(m.split(",")) if isinstance(m, str) and m else set()
            for s, m in gt[["source1_entity_id", "matched_entity_ids"]].values}


def fast_macro(sc, t, ntruth, one_to_one=True, detail=False):
    """sc: s1_id, cand_id, p, y (sorted by p desc). ntruth: Series s1_id -> #true (all entities)."""
    k = sc[sc["p"] >= t]
    if one_to_one:
        k = k.drop_duplicates("cand_id", keep="first")
    g = k.groupby("s1_id")["y"].agg(["size", "sum"])
    npred = g["size"].reindex(ntruth.index, fill_value=0).values
    tp = g["sum"].reindex(ntruth.index, fill_value=0).values
    nt = ntruth.values
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(npred > 0, tp / np.maximum(npred, 1), 0)
        rec = np.where(nt > 0, tp / np.maximum(nt, 1), 0)
        f = np.where(tp > 0, 1.25 * prec * rec / (0.25 * prec + rec), 0.0)
    f = np.where(nt == 0, (npred == 0).astype(float), f)
    if not detail:
        return f.mean()
    loss = 1 - f
    cat = np.where(nt == 0, "singleton_given", np.where(npred == 0, "left_empty",
                   np.where(npred > tp, "wrong_extra", "missing_only")))
    bd = {c: round(float(loss[cat == c].sum() / len(f)), 4) for c in
          ("singleton_given", "left_empty", "wrong_extra", "missing_only")}
    return f.mean(), bd


def evaluate(df, col, truth, label, grid=None):
    grid = grid if grid is not None else np.r_[np.arange(0.3, 0.9, 0.05), np.arange(0.9, 0.995, 0.005)]
    sc = df.loc[df[col] >= min(grid), ["s1_id", "cand_id", col, "y"]].rename(columns={col: "p"})
    sc = sc.sort_values(["p", "s1_id"], ascending=[False, True], kind="stable")
    ntruth = pd.Series({s: len(t) for s, t in truth.items()})
    res = [(t, fast_macro(sc, t, ntruth)) for t in grid]
    t, f = max(res, key=lambda r: r[1])
    _, bd = fast_macro(sc, t, ntruth, detail=True)
    log(f"  {label:34s} macro F0.5 = {f:.4f} @ {t:.3f}  {bd}")
    return t, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--countries", default="india,us")
    ap.add_argument("--emb", action="store_true", help="use feat_train_<co>_e.parquet + emb features")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    global F1, F2
    if a.emb:
        F1, F2 = F1E, F2E
    sfx = "_e" if a.emb else ""

    df = pd.concat([pd.read_parquet(P(f"feat_train_{c}{sfx}.parquet")) for c in a.countries.split(",")],
                   ignore_index=True)
    ids = [s for s in load("train_ids.pkl")]
    s1 = cached_source("train_source1")[["entity_id", "country_norm"]]
    keep = set(s1[s1["country_norm"].isin(a.countries.split(","))]["entity_id"]) & set(ids)
    truth = truth_for(keep)
    log(f"{len(df):,} pairs, {len(truth):,} entities, pos {df.y.sum():,}")
    tot = sum(len(t) for t in truth.values())
    log(f"recall word-only {df[(df.word_rank < 30) & (df.y == 1)].shape[0] / tot:.4f}  "
        f"union {df.y.sum() / tot:.4f}")

    df = add_block_feats(df)
    # entity folds
    uid = pd.Series(sorted(truth)); fold_of = dict(zip(uid, np.random.RandomState(0).permutation(len(uid)) % a.folds))
    df["fold"] = df["s1_id"].map(fold_of).astype(np.int8)

    evaluate(df[df.word_rank < 30], "p_old", truth, "old model, word-only cands")
    evaluate(df, "p_old", truth, "old model, word+char cands")

    # ---- stage 1 OOF ----
    df["p1"] = 0.0
    B1, B2 = [], []
    X1 = df[F1].values.astype(np.float32); y = df["y"].values
    for k in range(a.folds):
        tr, va = df["fold"].values != k, df["fold"].values == k
        b, p = fold_fit(f"ckpt{a.tag}_s1_f{k}.pkl", X1, y, tr, va)
        df.loc[va, "p1"] = p; B1.append(b)
        log(f"stage1 fold {k}: best_iter {b.best_iteration}")
    del X1; gc.collect()
    evaluate(df, "p1", truth, "stage 1 (OOF)")
    evaluate(df[df.word_rank < 30], "p1", truth, "stage 1 (OOF) word-only rows")

    # ---- stage 2 ----
    pool = pd.concat([cached_source("train_source2"), cached_source("train_source3")], ignore_index=True)
    s2rows = df[df["p1"] >= STAGE2_FLOOR].copy()
    pool = pool[pool["entity_id"].isin(set(s2rows["cand_id"]))]
    name_of = dict(zip(pool["entity_id"], pool["name_clean"])); addr_of = dict(zip(pool["entity_id"], pool["addr_clean"]))
    del pool; gc.collect()
    log(f"stage2 rows {len(s2rows):,}")
    rows_ck = f"stage2_rows_train{a.tag}_f{a.folds}.parquet"
    if exists(rows_ck):   # resume: stage-2 context features already computed from the same OOF p1
        s2rows = pd.read_parquet(P(rows_ck))
    else:
        s2rows = add_stage2_feats(s2rows, name_of, addr_of)
        s2rows.to_parquet(P(rows_ck + ".tmp")); os.replace(P(rows_ck + ".tmp"), P(rows_ck))
    s2rows["p2"] = 0.0
    X2 = s2rows[F2].values.astype(np.float32); y2 = s2rows["y"].values
    for k in range(a.folds):
        tr, va = s2rows["fold"].values != k, s2rows["fold"].values == k
        b, p = fold_fit(f"ckpt{a.tag}_s2_f{k}.pkl", X2, y2, tr, va)
        s2rows.loc[va, "p2"] = p; B2.append(b)
        log(f"stage2 fold {k}: best_iter {b.best_iteration}")
    save({"B1": B1, "B2": B2, "F1": F1, "F2": F2}, f"stage_models{a.tag}.pkl")
    log(f"saved stage_models{a.tag}.pkl")
    s2rows[["s1_id", "cand_id", "y", "p1", "p2", "p_old", "fold", "country"]].to_parquet(P(f"oof_train{a.tag}.parquet"))
    evaluate(s2rows, "p2", truth, "stage 2 (OOF)")
    for co in sorted(s2rows["country"].unique()):
        ids_co = set(s1.loc[s1.country_norm == co, "entity_id"])
        tc = {s: t for s, t in truth.items() if s in ids_co}
        evaluate(s2rows[s2rows.country == co], "p2", tc, f"stage 2 (OOF) {co}")
    report_ef(s2rows, truth)


def kept_macro(kept, df, truth):
    y = df.set_index(["s1_id", "cand_id"])["y"]
    k = kept.join(y, on=["s1_id", "cand_id"])
    g = k.groupby("s1_id")["y"].agg(["size", "sum"])
    ntruth = pd.Series({s: len(t) for s, t in truth.items()})
    npred = g["size"].reindex(ntruth.index, fill_value=0).values
    tp = g["sum"].reindex(ntruth.index, fill_value=0).values
    nt = ntruth.values
    f = np.where(tp > 0, 1.25 * tp / (0.25 * nt + npred), 0.0)
    f = np.where(nt == 0, (npred == 0).astype(float), f)
    return f.mean()


def report_ef(s2rows, truth):
    from stage3.model import ef_decide
    for temp in (1.0, 0.8, 1.25):
        for extra in (0.0, 0.1):
            kept = ef_decide(s2rows, "p2", temp=temp, extra_true=extra)
            log(f"  expected-F decision temp={temp} extra={extra}: macro F0.5 = "
                f"{kept_macro(kept, s2rows, truth):.4f}")


if __name__ == "__main__":
    main()
