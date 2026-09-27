"""Fine-tuned sentence embeddings: dense blocking + an embedding-similarity pair feature.

    python -m stage3.emb finetune            # contrastive fine-tune on train S1 -> match pairs
    python -m stage3.emb encode train|test   # S1 + pool embeddings -> er_work/emb_<split>_*.npy
    python -m stage3.emb block train|test    # top-K cosine per country -> cands_<split>_<co>_emb.pkl

The fine-tune uses training entities outside the stage-2 sample (block.py's
perm[TUNE_OFFSET:TUNE_OFFSET+300k]) so the emb features stay honest on OOF.
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

from stage2.block import TOP_K, TUNE_OFFSET
from stage2.common import P, cached_source, exists, load, log, save

BASE = os.environ.get("ER_EMB_BASE", "intfloat/multilingual-e5-base")
FT_DIR = P("emb_model")
MAX_LEN = 64
GT = "/teamspace/studios/this_studio/Amazon-ML/student_resource/dataset/train/train_ground_truth.tsv"


def text_of(df):
    return (df["name_clean"].fillna("") + " | " + df["addr_clean"].fillna("")).values


def _load(path):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).cuda()
    return tok, model


def _embed(model, batch):
    out = model(**batch).last_hidden_state
    m = batch["attention_mask"].unsqueeze(-1).to(out.dtype)
    return Fn.normalize((out * m).sum(1) / m.sum(1).clamp(min=1e-6), dim=-1)


def finetune(n_pairs=1_500_000, bs=1024, lr=1e-4, scale=30.0):
    if os.path.exists(os.path.join(FT_DIR, "config.json")):
        return
    s1 = cached_source("train_source1")
    perm = np.random.RandomState(42).permutation(len(s1))
    held = set(s1["entity_id"].values[perm[TUNE_OFFSET:TUNE_OFFSET + 300_000]])
    gt = pd.read_csv(GT, sep="\t", dtype=str).dropna()
    gt = gt[~gt["source1_entity_id"].isin(held)]
    rng = np.random.RandomState(0)
    gt = gt.sample(min(n_pairs, len(gt)), random_state=0)
    pos = np.array([rng.choice(m.split(",")) for m in gt["matched_entity_ids"].values], dtype=object)
    pool = pd.concat([cached_source("train_source2"), cached_source("train_source3")], ignore_index=True)
    pool = pool[pool["entity_id"].isin(set(pos))]
    ptext = dict(zip(pool["entity_id"], text_of(pool)))
    s1 = s1.set_index("entity_id").loc[gt["source1_entity_id"].values]
    a_txt, co = text_of(s1), s1["country_norm"].values
    p_txt = np.array([ptext[c] for c in pos], dtype=object)
    del pool, ptext; gc.collect()
    # country-homogeneous batches: in-batch negatives are then plausible confusers
    batches = []
    for c in np.unique(co):
        idx = rng.permutation(np.flatnonzero(co == c))
        batches += [idx[i:i + bs] for i in range(0, len(idx) - bs + 1, bs)]
    order = rng.permutation(len(batches))
    log(f"finetune: {len(a_txt):,} pairs, {len(batches)} batches of {bs}")
    tok, model = _load(BASE)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=len(batches), pct_start=0.05)
    labels = torch.arange(bs, device="cuda")
    t0 = time.time()
    for step, b in enumerate(order):
        ix = batches[b]
        ta = tok(list(a_txt[ix]), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
        tp = tok(list(p_txt[ix]), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ea, ep = _embed(model, ta), _embed(model, tp)
            sim = (ea @ ep.T).float() * scale
            loss = (Fn.cross_entropy(sim, labels) + Fn.cross_entropy(sim.T, labels)) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0:
            acc = (sim.argmax(1) == labels).float().mean().item()
            log(f"step {step}/{len(order)} loss {loss.item():.4f} in-batch acc {acc:.3f} ({time.time()-t0:.0f}s)")
    model.save_pretrained(FT_DIR + ".tmp"); tok.save_pretrained(FT_DIR + ".tmp")
    os.replace(FT_DIR + ".tmp", FT_DIR)
    log(f"saved {FT_DIR}")


PART = 1_048_576   # texts per resumable encoding part (multiple of the batch size)


@torch.no_grad()
def encode_texts(tok, model, texts, bs=4096, ckpt=None):
    """Length-sorted batched encoding. With ckpt, every PART texts are saved to
    <ckpt>.part<j>.npy so an interrupted run resumes from the last finished part."""
    lens = np.fromiter((len(t) for t in texts), dtype=np.int32, count=len(texts))
    order = np.argsort(lens, kind="stable")
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float16)
    t0 = time.time()
    for j, a in enumerate(range(0, len(order), PART)):
        part = order[a:a + PART]
        f = P(f"{ckpt}.part{j}.npy") if ckpt else None
        if f and os.path.exists(f):
            out[part] = np.load(f)
            continue
        for i in range(0, len(part), bs):
            ix = part[i:i + bs]
            t = tok([texts[k] for k in ix], padding=True, truncation=True, max_length=MAX_LEN,
                    return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[ix] = _embed(model, t).float().cpu().numpy().astype(np.float16)
        if f:
            np.save(f + ".tmp.npy", out[part]); os.replace(f + ".tmp.npy", f)
        log(f"  encoded {a + len(part):,}/{len(texts):,} ({time.time()-t0:.0f}s)")
    return out


def encode(split):
    tok, model = _load(FT_DIR)
    model.eval()
    for part in ("s1", "pool"):
        name = f"emb_{split}_{part}"
        if exists(name + ".npy"):
            continue
        if part == "s1":
            df = cached_source(f"{split}_source1")
            if split == "train":
                df = df[df["entity_id"].isin(set(load("train_ids.pkl")))]
        else:
            df = pd.concat([cached_source(f"{split}_source2"), cached_source(f"{split}_source3")],
                           ignore_index=True)
        log(f"encode {name}: {len(df):,}")
        E = encode_texts(tok, model, text_of(df), ckpt=name)
        save({"ids": df["entity_id"].values, "country": df["country_norm"].values}, name + "_ids.pkl")
        np.save(P(name + ".tmp.npy"), E); os.replace(P(name + ".tmp.npy"), P(name + ".npy"))
        import glob
        for f in glob.glob(P(name + ".part*.npy")):
            os.remove(f)
        del df, E; gc.collect()


class EmbLookup:
    """id -> embedding row, for the pair-level emb_sim feature."""
    def __init__(self, split):
        self.E, self.row = {}, {}
        for part in ("s1", "pool"):
            meta = load(f"emb_{split}_{part}_ids.pkl")
            self.E[part] = np.load(P(f"emb_{split}_{part}.npy"), mmap_mode="r")
            self.row[part] = pd.Series(np.arange(len(meta["ids"])), index=meta["ids"])

    def sim(self, s1_ids, cand_ids, chunk=250_000):
        a = self.row["s1"].reindex(s1_ids).values
        b = self.row["pool"].reindex(cand_ids).values
        out = np.full(len(a), -1.0, np.float32)
        ok = ~(np.isnan(a) | np.isnan(b))
        ia, ib = a[ok].astype(np.int64), b[ok].astype(np.int64)
        res = np.empty(len(ia), np.float32)
        for i in range(0, len(ia), chunk):
            xa = np.asarray(self.E["s1"][ia[i:i + chunk]])
            yb = np.asarray(self.E["pool"][ib[i:i + chunk]])
            try:   # GPU is shared with blocking/encoding: small chunks, CPU fallback on OOM
                x = torch.from_numpy(xa).cuda().float(); y = torch.from_numpy(yb).cuda().float()
                res[i:i + chunk] = (x * y).sum(1).cpu().numpy()
                del x, y
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                res[i:i + chunk] = np.einsum("ij,ij->i", xa.astype(np.float32), yb.astype(np.float32))
        out[ok] = res
        return out


@torch.no_grad()
def block(split, k=TOP_K, qchunk=1024):
    ms, mp = load(f"emb_{split}_s1_ids.pkl"), load(f"emb_{split}_pool_ids.pkl")
    Es, Ep = np.load(P(f"emb_{split}_s1.npy"), mmap_mode="r"), np.load(P(f"emb_{split}_pool.npy"), mmap_mode="r")
    for co in np.unique(ms["country"]):
        out = f"cands_{split}_{co}_emb.pkl"
        if exists(out):
            continue
        qi, pi = np.flatnonzero(ms["country"] == co), np.flatnonzero(mp["country"] == co)
        if len(pi) == 0:
            continue
        pool = torch.from_numpy(np.asarray(Ep[pi])).cuda()
        pids = mp["ids"][pi]
        Q = np.asarray(Es[qi])
        res = {}
        t0 = time.time()
        for i in range(0, len(qi), qchunk):
            q = torch.from_numpy(Q[i:i + qchunk]).cuda()
            v, r = torch.topk(q @ pool.T, min(k, len(pi)), dim=1)
            v, r = v.float().cpu().numpy(), r.cpu().numpy()
            for j in range(len(r)):
                res[ms["ids"][qi[i + j]]] = (pids[r[j]], v[j])
        save(res, out)
        log(f"[{split}/{co}] emb block {len(qi):,} x {len(pi):,} in {time.time()-t0:.0f}s")
        del pool; torch.cuda.empty_cache()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "finetune":
        finetune()
    elif cmd == "encode":
        encode(sys.argv[2])
    else:
        block(sys.argv[2])
