"""Run stage2.block unchanged, but size TF-IDF query chunks against CuPy's pool cap
(CUPY_GPU_MEMORY_LIMIT) instead of the device's free memory, so blocking can share
the GPU with the embedding jobs.

    python -m stage3.block_run <stage2.block args...>
"""
from entity_resolution.blocking import tfidf as T
from entity_resolution.config import TFIDF_MAX_CHUNK

_orig = T.TfidfIndex._chunk_size


def _chunk_size(self, n_pool):
    xp, _, is_gpu = T._backend()
    if not is_gpu:
        return _orig(self, n_pool)
    pool = xp.get_default_memory_pool()
    limit = pool.get_limit() or xp.cuda.Device().mem_info[1]
    avail = min(limit - pool.used_bytes(), xp.cuda.Device().mem_info[0] + pool.free_bytes())
    per_query = 4 * (n_pool + self.n_features) * 1.3
    # scores + top-k temporaries roughly double the scores matrix
    return int(max(1, min(TFIDF_MAX_CHUNK, avail * 0.4 // per_query)))


T.TfidfIndex._chunk_size = _chunk_size

CKPT_EVERY = 480   # seconds between partial-result checkpoints


def query_scored_ckpt(index, texts, countries, top_k):
    """stage2.block.query_scored with resumable progress: finished query chunks are saved
    to er_work/qs_<index>_<country>_<n>.pkl every CKPT_EVERY seconds and skipped on restart."""
    import time
    import numpy as np
    from tqdm import tqdm
    from stage2.common import exists, load, log, save
    xp, xsp, is_gpu = T._backend()
    texts, countries = np.asarray(texts), np.asarray(countries)
    out = [None] * len(texts)
    for co, (pool_mat, pool_ids, n_real) in index.parts.items():
        q_idx = np.flatnonzero(countries == co)
        if len(q_idx) == 0:
            continue
        ck = f"qs_{index.name}_{co}_{len(texts)}.pkl"
        done = 0
        if exists(ck):
            done, part = load(ck)
            for j, r in zip(q_idx[:done], part):
                out[j] = r
            log(f"resume {ck}: {done:,}/{len(q_idx):,} queries already scored")
        k = min(top_k, n_real)
        q_mat = T._parallel_transform(index.vec, texts[q_idx[done:]]).astype(np.float32)
        chunk = index._chunk_size(pool_mat.shape[0])
        last = time.time()
        for start in tqdm(range(0, len(q_idx) - done, chunk), desc=f"{index.name} [{co}]", mininterval=30):
            qc = q_mat[start:start + chunk].tocoo()
            c = qc.shape[0]
            dense_q = xp.zeros((index.n_features, c), dtype=xp.float32)
            dense_q[xp.asarray(qc.col), xp.asarray(qc.row)] = xp.asarray(qc.data)
            scores = xp.around(pool_mat @ dense_q, 5)
            rows, vals = T._block_topk(xp, scores, k)
            rows, vals = T._to_host(xp, rows), T._to_host(xp, vals)
            del scores, dense_q
            order = np.lexsort((rows, -vals), axis=0)
            rows = np.take_along_axis(rows, order, axis=0)
            vals = np.take_along_axis(vals, order, axis=0)
            for j in range(c):
                m = (vals[:, j] > 0) & (rows[:, j] < n_real)
                out[q_idx[done + start + j]] = (pool_ids[rows[m, j]], vals[m, j].astype(np.float32))
            if time.time() - last > CKPT_EVERY:
                n = done + start + c
                save((n, [out[j] for j in q_idx[:n]]), ck)
                last = time.time()
    if is_gpu:
        xp.get_default_memory_pool().free_all_blocks()
    empty = (np.array([], dtype=object), np.array([], dtype=np.float32))
    return [o if o is not None else empty for o in out]


if __name__ == "__main__":
    from stage2 import block
    block.query_scored = query_scored_ckpt
    block.main()
