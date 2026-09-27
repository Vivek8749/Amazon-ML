"""Stage checkpoints so a killed session resumes where it stopped.

A run (training, or test prediction for one model) gets its own folder under
CHECKPOINT_DIR, named by a hash of everything that changes its results: the
run's arguments plus every result-relevant setting in config.py. Change a
setting and the next run starts a fresh folder instead of reusing stale
stages. Each finished stage is pickled with write-then-rename, so a stage file
either exists complete or not at all.

    ckpt = Checkpoint("train", sample_size=50_000)
    X, y = ckpt.stage("features", lambda: build_training_data(...))
"""
import hashlib
import json
import os
import pickle
import re
import shutil
import time

from . import config

# Settings that don't change results (paths, hardware/throughput knobs).
_NOT_RESULT_RELEVANT = {
    "PKG_DIR", "SRC_DIR", "BASE_DIR", "TRAIN_S1", "TRAIN_S2", "TRAIN_S3", "TRAIN_GT",
    "TEST_S1", "TEST_S2", "TEST_S3", "OUTPUT_DIR", "MODEL_PATH", "CACHE_DIR",
    "EMBED_CACHE_DIR", "CHECKPOINT_DIR", "RESUME", "N_WORKERS", "FEAT_CHUNK",
    "PRED_BATCH_SIZE", "TFIDF_DEVICE", "TFIDF_GPU_MEM_FRACTION", "TFIDF_MAX_CHUNK",
    "TFIDF_CPU_CHUNK", "HNSW_BATCH_SIZE",
}


def config_snapshot():
    """Every result-relevant setting in config.py."""
    return {k: v for k, v in vars(config).items()
            if k.isupper() and k not in _NOT_RESULT_RELEVANT
            and isinstance(v, (bool, int, float, str, tuple, list))}


class Checkpoint:
    def __init__(self, kind, resume=None, **key):
        key = {"kind": kind, **key, "config": config_snapshot()}
        digest = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]
        self.dir = os.path.join(config.CHECKPOINT_DIR, f"{kind}_{digest}")
        resume = config.RESUME if resume is None else resume
        if not resume and os.path.isdir(self.dir):
            shutil.rmtree(self.dir)
            print(f"[Checkpoint] RESUME off — cleared {self.dir}")
        os.makedirs(self.dir, exist_ok=True)
        meta = os.path.join(self.dir, "run.json")
        if not os.path.exists(meta):
            with open(meta, "w", encoding="utf-8") as f:
                json.dump({**key, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f,
                          indent=1, default=str)
        done = self.completed()
        print(f"[Checkpoint] {self.dir}"
              + (f" — resuming, {len(done)} stage(s) already done" if done else " — new run"))

    def _path(self, stage):
        return os.path.join(self.dir, re.sub(r"[^A-Za-z0-9_.=-]+", "_", stage) + ".pkl")

    def has(self, stage):
        return os.path.exists(self._path(stage))

    def load(self, stage):
        with open(self._path(stage), "rb") as f:
            return pickle.load(f)

    def save(self, stage, obj):
        path = self._path(stage)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def stage(self, stage, fn):
        """Return the saved result of `stage`, or run `fn()`, save it and return it."""
        if self.has(stage):
            t0 = time.time()
            obj = self.load(stage)
            print(f"[Checkpoint] ✓ {stage} — loaded ({time.time() - t0:.1f}s)")
            return obj
        obj = fn()
        self.save(stage, obj)
        print(f"[Checkpoint] saved {stage}")
        return obj

    def completed(self):
        return sorted(f[:-4] for f in os.listdir(self.dir) if f.endswith(".pkl"))


def list_runs():
    """[(folder, kind, created, completed stages)] for every saved run."""
    if not os.path.isdir(config.CHECKPOINT_DIR):
        return []
    runs = []
    for name in sorted(os.listdir(config.CHECKPOINT_DIR)):
        d = os.path.join(config.CHECKPOINT_DIR, name)
        meta = os.path.join(d, "run.json")
        if not os.path.exists(meta):
            continue
        with open(meta, encoding="utf-8") as f:
            m = json.load(f)
        stages = sorted(x[:-4] for x in os.listdir(d) if x.endswith(".pkl"))
        runs.append((d, m.get("kind"), m.get("created"), stages))
    return runs


def clear_checkpoints():
    if os.path.isdir(config.CHECKPOINT_DIR):
        shutil.rmtree(config.CHECKPOINT_DIR)
        print(f"[Checkpoint] Cleared {config.CHECKPOINT_DIR}")
