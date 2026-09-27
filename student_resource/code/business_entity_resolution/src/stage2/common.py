"""Shared paths/helpers for the stage-2 experiments."""
import os, sys, glob, pickle, time
import numpy as np, pandas as pd
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
WORK = os.environ.get("ER_WORK", "/teamspace/studios/this_studio/er_work")
os.makedirs(WORK, exist_ok=True)
MAIN_SR = "/teamspace/studios/this_studio/Amazon-ML/student_resource"
CACHE = os.path.join(MAIN_SR, ".cache")
PRED_CKPT = os.path.join(MAIN_SR, ".checkpoints", "predict_d33eb5ff57aa")

def P(name):
    return os.path.join(WORK, name)

def save(obj, name):
    tmp = P(name) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=5)
    os.replace(tmp, P(name))

def load(name):
    with open(P(name), "rb") as f:
        return pickle.load(f)

def exists(name):
    return os.path.exists(P(name))

def cached_source(prefix):
    fs = sorted(glob.glob(os.path.join(CACHE, prefix + "_*.parquet")), key=os.path.getmtime)
    return pd.read_parquet(fs[-1])

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
