#!/usr/bin/env python3
"""
Business Entity Resolution — command-line entry point.

The implementation lives in the ``entity_resolution`` package next to this file.

Usage:
    python run_pipeline.py --mode train --sample-size 5000
    python run_pipeline.py --mode full
    python run_pipeline.py --mode predict
"""
import argparse
import multiprocessing as mp


def main():
    # Imported here, not at module level: on Windows each feature worker
    # re-imports this script, and should not pay for loading the pipeline.
    from entity_resolution import runtime
    from entity_resolution.cache import clear_cache
    from entity_resolution.config import CACHE_DIR, N_WORKERS
    from entity_resolution.pipeline import run_full, run_predict, run_train

    runtime.configure_stdio()
    runtime.report_optional_dependencies()
    runtime.check_cuda()

    print(f"[Config] Workers: {N_WORKERS}, CPUs: {mp.cpu_count()}, "
          f"Cache: {CACHE_DIR}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","full","predict"], default="train")
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete all cached .parquet files before running")
    args = parser.parse_args()
    if args.clear_cache:
        clear_cache()
    {"train": lambda: run_train(args.sample_size),
     "full":  lambda: run_full(args.sample_size),
     "predict": run_predict}[args.mode]()


if __name__ == "__main__":
    mp.freeze_support()       # needed on Windows
    main()
