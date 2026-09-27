#!/usr/bin/env python3
"""
Business Entity Resolution — command-line entry point.

The implementation lives in the ``entity_resolution`` package next to this file.

Usage:
    python run_pipeline.py --mode train --sample-size 5000
    python run_pipeline.py --mode full
    python run_pipeline.py --mode predict
    python run_pipeline.py --mode tune      # re-tune the saved model's threshold
"""
import argparse
import multiprocessing as mp


def main():
    # Imported here, not at module level: on Windows each feature worker
    # re-imports this script, and should not pay for loading the pipeline.
    from entity_resolution import runtime
    from entity_resolution.cache import clear_cache
    from entity_resolution.checkpoint import clear_checkpoints
    from entity_resolution.config import CACHE_DIR, CHECKPOINT_DIR, N_WORKERS
    from entity_resolution.pipeline import run_full, run_predict, run_train, run_tune

    runtime.configure_stdio()
    runtime.report_optional_dependencies()
    runtime.check_cuda()

    print(f"[Config] Workers: {N_WORKERS}, CPUs: {mp.cpu_count()}, "
          f"Cache: {CACHE_DIR}, Checkpoints: {CHECKPOINT_DIR}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","full","predict","tune"], default="train")
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--threshold", type=float, default=None,
                        help="--mode predict: override the saved model's decision threshold")
    parser.add_argument("--tune-entities", type=int, default=20_000,
                        help="--mode tune: unseen training entities used to re-tune the threshold")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete all cached .parquet files before running")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore saved stages for this run and recompute them")
    parser.add_argument("--clear-checkpoints", action="store_true",
                        help="Delete every saved checkpoint before running")
    args = parser.parse_args()
    if args.clear_cache:
        clear_cache()
    if args.clear_checkpoints:
        clear_checkpoints()
    resume = not args.no_resume
    {"train": lambda: run_train(args.sample_size, resume=resume),
     "full":  lambda: run_full(args.sample_size, resume=resume),
     "predict": lambda: run_predict(resume=resume, threshold=args.threshold),
     "tune":  lambda: run_tune(args.tune_entities, resume=resume)}[args.mode]()


if __name__ == "__main__":
    mp.freeze_support()       # needed on Windows
    main()
