#!/usr/bin/env python3
"""
Blocking evaluation at realistic pool density (see entity_resolution/blocking_eval.py).

Usage:
    python eval_blocking.py --country us --n-queries 5000                      # full pool
    python eval_blocking.py --country india --n-queries 2000 --pool-fraction 0.05
"""
import argparse
import multiprocessing as mp


def main():
    from entity_resolution import runtime
    from entity_resolution.blocking_eval import run

    runtime.configure_stdio()
    runtime.report_optional_dependencies()
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", required=True, help="country label, e.g. us, india")
    parser.add_argument("--n-queries", type=int, default=5_000,
                        help="S1 training entities used as queries")
    parser.add_argument("--pool-fraction", type=float, default=1.0,
                        help="share of the country's S2+S3 records in the pool "
                             "(true matches always kept); 1.0 = realistic full pool")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", help="optional JSON report path")
    args = parser.parse_args()
    run(args.country, args.n_queries, args.pool_fraction, args.seed, args.out)


if __name__ == "__main__":
    mp.freeze_support()
    main()
