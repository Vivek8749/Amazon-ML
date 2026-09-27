# Business Entity Resolution — Code README

## Overview

This pipeline resolves business entities across three independent data sources
(S1, S2, S3) by determining which records refer to the same real-world business.
It uses a **TF-IDF blocking + XGBoost classifier** approach optimised for the
precision-heavy F₀.₅ metric.

## Architecture

```
run_pipeline.py → entity_resolution/ (see Code Layout below)
│
├── Preprocessing
│   ├── Transliteration to ASCII (Devanagari/Kannada → Latin, é → e) via anyascii
│   ├── Vectorised text cleaning (lowercase, bracket/symbol handling)
│   ├── Legal suffix normalisation (Corporation→corp, Ltd.→ltd, S.A.R.L.→sarl)
│   └── Address abbreviation normalisation (Street→st, Boulevard→blvd, etc.)
│
├── Blocking / Candidate Generation (per country, exact top-K on GPU)
│   ├── Char 2–4-gram TF-IDF cosine over name + address (top 30)
│   ├── Word 1–2-gram TF-IDF cosine over name + address (top 30)
│   ├── Sentence-transformer + HNSW dense retrieval (if installed)
│   ├── Optional, off by default: name-only TF-IDF, key indexes, MinHash/LSH
│   └── Union → cheap pre-filter cap (100)
│
├── Feature Engineering (29 similarity features)
│   ├── Name: Levenshtein, Jaro-Winkler, token-sort/set/partial ratios,
│   │        Jaccard, overlap, Dice, containment, length ratio
│   ├── Address: same suite of string metrics
│   ├── Address numbers: Jaccard, overlap, match ratio
│   └── Cross: combined token ratios, combined Jaccard, token-count diff
│
├── Model Training
│   ├── XGBoost binary classifier (hist method, 500 trees, depth 8)
│   ├── Class-imbalanced via scale_pos_weight
│   └── Early stopping on validation loss
│
├── Threshold Optimisation
│   └── Maximises F₀.₅ on held-out validation precision-recall curve
│
└── Test Prediction
    └── Country-by-country processing for memory efficiency
```

### Code Layout

`run_pipeline.py` is a thin CLI; the implementation is the `entity_resolution`
package, one module per pipeline stage:

```
src/
├── run_pipeline.py            CLI entry point (--mode train|full|predict)
├── eval_blocking.py           CLI: blocking recall/cost at realistic pool density
├── models/xgb_model.pkl       saved model + threshold
└── entity_resolution/
    ├── config.py              paths and all hyperparameters
    ├── runtime.py             UTF-8 stdio, CUDA check, optional-dependency report
    ├── cache.py               parquet cache (preprocessed sources) + embedding cache
    ├── preprocessing.py       name/address normalisation (fast_preprocess)
    ├── data.py                source loading, training-pool sampling,
    │                          ground-truth parsing, output TSV writing
    ├── blocking/
    │   ├── tfidf.py           TfidfIndex: char/word/name TF-IDF, exact per-country
    │   │                      top-K (CuPy SpMM + block top-K, SciPy fallback)
    │   ├── dense.py           sentence-transformer embeddings + HNSW
    │   ├── minhash.py         MinHash/LSH per country
    │   ├── prefilter.py       cheap 3-signal score, caps candidates per entity
    │   └── candidates.py      build_blockers + generate_all_candidates (union, pre-filter)
    ├── blocking_eval.py       per-strategy recall / cost report (used by eval_blocking.py)
    ├── features.py            pairwise similarity features (parallel workers)
    ├── training.py            training pairs, XGBoost grid search,
    │                          F0.5 threshold, hard-negative mining
    ├── inference.py           predict_all, singleton post-processing
    ├── evaluation.py          macro F0.5 (leaderboard metric)
    └── pipeline.py            run_train / predict_test_by_country / run_full / run_predict
```

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
rapidfuzz>=3.0.0
tqdm>=4.60.0
scipy>=1.10.0
```

Install:

```bash
pip install -r requirements.txt
```

The pipeline requires an NVIDIA GPU with a working CUDA driver and fails at
startup when CUDA is unavailable. XGBoost and sentence-transformers use CUDA
explicitly, and CuPy provides GPU sparse-matrix operations. Pandas, text
normalisation, RapidFuzz, and HNSW index construction still use CPU-backed
libraries; moving those stages to GPU requires replacing them with
RAPIDS/cuDF/cuML-compatible implementations.

## How to Reproduce

All commands should be run from the `student_resource/` directory.

### 1. Train + Predict (end-to-end)

```bash
python code/business_entity_resolution/src/run_pipeline.py --mode full --sample-size 20000
```

This will:

1. Load training data (S1 + sampled S2/S3 pool)
2. Split into train/validation (90/10)
3. Build TF-IDF blocker and generate candidate pairs
4. Compute 29 similarity features for all candidate pairs
5. Train XGBoost classifier with early stopping
6. Optimise decision threshold for F₀.₅
7. Evaluate on validation set
8. Predict on test set (country-by-country for memory efficiency)
9. Write `output/matching_results.tsv` and `output/candidate_pairs.tsv`

### Resuming after a killed session

Every expensive stage is checkpointed under `student_resource/.checkpoints/`: the training
sample, candidates, features, each XGBoost grid config, hard negatives, round 2, the final
model, and — for the test set — each country's candidates and every scored shard of
100,000 entities. Rerun the same command (or **Run All** in the notebook) and finished
stages are loaded instead of recomputed. Resumed output is identical to an uninterrupted
run.

A run's checkpoints are keyed on its arguments and every result-relevant setting in
`config.py`, so changing a setting starts a fresh run automatically. After a code change
that alters results, bump `CHECKPOINT_TAG` in `config.py`. Flags: `--no-resume`
(recompute this run's stages), `--clear-checkpoints` (delete all saved stages). Set
`ER_CHECKPOINT_DIR` to keep checkpoints elsewhere (must survive restarts).

### 2. Train Only (for iteration)

```bash
python code/business_entity_resolution/src/run_pipeline.py --mode train --sample-size 10000
```

Trains and evaluates on a sample without test prediction. Faster for
hyperparameter tuning.

### 3. Predict Only (with saved model)

```bash
python code/business_entity_resolution/src/run_pipeline.py --mode predict
```

Loads the saved model from `src/models/xgb_model.pkl` and runs test prediction.

### 4. Evaluate Blocking (recall vs. cost)

```bash
cd code/business_entity_resolution/src
python eval_blocking.py --country us --n-queries 5000                       # full pool
python eval_blocking.py --country india --n-queries 1000 --pool-fraction 0.05 --out india.json
```

Blocks S1 training queries against the country's S2+S3 pool (full, or a
Bernoulli sample with every true match kept) and prints, per strategy, pair
and entity recall, the share of true pairs no other strategy finds,
candidates per entity and ms per query, plus a pre-filter cap → recall curve.
Use the full pool (`--pool-fraction 1.0`) for decisions: the training
pipeline's sampled pool overstates blocking recall.

### 5. Validate Output

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Key Design Decisions

1. **Country-by-country test prediction** — The test set has ~1.7M S1 entities
   and ~10M pool records. Processing all at once would require >32GB RAM.
   Country-by-country processing keeps peak memory under 8GB.

2. **Sampled pool for training** — Instead of loading all S2/S3 training records
   (~10M), we load only ground-truth-referenced records plus a random sample per
   country. This keeps training fast without losing positive signal.

3. **Character n-gram TF-IDF** — `char_wb` analyser with (2,4)-grams handles
   typos, abbreviations, and transliterations better than word-level TF-IDF.

4. **Dense scoring for TF-IDF retrieval** — common character n-grams make
   every query-vs-pool similarity row ~100% dense, so sparse×sparse scipy
   blocking cost ~0.6 s/query on the India test pool. Queries are instead
   densified in chunks and scored with one sparse(pool)×dense(queries) product
   on the GPU, followed by an exact two-stage block top-K (~500× faster than
   CuPy's sort-based argpartition). `max_df=0.05` drops the most common n-grams.

5. **Blocker selection by measurement** — `eval_blocking.py` showed the name
   prefix / sorted-token / address-number key indexes add ≤0.12% unique recall
   but 110–145 candidates per entity, pushing the union past the pre-filter
   cap and lowering final recall (India 98.5% → 88.9%). They are off by default.

6. **Transliteration** — ≥10.8% of Indian true pairs have the name in a
   different script on each side; transliterating raises char TF-IDF entity
   recall from 85.5% to 91.2% (India, 5% pool sample).

7. **Parallelisation** — `ThreadPoolExecutor` for I/O-bound file loading;
   `ProcessPoolExecutor` for CPU-bound feature computation.

## Output Files

- `output/matching_results.tsv` — Final entity matches (scored on leaderboard)
- `output/candidate_pairs.tsv` — Blocking candidate set (for pipeline audit)
