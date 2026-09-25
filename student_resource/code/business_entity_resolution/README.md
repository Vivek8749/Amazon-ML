# Business Entity Resolution — Code README

## Overview

This pipeline resolves business entities across three independent data sources
(S1, S2, S3) by determining which records refer to the same real-world business.
It uses a **TF-IDF blocking + XGBoost classifier** approach optimised for the
precision-heavy F₀.₅ metric.

## Architecture

```
run_pipeline.py (standalone, self-contained entry point)
│
├── Preprocessing
│   ├── Vectorised text cleaning (lowercase, bracket/symbol handling)
│   ├── Legal suffix normalisation (Corporation→corp, Ltd.→ltd, S.A.R.L.→sarl)
│   └── Address abbreviation normalisation (Street→st, Boulevard→blvd, etc.)
│
├── Blocking / Candidate Generation
│   ├── TF-IDF character n-gram (2–4) cosine similarity (top-K per query)
│   ├── Name-key inverted index (first 5 characters + country)
│   └── Address-number inverted index (sorted numeric tokens + country)
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

### Modular Code (Alternative Reference)

The `src/` directory also contains a modular version of the pipeline:

| File | Purpose |
|---|---|
| `config.py` | Centralised configuration constants |
| `preprocessing.py` | Text normalisation with legal suffixes, address abbrevs |
| `blocking.py` | `TFIDFBlocker` class + supplementary index generation |
| `features.py` | 33 similarity features via `compute_features()` |
| `matcher.py` | XGBoost train/predict with threshold optimisation |
| `evaluation.py` | Macro-averaged F₀.₅ scoring |
| `io_utils.py` | TSV output formatting |
| `pipeline.py` | Orchestration (train/full/predict modes) |

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

### 4. Validate Output

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

4. **Multi-strategy blocking** — TF-IDF is supplemented by name-key and
   address-number inverted indexes to catch cases where character similarity
   is low but exact tokens overlap.

5. **Parallelisation** — `ThreadPoolExecutor` for I/O-bound file loading;
   `ProcessPoolExecutor` for CPU-bound feature computation. TF-IDF blocking
   uses thread-level batch parallelism.

## Output Files

- `output/matching_results.tsv` — Final entity matches (scored on leaderboard)
- `output/candidate_pairs.tsv` — Blocking candidate set (for pipeline audit)
