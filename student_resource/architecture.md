# Business Entity Resolution: System Architecture

This document outlines the architecture and system design of the highly scalable Business Entity Resolution pipeline developed for the ML Challenge.

## System Overview

The entity resolution pipeline is designed to overcome the $O(N^2)$ scalability bottleneck of comparing millions of records across disparate data sources (Source 1, Source 2, Source 3) while maintaining a strict focus on precision-weighted recall (optimizing for the F₀.₅ score). 

The system operates in a multi-stage funnel:
1. **Preprocessing:** Standardization and normalization.
2. **Blocking (Candidate Generation):** Reducing the search space using multi-strategy indices.
3. **Pre-filtering:** Pruning weak candidates to reduce noise.
4. **Feature Engineering:** Extracting rich, multi-dimensional similarities.
5. **Classification:** Two-stage XGBoost model with hard-negative mining.
6. **Inference & Singleton Detection:** Threshold-based classification with margin safety.

---

## 1. Preprocessing & Normalization

To handle noisy business names, legal suffixes, and inconsistent address formats across multiple countries (US, India, France), the preprocessing layer applies:
- **Lowercasing & Whitespace Normalization:** Stripping excess whitespace and standardizing spacing.
- **Punctuation Stripping:** Removing special characters that do not contribute to entity identity.
- **Null Handling:** Safely managing missing fields (especially addresses).
- **Format Standardization:** Translating raw formats into a uniform schema for downstream blocking.

## 2. Multi-Strategy Blocking

A naive comparison of 2.2M Source 1 entities against ~5M S2/S3 records results in over $10^{13}$ pairs. The Blocking Layer reduces this to roughly 100 candidate pairs per entity using an ensemble of distinct algorithms to guarantee high recall:

- **TF-IDF Char N-Grams:** Creates sparse vectors using character n-grams (sizes 2-4) and retrieves top-K matches via fast matrix multiplication.
- **Word-Level TF-IDF:** Captures token-level similarities that character n-grams might miss.
- **Exact Match Strategies:** 
  - *Name Key:* Matches on exact normalized business names.
  - *Address Number:* Extracts and matches numeric components of addresses.
- **MinHash / LSH (Locality Sensitive Hashing):** Efficiently approximates Jaccard similarity for rapid nearest-neighbor lookup on n-grams.
- **HNSW Semantic Search:** Uses dense embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) indexed in a Hierarchical Navigable Small World (HNSW) graph for semantic matching, handling transliterations and abbreviations that bypass exact text matching.

## 3. Pre-filtering

The blocking ensemble yields hundreds of candidates (often ~200+ per entity). To prevent the classifier from being overwhelmed by "easy" negatives, a fast heuristic filter is applied:
- **`PREFILTER_MAX_CANDIDATES` (100):** Strictly caps the number of candidates passed to the classifier.
- **`PREFILTER_MIN_SCORE` (0.15):** Drops candidates that fall below a basic string-similarity heuristic.
This focused candidate set ensures the machine learning model spends its capacity differentiating hard boundaries.

## 4. Feature Engineering

For each candidate pair, the pipeline computes a dense feature vector of ~40-48 dimensions:
- **String Distances:** Jaro-Winkler, Levenshtein, and generic RapidFuzz ratios for names, addresses, and concatenated text.
- **TF-IDF Similarities:** Precomputed cosine similarities from the blocking stage.
- **Token Overlap:** Intersection-over-Union (IoU) of tokens.
- **Phonetic Encoding (Dynamic):** If `jellyfish` is available, extracts Soundex, Metaphone, and NYSIIS phonetic codes to capture auditory similarities (crucial for transliterations).
- **Numeric Extraction:** Matches ZIP codes, PO boxes, and street numbers.

## 5. Classification Model (XGBoost)

The core classifier is an XGBoost model optimized for `binary:logistic` objective with an `eval_metric` of `logloss`.

### Training Dynamics:
- **Negative-to-Positive Ratio (5:1):** Heavily samples negative pairs from the blocking output to teach the model strict precision.
- **Hard Negative Mining:** 
  1. **Round 1:** Trains a baseline model.
  2. **Mining:** Evaluates the model on the training set to find *False Positives* (pairs scored highly by the model but are actually negatives).
  3. **Round 2:** Injects these "Hard Negatives" into the training set (at a 60/40 mix of hard to random negatives) and retrains the model. This forces the decision boundary to become much sharper.
- **Hyperparameter Tuning:** Grid search ensures optimal depth, learning rate, and regularization on the validation set.

## 6. Inference & Post-Processing

- **F₀.₅ Threshold Optimization:** Instead of a default 0.5 threshold, the pipeline empirically searches for the exact probability threshold that maximizes the F₀.₅ score on the validation set (heavily weighting precision).
- **Singleton Detection:** Entities that genuinely have no match in the pool shouldn't be forced into a pair. The system applies heuristics:
  - If the highest-scoring candidate is below `SINGLETON_MAX_SCORE_THRESHOLD` (0.4).
  - If the gap between the top candidate and the second-best candidate is below `SINGLETON_MARGIN_THRESHOLD` (0.15) indicating low confidence.
  In these cases, the entity is predicted as a singleton.

## 7. Caching & State Management

To enable rapid iteration, intermediate states (like encoded TF-IDF matrices and HNSW graphs) are cached to disk as `.parquet` files. The system automatically bypasses expensive blocking calculations on subsequent runs if the dataset sample size hasn't changed.
