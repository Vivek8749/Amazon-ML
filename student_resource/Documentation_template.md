# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
We built a scalable entity resolution pipeline using TF-IDF character n-gram blocking combined with an XGBoost binary classifier. The pipeline handles noisy business names (abbreviations, legal suffixes, typos) and inconsistent addresses (format variations, missing components, transliterations) across three countries (US, India, France). Our approach prioritises precision via F₀.₅-optimised threshold selection and country-filtered blocking.

---

## 2. Methodology

### 2.1 Problem Analysis
Key insights discovered during EDA:

- **Scale**: ~2.2M Source 1 entities in training, ~1.7M in test, with ~5M records each in S2/S3. Naive all-pairs comparison is intractable (~10¹³ pairs).
- **Noise patterns observed**:
  - **Legal suffixes**: Corporation vs Corp vs Corp., Inc vs Incorporated, Ltd vs Limited, Pvt vs Private, S.A.R.L. vs SARL (French entities)
  - **Address variations**: Street vs St vs St., Road vs Rd, Avenue vs Ave; missing PIN codes; landmark-based references ("Near SBI ATM"); component reordering
  - **Transliterations**: Hindi names in Latin script with varying spellings
  - **Missing data**: ~5.6% of S1 training entities are singletons (no matches)
- **Unseen country**: France appears only in the test set, requiring country-agnostic feature engineering
- **Metric**: F₀.₅ penalises false merges 2× more than missed matches — precision is paramount

### 2.2 Solution Strategy
**Approach Type:** TF-IDF Blocking + XGBoost Classifier (Hybrid)  
**Core Innovation:** Multi-strategy blocking with vectorised entity-resolution-specific text normalisation, combined with a 29-feature similarity profile scored by an XGBoost classifier with F₀.₅-optimised threshold

---

## 3. Candidate Generation (Blocking)
We reduce the O(N²) comparison space using a multi-strategy blocking approach:

### 3.1 Primary: TF-IDF Character N-gram Blocking
- **Vectorisation**: TF-IDF with `char_wb` analyser, (2,4)-gram range, 200K max features, sublinear TF
- **Similarity**: Sparse cosine similarity between S1 query vectors and the S2+S3 pool matrix
- **Top-K**: 20 candidates per S1 entity after country filtering
- **Batching**: S1 queries processed in batches of 10,000 with thread-level parallelism

### 3.2 Supplementary: Name-Key Inverted Index
- Key: first 5 characters of cleaned business name + country
- Adds up to 50 candidates per key match
- Catches entities where character n-gram TF-IDF scores are low but name prefixes match exactly

### 3.3 Supplementary: Address-Number Inverted Index
- Key: sorted first 3 numeric tokens from address + country
- Adds up to 50 candidates per key match
- Catches entities sharing street numbers/PIN codes even when textual similarity is low

### 3.4 Country Filtering
All blocking strategies filter candidates to the same country as the S1 entity, eliminating cross-country false positives.

- **Blocking keys used:** TF-IDF char n-grams, name-prefix (5 chars), address numeric tokens
- **How we ensured true matches were not lost:** Union of three complementary strategies maximises recall; validation blocking recall tracked during training

---

## 4. Matching Model

**Features used (29 total):**
- **Name features (13):** Normalised Levenshtein similarity, Jaro-Winkler similarity, token-sort ratio, token-set ratio, partial ratio, basic ratio, Jaccard similarity, overlap coefficient, Dice coefficient, directional containment (×2), first-token match, length ratio
- **Address features (9):** Normalised Levenshtein, Jaro-Winkler, token-sort ratio, token-set ratio, partial ratio, Jaccard, overlap, Dice, length ratio
- **Address number features (3):** Jaccard, overlap, directional match ratio
- **Cross features (4):** Combined token-sort ratio, combined token-set ratio, combined Jaccard, absolute token count difference

**Preprocessing before features:**
- Lowercase, bracket removal, `&`/`+` → "and"
- Legal suffix normalisation: `Incorporated` → `inc`, `Corporation` → `corp`, `Limited` → `ltd`, `Private` → `pvt`, `S.A.R.L.` → `sarl`, `S.A.S.` → `sas`, `L.L.C.` → `llc`
- Address abbreviation normalisation: `Street` → `st`, `Road` → `rd`, `Avenue` → `ave`, `Boulevard` → `blvd`, `Building` → `bldg`, `Apartment` → `apt`, `Suite` → `ste`, `District` → `dist`, `Nagar` → `ngr`, `Sector` → `sec`
- Domain suffix removal from names (`.com`, `.org`, `.net`, `.in`, `.fr`)

**Model type:** XGBoost (binary classifier)
- `max_depth=8`, `learning_rate=0.1`, `n_estimators=500`
- `tree_method="hist"` for fast training
- `scale_pos_weight` computed from class ratio for imbalance handling
- Early stopping (30 rounds) on validation log-loss
- `subsample=0.8`, `colsample_bytree=0.8` for regularisation

**Threshold selection method:** F₀.₅ optimisation on held-out validation set using sklearn's `precision_recall_curve`

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [Updated after pipeline completes]
- **Common false positives (wrong merges):**
  - Businesses with very similar names but different addresses in the same city (e.g., franchise locations)
  - Different businesses at the same address (shared office buildings)
- **Common false negatives (missed matches):**
  - Severe name truncation or transliteration differences
  - Complete address reformatting (e.g., "Near SBI ATM, MG Road" vs "12 Mahatma Gandhi Road")
  - Entities outside TF-IDF blocking recall

---

## 6. Conclusion
We built a scalable entity resolution pipeline that handles multi-source, multi-country business records with significant noise. The TF-IDF character n-gram blocking efficiently reduces the search space while maintaining high recall, and the XGBoost classifier with 29 similarity features provides precise matching. The F₀.₅-optimised threshold ensures the precision-heavy metric is maximised. Country-by-country test processing enables the pipeline to handle the full 1.7M-entity test set within memory constraints.

---

## Appendix

### A. Code Artefacts
The complete, runnable code is in `code/business_entity_resolution/`:

```
code/business_entity_resolution/
├── README.md               # Run instructions
├── requirements.txt        # Pinned dependencies
└── src/
    ├── run_pipeline.py     # ★ Main entry point (standalone, self-contained)
    ├── config.py           # Configuration constants
    ├── preprocessing.py    # Text normalisation utilities
    ├── blocking.py         # TFIDFBlocker + inverted indexes
    ├── features.py         # 33 similarity features (modular version)
    ├── matcher.py          # XGBoost train/predict/save/load
    ├── evaluation.py       # Macro F₀.₅ scoring
    ├── io_utils.py         # TSV output formatting
    ├── pipeline.py         # Modular pipeline orchestration
    └── models/
        └── xgb_model.pkl   # Trained model + threshold
```

**To reproduce end-to-end** (from `student_resource/` directory):
```bash
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/run_pipeline.py --mode full --sample-size 20000
```

This generates both `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### B. Additional Results
*Feature importance (top 10, from XGBoost model):*
Will be populated from training output. Key expected top features: `name_jw` (Jaro-Winkler on names), `name_tset` (token-set ratio), `addr_jw` (Jaro-Winkler on addresses), `name_partial` (partial ratio), `comb_tset` (combined token-set ratio).

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
