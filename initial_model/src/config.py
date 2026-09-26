"""
Configuration constants for the Entity Resolution pipeline.
"""
import os

GPU_DEVICE = "cuda"

# ΓöÇΓöÇΓöÇ paths ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TRAIN_S1 = os.path.join(BASE_DIR, "student_resource", "dataset", "train", "train_source1.tsv")
TRAIN_S2 = os.path.join(BASE_DIR, "student_resource", "dataset", "train", "train_source2.tsv")
TRAIN_S3 = os.path.join(BASE_DIR, "student_resource", "dataset", "train", "train_source3.tsv")
TRAIN_GT = os.path.join(BASE_DIR, "student_resource", "dataset", "train", "train_ground_truth.tsv")

TEST_S1 = os.path.join(BASE_DIR, "student_resource", "dataset", "test", "test_source1.tsv")
TEST_S2 = os.path.join(BASE_DIR, "student_resource", "dataset", "test", "test_source2.tsv")
TEST_S3 = os.path.join(BASE_DIR, "student_resource", "dataset", "test", "test_source3.tsv")

OUTPUT_DIR = os.path.join(BASE_DIR, "student_resource", "output")
MATCHING_RESULTS = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

MODEL_DIR = os.path.join(BASE_DIR, "student_resource", "code", "business_entity_resolution", "models")


# ΓöÇΓöÇΓöÇ pipeline parameters ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Blocking
TFIDF_TOP_K = 20           # top-K candidates per query from TF-IDF blocking
TFIDF_NGRAM_RANGE = (2, 4) # character n-gram range for TF-IDF
TFIDF_MAX_FEATURES = 200_000

# Training
VAL_FRACTION = 0.1         # fraction of S1 entities to hold out for validation
RANDOM_SEED = 42
NEG_POS_RATIO = 3          # negatives per positive in training pairs

# XGBoost
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 500,
    "early_stopping_rounds": 30,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "tree_method": "hist",
    "device": GPU_DEVICE,
}

# Threshold for final matching (optimized during validation)
DEFAULT_MATCH_THRESHOLD = 0.5

# Processing
BATCH_SIZE = 50_000        # for batch blocking
NUM_WORKERS = os.cpu_count() or 4
