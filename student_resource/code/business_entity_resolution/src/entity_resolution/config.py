"""Paths and hyperparameters for the entity-resolution pipeline."""
import multiprocessing as mp
import os

# ===== PATHS ==================================================================
PKG_DIR  = os.path.dirname(os.path.abspath(__file__))          # src/entity_resolution
SRC_DIR  = os.path.dirname(PKG_DIR)                            # src
BASE_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))  # student_resource
TRAIN_S1 = os.path.join(BASE_DIR, "dataset", "train", "train_source1.tsv")
TRAIN_S2 = os.path.join(BASE_DIR, "dataset", "train", "train_source2.tsv")
TRAIN_S3 = os.path.join(BASE_DIR, "dataset", "train", "train_source3.tsv")
TRAIN_GT = os.path.join(BASE_DIR, "dataset", "train", "train_ground_truth.tsv")
TEST_S1  = os.path.join(BASE_DIR, "dataset", "test", "test_source1.tsv")
TEST_S2  = os.path.join(BASE_DIR, "dataset", "test", "test_source2.tsv")
TEST_S3  = os.path.join(BASE_DIR, "dataset", "test", "test_source3.tsv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(SRC_DIR, "models", "xgb_model.pkl")
CACHE_DIR  = os.path.join(BASE_DIR, ".cache")       # parquet cache for preprocessed data
EMBED_CACHE_DIR = os.path.join(CACHE_DIR, "embeddings")  # .npy cache for HNSW embeddings

# ===== GENERAL ================================================================
TFIDF_TOP_K       = 30              # precision-focused (was 100; caused candidate explosion)
TFIDF_MAX_FEATURES= 200_000         # 200K vocab — plenty of RAM
NEG_POS_RATIO     = 5               # more negatives → more conservative model (was 3)
RANDOM_SEED       = 42
VAL_FRACTION      = 0.1
FEAT_CHUNK        = 25_000          # feature-computation chunk for workers
N_WORKERS         = min(mp.cpu_count(), 16)   # use all 16 cores
PRED_BATCH_SIZE   = 500_000         # large CPU predict batch
WORD_TFIDF_MAX    = 100_000         # bigger word vocab for better recall
WORD_TFIDF_TOP_K  = 30              # word-level top-K

# ---- Preprocessing ----
TRANSLITERATE     = True            # non-Latin scripts (Devanagari, Kannada, ...) and accents -> ASCII
                                    # via anyascii; >=10.8% of Indian true pairs mix scripts

# ---- TF-IDF retrieval (exact top-K cosine, GPU via CuPy with CPU fallback) ----
TFIDF_MAX_DF      = 0.05            # drop n-grams in >5% of pool records: same recall,
                                    # char retrieval ~34% faster (eval_blocking, India 5% pool)
NAME_TFIDF_TOP_K  = 0               # name-only char TF-IDF top-K (0 disables). Measured +0.2pp
                                    # recall for +26 cands/entity on India — off by default
NAME_TFIDF_MAX_FEATURES = 200_000
TFIDF_DEVICE      = "auto"          # "auto" (GPU if CuPy/cuSPARSE work), "cuda" or "cpu"
TFIDF_GPU_MEM_FRACTION = 0.5        # share of free GPU memory one query chunk may use
TFIDF_MAX_CHUNK   = 4_096           # max queries scored per chunk
TFIDF_CPU_CHUNK   = 256             # queries per chunk on the CPU fallback

# ---- Which blockers run (all feed the union before the pre-filter) ----
USE_CHAR_TFIDF    = True
USE_WORD_TFIDF    = True
USE_KEY_INDEXES   = False           # name prefix-5/4, sorted-token, address-number keys. Off:
                                    # <=0.12% unique recall, but +110-145 cands/entity that push
                                    # the union past the pre-filter cap (final recall IN 88.9%
                                    # -> 98.5%, US 98.5% -> 99.8% without them)
USE_HNSW          = True            # only if sentence-transformers + hnswlib are installed
USE_LSH           = False           # datasketch MinHash: duplicates char TF-IDF and its
                                    # pure-Python build takes ~40 min per country at test size

# ---- HNSW configuration ----
HNSW_MODEL_NAME   = "paraphrase-multilingual-MiniLM-L12-v2"  # ~420MB, supports en/hi/fr
HNSW_TOP_K        = 50              # candidates from dense retrieval per query
HNSW_EF_CONSTRUCT = 200             # construction-time accuracy (higher = slower build, better recall)
HNSW_EF_SEARCH    = 100             # query-time accuracy (higher = slower query, better recall)
HNSW_M            = 48              # connections per node (higher = more RAM, better recall)
HNSW_BATCH_SIZE   = 512             # encode batch size for sentence-transformers

# ---- MinHash/LSH configuration ----
LSH_NUM_PERM      = 128             # number of permutations (higher = slower but more accurate)
LSH_THRESHOLD     = 0.3             # Jaccard similarity threshold for LSH
LSH_NGRAM_SIZE    = 3               # character n-gram size for MinHash shingling

# ---- Pre-filter configuration ----
PREFILTER_MAX_CANDIDATES = 100      # balanced cap (80 hurt recall, 200 too noisy)
PREFILTER_MIN_SCORE      = 0.15     # keep soft threshold — let the cap do the filtering

# ---- Hard negative mining ----
HARD_NEG_SCORE_FLOOR = 0.3          # only mine false positives scored above this
HARD_NEG_MIX_RATIO   = 0.6          # 60% hard negatives, 40% random in round 2
HARD_NEG_MAX_RATIO   = 5            # never more than 5:1 neg-to-pos after mining

# ---- Singleton detection ----
SINGLETON_MAX_SCORE_THRESHOLD = 0.4 # if best candidate score < this, mark as singleton
SINGLETON_MARGIN_THRESHOLD    = 0.15 # if gap between best and 2nd-best is < this, cautious
