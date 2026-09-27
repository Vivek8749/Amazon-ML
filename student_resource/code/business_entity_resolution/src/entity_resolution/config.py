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

# ===== CHECKPOINTS (resume after a killed session) =============================
# Each finished stage is saved here and skipped on the next run with the same
# settings. Must be on storage that survives restarts (on Lightning AI the
# studio's home directory does). Override with the ER_CHECKPOINT_DIR env var.
CHECKPOINT_DIR = os.environ.get("ER_CHECKPOINT_DIR", os.path.join(BASE_DIR, ".checkpoints"))
RESUME         = True               # False: ignore and overwrite saved stages for this run
CHECKPOINT_TAG = "v2"               # bump after a code change that alters results, so
                                    # stages saved by older code are not reused
PRED_SHARD_ENTITIES = 100_000       # test entities scored per checkpointed shard

# ===== GENERAL ================================================================
TFIDF_TOP_K       = 30              # precision-focused (was 100; caused candidate explosion)
TFIDF_MAX_FEATURES= 200_000         # 200K vocab — plenty of RAM
NEG_POS_RATIO     = None            # None: every blocked candidate is a training pair (the
                                    # model learns the real candidate mix; +0.0023 held-out
                                    # F0.5 vs 5 negatives per positive). An int samples negatives
N_MODELS          = 3               # final model = average of this many seeds of the best grid
                                    # config (+0.0021 held-out F0.5 vs a single model)
TRAIN_POOL_EXTRA_PER_COUNTRY = "full"  # training S2+S3 pool: true matches of the sample plus
                                    # this many random records per country; "full" = every
                                    # record of the country, as dense as the test pool
                                    # (None = old default, 20-30K per country)
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
TFIDF_FIT_SAMPLE  = 1_000_000       # learn vocabulary/IDF on this many pool records, then
                                    # transform all records in parallel (fit on 10M took 19 min)

# ---- Which blockers run (all feed the union before the pre-filter) ----
USE_CHAR_TFIDF    = True
USE_WORD_TFIDF    = True
USE_KEY_INDEXES   = False           # name prefix-5/4, sorted-token, address-number keys. Off:
                                    # <=0.12% unique recall, but +110-145 cands/entity that push
                                    # the union past the pre-filter cap (final recall IN 88.9%
                                    # -> 98.5%, US 98.5% -> 99.8% without them)
USE_HNSW          = os.environ.get("ER_USE_HNSW", "1") == "1"  # env ER_USE_HNSW=0 skips dense retrieval;
                                    # only runs if sentence-transformers + hnswlib are installed
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

# ---- Final decision (inference.decide_matches) ----
ONE_TO_ONE        = True            # each S2/S3 record goes to at most one S1 entity (its
                                    # highest-scoring one): in the training ground truth no
                                    # record belongs to two S1 entities (7.64M ids, all distinct)
SCORE_KEEP_FLOOR  = 0.01            # test-time pair scores below this are not stored; must stay
                                    # below the threshold and SINGLETON_MAX_SCORE_THRESHOLD

# ---- Singleton detection ----
SINGLETON_MAX_SCORE_THRESHOLD = 0.4 # if best candidate score < this, mark as singleton
SINGLETON_MARGIN_THRESHOLD    = 0.15 # if gap between best and 2nd-best is < this, cautious
