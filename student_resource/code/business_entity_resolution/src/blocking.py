"""
Blocking / Candidate Generation module.

Uses a multi-strategy approach:
1. TF-IDF character n-gram blocking on combined name+address text
2. Exact-token overlap blocking (numeric tokens for address matching)
3. Name-key blocking (consonant skeleton)

All strategies are unioned to maximise recall, then country-filtered.
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import vstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from . import preprocessing as pp
from .config import (
    TFIDF_TOP_K, TFIDF_NGRAM_RANGE, TFIDF_MAX_FEATURES,
    BATCH_SIZE, MODEL_DIR,
)


def _safe_str(val) -> str:
    """Convert a value to string safely, handling NaN."""
    if pd.isna(val):
        return ""
    return str(val)


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """Load a source TSV and add normalised columns."""
    df = pd.read_csv(filepath, sep="\t", dtype=str)
    df = df.fillna("")

    # Normalise
    df["name_norm"] = df.apply(
        lambda r: pp.normalise_name(_safe_str(r["business_name"])), axis=1
    )
    df["addr_norm"] = df.apply(
        lambda r: pp.normalise_address(
            _safe_str(r["business_address"]), _safe_str(r["country"])
        ),
        axis=1,
    )
    df["combined"] = df.apply(
        lambda r: pp.create_combined_text(
            _safe_str(r["business_name"]),
            _safe_str(r["business_address"]),
            _safe_str(r["country"]),
        ),
        axis=1,
    )
    df["name_tokens"] = df["name_norm"].apply(pp.extract_tokens)
    df["addr_tokens"] = df["addr_norm"].apply(pp.extract_tokens)
    df["addr_nums"] = df["addr_norm"].apply(pp.extract_numeric_tokens)
    df["name_key"] = df["name_norm"].apply(pp.get_name_key)

    df["country_norm"] = df["country"].str.lower().str.strip()

    return df


class TFIDFBlocker:
    """
    TF-IDF character n-gram based blocking.
    Fits a TF-IDF vectoriser on the candidate pool (S2+S3),
    then queries S1 entities to retrieve top-K candidates.
    """

    def __init__(self, ngram_range=TFIDF_NGRAM_RANGE,
                 max_features=TFIDF_MAX_FEATURES, top_k=TFIDF_TOP_K):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.top_k = top_k
        self.vectoriser = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.pool_matrix = None
        self.pool_ids = None
        self.pool_countries = None

    def fit(self, pool_df: pd.DataFrame):
        """Fit the vectoriser on the candidate pool (S2+S3)."""
        print(f"[TFIDFBlocker] Fitting on {len(pool_df)} candidates...")
        self.pool_ids = pool_df["entity_id"].values
        self.pool_countries = pool_df["country_norm"].values

        texts = pool_df["combined"].values
        self.pool_matrix = self.vectoriser.fit_transform(texts)
        print(f"[TFIDFBlocker] Pool matrix shape: {self.pool_matrix.shape}")
        return self

    def query_batch(self, query_texts: np.ndarray, query_countries: np.ndarray) -> list:
        """
        For a batch of query texts, return top-K candidate IDs per query.
        Country filtering applied: only candidates from the same country are considered.
        """
        query_matrix = self.vectoriser.transform(query_texts)
        # Compute cosine similarity (both are L2-normalised by TF-IDF)
        scores = query_matrix.dot(self.pool_matrix.T)

        results = []
        for i in range(scores.shape[0]):
            row = scores[i]
            if hasattr(row, "toarray"):
                row = row.toarray().ravel()
            else:
                row = np.asarray(row).ravel()

            # Country filter
            country = query_countries[i]
            country_mask = (self.pool_countries == country)
            row[~country_mask] = 0.0

            # Top-K
            if self.top_k < len(row):
                top_idx = np.argpartition(row, -self.top_k)[-self.top_k:]
                top_idx = top_idx[row[top_idx] > 0]
            else:
                top_idx = np.where(row > 0)[0]

            candidates = self.pool_ids[top_idx].tolist()
            results.append(candidates)
        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "TFIDFBlocker":
        with open(path, "rb") as f:
            return pickle.load(f)


def token_overlap_blocking(s1_row: pd.Series, pool_df: pd.DataFrame,
                           min_name_overlap: int = 1,
                           min_addr_num_overlap: int = 1) -> list:
    """
    Token-overlap based blocking for a single S1 entity.
    Returns candidate entity_ids from pool_df.
    """
    s1_name_tokens = s1_row["name_tokens"]
    s1_addr_nums = s1_row["addr_nums"]
    s1_country = s1_row["country_norm"]

    # Filter by country first
    pool_country = pool_df[pool_df["country_norm"] == s1_country]

    candidates = set()

    # Name token overlap
    if s1_name_tokens:
        for idx, row in pool_country.iterrows():
            overlap = len(s1_name_tokens & row["name_tokens"])
            if overlap >= min_name_overlap:
                # Additionally check address numeric overlap if available
                if s1_addr_nums and row["addr_nums"]:
                    num_overlap = len(s1_addr_nums & row["addr_nums"])
                    if num_overlap >= min_addr_num_overlap:
                        candidates.add(row["entity_id"])
                elif overlap >= 2:  # Require stronger name overlap if no addr nums
                    candidates.add(row["entity_id"])

    return list(candidates)


def generate_candidates(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    tfidf_blocker: TFIDFBlocker | None = None,
    top_k: int = TFIDF_TOP_K,
) -> dict:
    """
    Generate candidate pairs for all S1 entities.
    Uses TF-IDF blocking as the primary strategy.
    Returns dict: {s1_entity_id: [list of candidate entity_ids]}
    """
    print(f"[Blocking] Generating candidates for {len(s1_df)} S1 entities...")

    if tfidf_blocker is None:
        tfidf_blocker = TFIDFBlocker(top_k=top_k)
        tfidf_blocker.fit(pool_df)

    # Build name-key index for supplementary blocking
    print("[Blocking] Building name-key inverted index...")
    name_key_index = {}
    for _, row in tqdm(pool_df.iterrows(), total=len(pool_df), desc="Name-key index"):
        key = row["name_key"]
        country = row["country_norm"]
        if key:
            name_key_index.setdefault((key, country), []).append(row["entity_id"])

    # Build addr-num index for supplementary blocking
    print("[Blocking] Building addr-num inverted index...")
    addr_num_index = {}
    for _, row in tqdm(pool_df.iterrows(), total=len(pool_df), desc="Addr-num index"):
        nums = row["addr_nums"]
        country = row["country_norm"]
        if nums:
            nums_key = "_".join(sorted(nums)[:3])
            if nums_key:
                addr_num_index.setdefault((nums_key, country), []).append(row["entity_id"])

    candidates = {}
    total_s1 = len(s1_df)

    # Process in batches for TF-IDF
    for batch_start in tqdm(range(0, total_s1, BATCH_SIZE), desc="TF-IDF blocking"):
        batch_end = min(batch_start + BATCH_SIZE, total_s1)
        batch = s1_df.iloc[batch_start:batch_end]

        batch_texts = batch["combined"].values
        batch_countries = batch["country_norm"].values
        batch_ids = batch["entity_id"].values

        tfidf_candidates = tfidf_blocker.query_batch(batch_texts, batch_countries)

        for i, s1_id in enumerate(batch_ids):
            cands = set(tfidf_candidates[i])

            # Add name-key candidates
            nk = batch.iloc[i]["name_key"]
            country = batch_countries[i]
            if nk and (nk, country) in name_key_index:
                cands.update(name_key_index[(nk, country)][:50])

            # Add addr-num candidates
            nums = batch.iloc[i]["addr_nums"]
            if nums:
                nums_key = "_".join(sorted(nums)[:3])
                if nums_key and (nums_key, country) in addr_num_index:
                    cands.update(addr_num_index[(nums_key, country)][:50])

            candidates[s1_id] = list(cands)

    print(f"[Blocking] Generated candidates for {len(candidates)} S1 entities")
    total_cands = sum(len(v) for v in candidates.values())
    avg_cands = total_cands / max(len(candidates), 1)
    print(f"[Blocking] Total candidate pairs: {total_cands:,}, avg per S1: {avg_cands:.1f}")

    return candidates, tfidf_blocker
