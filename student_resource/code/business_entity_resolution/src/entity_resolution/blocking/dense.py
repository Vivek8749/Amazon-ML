"""Dense retrieval blocking: sentence-transformer embeddings + HNSW index."""
import time

import numpy as np

from ..cache import embed_cache_key, embed_cache_load, embed_cache_save
from ..config import (
    HNSW_BATCH_SIZE, HNSW_EF_CONSTRUCT, HNSW_EF_SEARCH, HNSW_M,
    HNSW_MODEL_NAME, HNSW_TOP_K,
)

try:
    from sentence_transformers import SentenceTransformer
    import hnswlib
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False

# ===== HNSW DENSE RETRIEVAL ====================================================

def load_sbert_model():
    """Load the sentence-transformer model (cached after first download)."""
    if not HNSW_AVAILABLE:
        return None
    print(f"[HNSW] Loading model: {HNSW_MODEL_NAME}...")
    t0 = time.time()
    model = SentenceTransformer(HNSW_MODEL_NAME, device="cuda")
    print(f"[HNSW] Model loaded in {time.time()-t0:.1f}s")
    return model


def _encode_texts(model, texts, batch_size=HNSW_BATCH_SIZE, desc="Encoding",
                   use_cache=True):
    """Encode texts to dense vectors using sentence-transformers.
    Results are cached to disk as .npy files keyed on text content hash."""
    # --- Try embedding cache first ---
    cache_key = None
    if use_cache:
        cache_key = embed_cache_key(texts)
        cached = embed_cache_load(cache_key)
        if cached is not None:
            return cached

    print(f"[HNSW] Encoding {len(texts):,} texts ({desc})...")
    t0 = time.time()
    embeddings = model.encode(
        texts.tolist() if hasattr(texts, 'tolist') else list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalise for cosine similarity via inner product
    )
    embeddings = embeddings.astype(np.float32)
    print(f"[HNSW] Encoded in {time.time()-t0:.1f}s, shape={embeddings.shape}")

    # --- Save to embedding cache ---
    if use_cache and cache_key:
        embed_cache_save(cache_key, embeddings)

    return embeddings


def build_hnsw_index(pool_df, sbert_model):
    """Build an HNSW index over the pool (S2+S3) embeddings.

    Returns: (index, pool_ids, pool_countries, sbert_model)
    """
    if not HNSW_AVAILABLE or sbert_model is None:
        print("[HNSW] Skipping — not available")
        return None, None, None, None

    pool_texts = pool_df["combined"].values
    pool_ids = pool_df["entity_id"].values
    pool_countries = pool_df["country_norm"].values

    # Encode pool
    embeddings = _encode_texts(sbert_model, pool_texts, desc="Pool encoding")
    dim = embeddings.shape[1]

    # Build HNSW index
    print(f"[HNSW] Building index (dim={dim}, M={HNSW_M}, ef_construct={HNSW_EF_CONSTRUCT})...")
    t0 = time.time()
    index = hnswlib.Index(space='ip', dim=dim)  # inner product ≈ cosine (with L2-normed vecs)
    index.init_index(max_elements=len(embeddings), ef_construction=HNSW_EF_CONSTRUCT, M=HNSW_M)
    index.add_items(embeddings, np.arange(len(embeddings)))
    index.set_ef(HNSW_EF_SEARCH)
    print(f"[HNSW] Index built in {time.time()-t0:.1f}s")

    return index, pool_ids, pool_countries, sbert_model


def hnsw_block_batch(query_texts, query_countries, index, pool_ids, pool_countries,
                     sbert_model, top_k=HNSW_TOP_K):
    """Query HNSW index for nearest neighbours, with country filtering.

    Args:
        query_texts: array of combined text strings for S1 entities
        query_countries: array of country codes for S1 entities
        index: hnswlib.Index
        pool_ids: array mapping index position -> entity_id
        pool_countries: array mapping index position -> country
        sbert_model: sentence-transformer model for encoding queries
        top_k: number of candidates per query

    Returns:
        list of lists of candidate entity_ids (one list per query)
    """
    if not HNSW_AVAILABLE or index is None or sbert_model is None:
        return [[] for _ in range(len(query_texts))]

    # Encode queries
    query_embeddings = _encode_texts(sbert_model, query_texts, desc="Query encoding")

    # Query HNSW — retrieve more than top_k to account for country filtering
    fetch_k = min(top_k * 3, index.get_current_count())
    print(f"[HNSW] Querying {len(query_embeddings):,} queries (fetch_k={fetch_k})...")
    t0 = time.time()
    labels, distances = index.knn_query(query_embeddings, k=fetch_k)
    print(f"[HNSW] Query done in {time.time()-t0:.1f}s")

    # Country-filtered results
    results = []
    for i in range(len(query_texts)):
        country = query_countries[i]
        row_labels = labels[i]
        # Filter by country, take top_k
        filtered = []
        for idx in row_labels:
            if idx < len(pool_countries) and pool_countries[idx] == country:
                filtered.append(pool_ids[idx])
                if len(filtered) >= top_k:
                    break
        results.append(filtered)

    return results
