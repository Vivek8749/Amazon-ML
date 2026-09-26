"""MinHash/LSH blocking on character shingles, one LSH index per country."""
import time

from tqdm import tqdm

from ..config import LSH_NGRAM_SIZE, LSH_NUM_PERM, LSH_THRESHOLD

try:
    from datasketch import MinHash, MinHashLSH
    LSH_AVAILABLE = True
except ImportError:
    LSH_AVAILABLE = False

# ===== MINHASH / LSH ==========================================================

def _text_to_shingles(text, n=LSH_NGRAM_SIZE):
    """Convert text to a set of character n-gram shingles."""
    if not text or len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def _create_minhash(shingles, num_perm=LSH_NUM_PERM):
    """Create a MinHash signature from a set of shingles."""
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode('utf-8'))
    return m


def build_minhash_lsh(pool_df, num_perm=LSH_NUM_PERM, threshold=LSH_THRESHOLD):
    """Build a MinHash LSH index over the pool (S2+S3).

    Creates separate LSH indexes per country for efficient country-filtered lookup.

    Returns: (lsh_index_dict, key_to_ids_dict)
        - lsh_index_dict: {country: MinHashLSH}
        - key_to_ids_dict: {lsh_key: entity_id}
    """
    if not LSH_AVAILABLE:
        print("[LSH] Skipping — datasketch not available")
        return None, None

    print(f"[LSH] Building MinHash LSH (num_perm={num_perm}, threshold={threshold})...")
    t0 = time.time()

    countries = pool_df["country_norm"].unique()
    lsh_dict = {}
    key_to_id = {}

    for country in countries:
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        co_df = pool_df[pool_df["country_norm"] == country]

        for _, row in tqdm(co_df.iterrows(), total=len(co_df),
                           desc=f"LSH index [{country}]", leave=False):
            eid = row["entity_id"]
            text = row.get("combined", "")
            shingles = _text_to_shingles(text)
            if not shingles:
                continue
            mh = _create_minhash(shingles, num_perm)
            lsh_key = f"{country}_{eid}"
            try:
                lsh.insert(lsh_key, mh)
                key_to_id[lsh_key] = eid
            except ValueError:
                pass  # duplicate key — skip

        lsh_dict[country] = lsh

    print(f"[LSH] Index built in {time.time()-t0:.1f}s "
          f"({len(key_to_id):,} entries across {len(lsh_dict)} countries)")
    return lsh_dict, key_to_id


def minhash_block_batch(query_texts, query_countries, lsh_dict, key_to_id,
                        num_perm=LSH_NUM_PERM, max_per_query=100):
    """Query the MinHash LSH index for approximate Jaccard neighbours.

    Args:
        query_texts: array of combined text strings
        query_countries: array of country codes
        lsh_dict: {country: MinHashLSH}
        key_to_id: {lsh_key: entity_id}
        num_perm: number of permutations (must match build)
        max_per_query: max candidates per query

    Returns:
        list of lists of candidate entity_ids
    """
    if not LSH_AVAILABLE or lsh_dict is None:
        return [[] for _ in range(len(query_texts))]

    results = []
    for i in range(len(query_texts)):
        country = query_countries[i]
        text = query_texts[i]
        shingles = _text_to_shingles(text)

        if not shingles or country not in lsh_dict:
            results.append([])
            continue

        mh = _create_minhash(shingles, num_perm)
        hits = lsh_dict[country].query(mh)

        # Map LSH keys back to entity_ids
        cands = []
        for key in hits[:max_per_query]:
            if key in key_to_id:
                cands.append(key_to_id[key])
        results.append(cands)

    return results
