"""Cheap 3-signal score used to cap candidates before full feature computation."""
from rapidfuzz.distance import JaroWinkler

from ..config import PREFILTER_MAX_CANDIDATES, PREFILTER_MIN_SCORE

# ===== LIGHTWEIGHT PRE-FILTER =================================================

def quick_prefilter(candidates, s1_names, s1_ids,
                    pool_name_lookup, pool_addr_lookup, s1_addr_lookup,
                    max_candidates=PREFILTER_MAX_CANDIDATES,
                    min_score=PREFILTER_MIN_SCORE):
    """Reduce candidate count per entity using a fast 3-feature score.

    For each (S1, candidate) pair, computes:
      1. Jaro-Winkler similarity on normalised names (weight 0.45)
      2. Jaro-Winkler similarity on normalised addresses (weight 0.30)
      3. Token overlap coefficient on name tokens (weight 0.25)

    Candidates below min_score are dropped. If more than max_candidates remain,
    only the top-scoring ones are kept. This is ~50x faster than the full
    40-feature computation because it avoids Levenshtein, partial ratio,
    SequenceMatcher, and cross-field features.

    Args:
        candidates: {s1_id: [candidate_ids]}
        s1_names: array of s1 name_clean values
        s1_ids: array of s1 entity_ids
        pool_name_lookup: {entity_id: name_clean}
        pool_addr_lookup: {entity_id: addr_clean}
        s1_addr_lookup: {entity_id: addr_clean}
        max_candidates: maximum candidates to keep per entity
        min_score: minimum score threshold

    Returns:
        filtered {s1_id: [candidate_ids]}
    """
    s1_name_map = dict(zip(s1_ids, s1_names))
    filtered = {}
    total_before = 0
    total_after = 0

    for s1_id, cands in candidates.items():
        total_before += len(cands)

        if len(cands) <= max_candidates:
            filtered[s1_id] = cands
            total_after += len(cands)
            continue

        n1 = s1_name_map.get(s1_id, "")
        a1 = s1_addr_lookup.get(s1_id, "")
        nt1 = set(n1.split()) if n1 else set()

        scored = []
        for cid in cands:
            n2 = pool_name_lookup.get(cid, "")
            a2 = pool_addr_lookup.get(cid, "")

            # Feature 1: Jaro-Winkler on names
            if n1 and n2:
                name_jw = JaroWinkler.similarity(n1, n2)
            elif not n1 and not n2:
                name_jw = 1.0
            else:
                name_jw = 0.0

            # Feature 2: Jaro-Winkler on addresses
            if a1 and a2:
                addr_jw = JaroWinkler.similarity(a1, a2)
            elif not a1 and not a2:
                addr_jw = 1.0
            else:
                addr_jw = 0.0

            # Feature 3: Token overlap on names
            nt2 = set(n2.split()) if n2 else set()
            if nt1 and nt2:
                token_ovl = len(nt1 & nt2) / min(len(nt1), len(nt2))
            elif not nt1 and not nt2:
                token_ovl = 1.0
            else:
                token_ovl = 0.0

            score = 0.45 * name_jw + 0.30 * addr_jw + 0.25 * token_ovl
            if score >= min_score:
                scored.append((score, cid))

        # Sort descending and take top max_candidates
        scored.sort(key=lambda x: -x[0])
        filtered[s1_id] = [cid for _, cid in scored[:max_candidates]]
        total_after += len(filtered[s1_id])

    return filtered
