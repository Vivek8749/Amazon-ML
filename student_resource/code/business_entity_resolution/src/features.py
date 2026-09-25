"""
Feature engineering for entity resolution pairs.
Computes similarity features between S1 and candidate (S2/S3) records.
"""
import re
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

from . import preprocessing as pp


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def overlap_coefficient(set_a: set, set_b: set) -> float:
    """Overlap coefficient (Szymkiewicz–Simpson)."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    min_size = min(len(set_a), len(set_b))
    return intersection / min_size if min_size > 0 else 0.0


def dice_coefficient(set_a: set, set_b: set) -> float:
    """Dice coefficient between two token sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    return 2 * intersection / (len(set_a) + len(set_b))


def containment_similarity(set_a: set, set_b: set) -> float:
    """
    Asymmetric containment: |A ∩ B| / |A|.
    Useful for checking if the shorter name is 'contained' in the longer.
    """
    if not set_a:
        return 0.0
    return len(set_a & set_b) / len(set_a)


def token_sort_ratio(s1: str, s2: str) -> float:
    """Token-sort fuzzy ratio (order-invariant)."""
    return fuzz.token_sort_ratio(s1, s2) / 100.0


def token_set_ratio(s1: str, s2: str) -> float:
    """Token-set fuzzy ratio (handles subset/superset)."""
    return fuzz.token_set_ratio(s1, s2) / 100.0


def partial_ratio(s1: str, s2: str) -> float:
    """Partial ratio: best partial string match."""
    return fuzz.partial_ratio(s1, s2) / 100.0


def normalised_levenshtein(s1: str, s2: str) -> float:
    """1 - normalised Levenshtein distance."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    dist = Levenshtein.normalized_distance(s1, s2)
    return 1.0 - dist


def jaro_winkler_sim(s1: str, s2: str) -> float:
    """Jaro-Winkler similarity."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return JaroWinkler.similarity(s1, s2)


def _extract_numbers(text: str) -> list:
    """Extract all numbers from a string."""
    return re.findall(r"\d+", text)


def number_match_ratio(s1: str, s2: str) -> float:
    """Fraction of numbers in s1 that also appear in s2."""
    nums1 = set(_extract_numbers(s1))
    nums2 = set(_extract_numbers(s2))
    if not nums1:
        return 1.0 if not nums2 else 0.5  # No numbers in either = neutral
    return len(nums1 & nums2) / len(nums1)


def first_token_match(s1: str, s2: str) -> float:
    """Whether the first token matches (important for business names)."""
    t1 = s1.split()
    t2 = s2.split()
    if not t1 or not t2:
        return 0.0
    return 1.0 if t1[0] == t2[0] else 0.0


def length_ratio(s1: str, s2: str) -> float:
    """Ratio of lengths (shorter/longer)."""
    l1, l2 = len(s1), len(s2)
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    return min(l1, l2) / max(l1, l2)


def compute_features(s1_record: dict, s2_record: dict) -> dict:
    """
    Compute all similarity features between an S1 record and an S2/S3 record.
    
    Args:
        s1_record: dict with keys: name_norm, addr_norm, name_tokens, addr_tokens,
                   addr_nums, country_norm, business_name, business_address
        s2_record: dict with same keys
    
    Returns:
        dict of feature_name -> float
    """
    n1 = _safe_str(s1_record.get("name_norm", ""))
    n2 = _safe_str(s2_record.get("name_norm", ""))
    a1 = _safe_str(s1_record.get("addr_norm", ""))
    a2 = _safe_str(s2_record.get("addr_norm", ""))

    # Token sets
    nt1 = s1_record.get("name_tokens", set()) or pp.extract_tokens(n1)
    nt2 = s2_record.get("name_tokens", set()) or pp.extract_tokens(n2)
    at1 = s1_record.get("addr_tokens", set()) or pp.extract_tokens(a1)
    at2 = s2_record.get("addr_tokens", set()) or pp.extract_tokens(a2)
    an1 = s1_record.get("addr_nums", set()) or pp.extract_numeric_tokens(a1)
    an2 = s2_record.get("addr_nums", set()) or pp.extract_numeric_tokens(a2)

    features = {}

    # ─── Name features ────────────────────────────────────────────────────
    features["name_levenshtein"] = normalised_levenshtein(n1, n2)
    features["name_jaro_winkler"] = jaro_winkler_sim(n1, n2)
    features["name_token_sort_ratio"] = token_sort_ratio(n1, n2)
    features["name_token_set_ratio"] = token_set_ratio(n1, n2)
    features["name_partial_ratio"] = partial_ratio(n1, n2)
    features["name_jaccard"] = jaccard_similarity(nt1, nt2)
    features["name_overlap"] = overlap_coefficient(nt1, nt2)
    features["name_dice"] = dice_coefficient(nt1, nt2)
    features["name_containment_12"] = containment_similarity(nt1, nt2)
    features["name_containment_21"] = containment_similarity(nt2, nt1)
    features["name_first_token_match"] = first_token_match(n1, n2)
    features["name_length_ratio"] = length_ratio(n1, n2)
    features["name_token_count_diff"] = abs(len(nt1) - len(nt2))

    # ─── Address features ─────────────────────────────────────────────────
    features["addr_levenshtein"] = normalised_levenshtein(a1, a2)
    features["addr_jaro_winkler"] = jaro_winkler_sim(a1, a2)
    features["addr_token_sort_ratio"] = token_sort_ratio(a1, a2)
    features["addr_token_set_ratio"] = token_set_ratio(a1, a2)
    features["addr_partial_ratio"] = partial_ratio(a1, a2)
    features["addr_jaccard"] = jaccard_similarity(at1, at2)
    features["addr_overlap"] = overlap_coefficient(at1, at2)
    features["addr_dice"] = dice_coefficient(at1, at2)
    features["addr_length_ratio"] = length_ratio(a1, a2)

    # Address number matching
    features["addr_num_jaccard"] = jaccard_similarity(an1, an2)
    features["addr_num_overlap"] = overlap_coefficient(an1, an2)
    features["addr_num_match_ratio_12"] = number_match_ratio(a1, a2)
    features["addr_num_match_ratio_21"] = number_match_ratio(a2, a1)

    # ─── Cross features ───────────────────────────────────────────────────
    # Combined name + address similarity
    combined1 = f"{n1} {a1}"
    combined2 = f"{n2} {a2}"
    features["combined_token_sort"] = token_sort_ratio(combined1, combined2)
    features["combined_token_set"] = token_set_ratio(combined1, combined2)
    features["combined_jaccard"] = jaccard_similarity(nt1 | at1, nt2 | at2)

    # Country match
    c1 = _safe_str(s1_record.get("country_norm", ""))
    c2 = _safe_str(s2_record.get("country_norm", ""))
    features["country_match"] = 1.0 if c1 == c2 else 0.0

    # Name-in-address check (sometimes name is part of address or vice versa)
    features["name_in_addr"] = 1.0 if (n1 and n1 in a2) or (n2 and n2 in a1) else 0.0

    # Maximum of symmetric features (useful for unordered comparison)
    features["max_name_containment"] = max(
        features["name_containment_12"], features["name_containment_21"]
    )
    features["max_addr_num_match"] = max(
        features["addr_num_match_ratio_12"], features["addr_num_match_ratio_21"]
    )

    # Aggregate score (simple average of best features)
    features["agg_name_score"] = np.mean([
        features["name_jaro_winkler"],
        features["name_token_sort_ratio"],
        features["name_jaccard"],
    ])
    features["agg_addr_score"] = np.mean([
        features["addr_jaro_winkler"],
        features["addr_token_sort_ratio"],
        features["addr_jaccard"],
    ])
    features["agg_combined_score"] = (
        0.5 * features["agg_name_score"] + 0.5 * features["agg_addr_score"]
    )

    return features


FEATURE_NAMES = list(compute_features(
    {"name_norm": "", "addr_norm": "", "name_tokens": set(),
     "addr_tokens": set(), "addr_nums": set(), "country_norm": ""},
    {"name_norm": "", "addr_norm": "", "name_tokens": set(),
     "addr_tokens": set(), "addr_nums": set(), "country_norm": ""},
).keys())
