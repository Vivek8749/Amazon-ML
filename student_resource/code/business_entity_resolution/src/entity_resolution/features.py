"""Pairwise similarity features and their parallel computation."""
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
from tqdm import tqdm

from .config import N_WORKERS

try:
    import jellyfish
    PHONETIC_AVAILABLE = True
except ImportError:
    PHONETIC_AVAILABLE = False

# ===== FEATURE ENGINEERING =====================================================

def _tokens(t):  return set(t.split()) if t else set()
def _nums(t):    return set(re.findall(r"\d+", t)) if t else set()
def _jac(a, b):
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)
def _ovl(a, b):
    if not a or not b: return 0.0
    return len(a & b) / min(len(a), len(b))
def _dice(a, b):
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return 2*len(a & b) / (len(a)+len(b))
def _lr(a, b):
    la, lb = len(a), len(b)
    if la==0 and lb==0: return 1.0
    if la==0 or  lb==0: return 0.0
    return min(la,lb)/max(la,lb)
def _char_ngrams(s, n=3):
    """Character n-grams as a set."""
    if not s or len(s) < n: return set()
    return {s[i:i+n] for i in range(len(s)-n+1)}
def _containment(s1, s2):
    """Max substring containment ratio."""
    if not s1 and not s2: return 1.0
    if not s1 or not s2:  return 0.0
    if s1 in s2: return len(s1) / len(s2)
    if s2 in s1: return len(s2) / len(s1)
    return 0.0


# ---- Phonetic helpers --------------------------------------------------------

def _soundex(s):
    """Compute Soundex code for the first meaningful word."""
    if not PHONETIC_AVAILABLE or not s:
        return ""
    # Take first alphabetic token
    words = re.findall(r'[a-zA-Z]+', s)
    if not words:
        return ""
    try:
        return jellyfish.soundex(words[0])
    except Exception:
        return ""

def _metaphone(s):
    """Compute Metaphone code for the first meaningful word."""
    if not PHONETIC_AVAILABLE or not s:
        return ""
    words = re.findall(r'[a-zA-Z]+', s)
    if not words:
        return ""
    try:
        return jellyfish.metaphone(words[0])
    except Exception:
        return ""

def _nysiis(s):
    """Compute NYSIIS code for the first meaningful word."""
    if not PHONETIC_AVAILABLE or not s:
        return ""
    words = re.findall(r'[a-zA-Z]+', s)
    if not words:
        return ""
    try:
        return jellyfish.nysiis(words[0])
    except Exception:
        return ""

def _soundex_all_tokens(s):
    """Soundex codes for all alphabetic tokens as a set."""
    if not PHONETIC_AVAILABLE or not s:
        return set()
    words = re.findall(r'[a-zA-Z]{2,}', s)
    codes = set()
    for w in words:
        try:
            codes.add(jellyfish.soundex(w))
        except Exception:
            pass
    return codes

def _extract_pin_codes(addr):
    """Extract PIN/ZIP codes: 5-6 digit numbers from address."""
    if not addr:
        return set()
    return set(re.findall(r'\b\d{5,6}\b', addr))

def _extract_city_tokens(addr):
    """Extract likely city tokens (alphabetic tokens of length >= 3, not common abbreviations)."""
    if not addr:
        return set()
    _skip = {'st', 'rd', 'ave', 'blvd', 'dr', 'ln', 'ct', 'pl', 'hwy', 'apt',
             'ste', 'bldg', 'fl', 'no', 'ngr', 'dist', 'sec', 'blk', 'col',
             'and', 'near', 'opp', 'behind', 'next', 'the', 'of', 'in', 'at'}
    tokens = re.findall(r'[a-zA-Z]{3,}', addr.lower())
    return {t for t in tokens if t not in _skip}


def compute_pair_features(n1, a1, n2, a2):
    """48 similarity features for one (S1, candidate) pair."""
    nt1, nt2 = _tokens(n1), _tokens(n2)
    at1, at2 = _tokens(a1), _tokens(a2)
    an1, an2 = _nums(a1),   _nums(a2)
    safe_n = (n1 and n2)
    safe_a = (a1 and a2)
    c1, c2 = f"{n1} {a1}", f"{n2} {a2}"
    feats = [
        # ---- name (19) ----
        1.0 - Levenshtein.normalized_distance(n1, n2) if safe_n else (1.0 if not n1 and not n2 else 0.0),
        JaroWinkler.similarity(n1, n2)                if safe_n else (1.0 if not n1 and not n2 else 0.0),
        fuzz.token_sort_ratio(n1, n2) / 100.0,
        fuzz.token_set_ratio(n1, n2)  / 100.0,
        fuzz.partial_ratio(n1, n2)    / 100.0,
        fuzz.ratio(n1, n2)            / 100.0,
        _jac(nt1, nt2),
        _ovl(nt1, nt2),
        _dice(nt1, nt2),
        len(nt1 & nt2) / max(len(nt1), 1) if nt1 else 0.0,
        len(nt1 & nt2) / max(len(nt2), 1) if nt2 else 0.0,
        1.0 if (nt1 and nt2 and min(nt1) == min(nt2)) else 0.0,
        _lr(n1, n2),
        _containment(n1, n2),
        _jac(_char_ngrams(n1, 3), _char_ngrams(n2, 3)),
        SequenceMatcher(None, n1, n2).ratio() if safe_n else (1.0 if not n1 and not n2 else 0.0),
        1.0 if (nt1 and nt2 and sorted(nt1)[0] == sorted(nt2)[0]) else 0.0,
        min(len(nt1), len(nt2)) / max(len(nt1), len(nt2), 1),
        1.0 if (safe_n and len(n1) >= 3 and len(n2) >= 3 and n1[:3] == n2[:3]) else 0.0,
        # ---- address (11) ----
        1.0 - Levenshtein.normalized_distance(a1, a2) if safe_a else (1.0 if not a1 and not a2 else 0.0),
        JaroWinkler.similarity(a1, a2)                if safe_a else (1.0 if not a1 and not a2 else 0.0),
        fuzz.token_sort_ratio(a1, a2) / 100.0,
        fuzz.token_set_ratio(a1, a2)  / 100.0,
        fuzz.partial_ratio(a1, a2)    / 100.0,
        _jac(at1, at2),
        _ovl(at1, at2),
        _dice(at1, at2),
        _lr(a1, a2),
        _jac(at1 - an1, at2 - an2),
        _containment(a1, a2),
        # ---- address numbers (5) ----
        _jac(an1, an2),
        _ovl(an1, an2),
        len(an1 & an2) / max(len(an1), 1) if an1 else (1.0 if not an2 else 0.5),
        1.0 if (an1 and an2 and sorted(an1)[0] == sorted(an2)[0]) else (1.0 if not an1 and not an2 else 0.0),
        abs(len(an1) - len(an2)),
        # ---- cross (5) ----
        fuzz.token_sort_ratio(c1, c2) / 100.0,
        fuzz.token_set_ratio(c1, c2)  / 100.0,
        _jac(nt1 | at1, nt2 | at2),
        abs(len(nt1) - len(nt2)),
        1.0 - Levenshtein.normalized_distance(c1, c2) if (c1.strip() and c2.strip()) else (1.0 if not c1.strip() and not c2.strip() else 0.0),
    ]

    # ---- phonetic features (8) — ONLY included when jellyfish is installed ----
    if PHONETIC_AVAILABLE:
        sx1, sx2 = _soundex(n1), _soundex(n2)
        mp1, mp2 = _metaphone(n1), _metaphone(n2)
        ny1, ny2 = _nysiis(n1), _nysiis(n2)
        sxa1, sxa2 = _soundex_all_tokens(n1), _soundex_all_tokens(n2)
        pin1, pin2 = _extract_pin_codes(a1), _extract_pin_codes(a2)
        city1, city2 = _extract_city_tokens(a1), _extract_city_tokens(a2)

        feats.extend([
            # Soundex exact match on first word
            1.0 if (sx1 and sx2 and sx1 == sx2) else 0.0,
            # Metaphone exact match on first word
            1.0 if (mp1 and mp2 and mp1 == mp2) else 0.0,
            # Soundex Jaro-Winkler (phonetic fuzzy match)
            JaroWinkler.similarity(sx1, sx2) if (sx1 and sx2) else 0.0,
            # NYSIIS match on first word
            1.0 if (ny1 and ny2 and ny1 == ny2) else 0.0,
            # Soundex Jaccard across all name tokens
            _jac(sxa1, sxa2),
            # PIN/ZIP code exact match
            1.0 if (pin1 and pin2 and pin1 & pin2) else (1.0 if not pin1 and not pin2 else 0.0),
            # City token overlap
            _ovl(city1, city2),
            # Address numeric token count match (same number of numbers = structural similarity)
            1.0 if len(an1) == len(an2) else 1.0 / (1.0 + abs(len(an1) - len(an2))),
        ])

    return feats

# Feature count is DYNAMIC: 40 base features + 8 phonetic features if jellyfish installed
N_FEATURES = 48 if PHONETIC_AVAILABLE else 40
_BASE_FEATURE_NAMES = [
    "name_lev","name_jw","name_tsort","name_tset","name_partial","name_ratio",
    "name_jac","name_ovl","name_dice","name_cont12","name_cont21",
    "name_first","name_lr",
    "name_contain","name_char3_jac","name_lcs","name_first_sorted","name_tok_ratio",
    "name_prefix3",
    "addr_lev","addr_jw","addr_tsort","addr_tset","addr_partial",
    "addr_jac","addr_ovl","addr_dice","addr_lr",
    "addr_nonum_jac","addr_contain",
    "anum_jac","anum_ovl","anum_match12","anum_first","anum_cnt_diff",
    "comb_tsort","comb_tset","comb_jac","name_tok_diff","comb_lev",
]
_PHONETIC_FEATURE_NAMES = [
    "phon_soundex_match","phon_metaphone_match","phon_soundex_jw",
    "phon_nysiis_match","phon_soundex_jac",
    "addr_pin_match","addr_city_ovl","addr_num_cnt_match",
]
FEATURE_NAMES = _BASE_FEATURE_NAMES + (_PHONETIC_FEATURE_NAMES if PHONETIC_AVAILABLE else [])


# ---- parallel feature workers ------------------------------------------------

def _compute_features_chunk(chunk):
    """Worker: compute features for a list of (n1,a1,n2,a2) tuples.
       Returns np.array of shape (len(chunk), N_FEATURES)."""
    out = np.empty((len(chunk), N_FEATURES), dtype=np.float32)
    for i, (n1, a1, n2, a2) in enumerate(chunk):
        out[i] = compute_pair_features(n1, a1, n2, a2)
    return out


def parallel_compute_features_ordered(pairs_data: list, desc="Features") -> np.ndarray:
    """Same as above but preserves order with granular progress reporting."""
    n = len(pairs_data)
    if n == 0:
        return np.empty((0, N_FEATURES), dtype=np.float32)

    chunk_size = max(min(2_500, n // (N_WORKERS * 2)), 500)
    chunks = [(i, pairs_data[i:i+chunk_size]) for i in range(0, n, chunk_size)]

    print(f"[Feat] {n:,} pairs -> {len(chunks)} chunks across {N_WORKERS} workers")
    t0 = time.time()

    ordered_results = [None] * len(chunks)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        future_to_idx = {}
        for ci, (start, chunk) in enumerate(chunks):
            fut = pool.submit(_compute_features_chunk, chunk)
            future_to_idx[fut] = ci
        for fut in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc=desc, unit="chunk"):
            ci = future_to_idx[fut]
            ordered_results[ci] = fut.result()

    X = np.vstack(ordered_results) if ordered_results else np.empty((0, N_FEATURES), dtype=np.float32)
    print(f"[Feat] {X.shape} in {time.time()-t0:.1f}s")
    return X
