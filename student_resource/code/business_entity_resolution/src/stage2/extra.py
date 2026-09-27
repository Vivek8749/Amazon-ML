"""Hard-negative features: house numbers and name tokens that differ.

The test pool holds sibling branches ("Compass East | 193 Woodfield Dr" vs
"Compass | 182 Woodfield Dr") and unrelated businesses at the same street, which
the base string features score as matches.
"""
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

_NUM = re.compile(r"\d+")
LEGAL = {
    "inc", "llc", "l.l.c.", "ltd", "limited", "pvt", "private", "corp", "corporation", "co", "company",
    "llp", "lp", "plc", "sa", "sas", "sasu", "sarl", "eurl", "sci", "s.a.", "s.a.s.", "the", "and", "of",
    "&", "group", "(group)", "[group]", "dba", "dba:", "aka", "a/k/a", "formerly", "known", "as", "fka",
    "sri", "shri", "dr", "m/s", "ms", "mr", "de", "des", "du", "la", "le", "les", "et",
}
EXTRA_NAMES = [
    "hn_first_eq", "hn_first_absdiff", "hn_first_substr", "hn_first_lev", "hn_any_eq", "hn_all_s1_in_c",
    "hn_both", "hn_first_ratio",
    "nm_extra_c", "nm_extra_s1", "nm_extra_c_fuzzy", "nm_extra_s1_fuzzy", "nm_core_tset", "nm_core_ratio",
    "nm_core_eq", "nm_is_web", "nm_c_len_tok", "nm_formerly",
    "ad_street_tset", "ad_street_eq",
]


def _tok(s):
    return [t for t in re.split(r"[\s,.\-()\[\]/]+", s.lower()) if t] if s else []


def _core(tokens):
    return [t for t in tokens if t not in LEGAL]


def _fuzzy_unmatched(a, b):
    """tokens of a with no fuzzy (>=80) counterpart in b"""
    n = 0
    for t in a:
        if t in b:
            continue
        if any(fuzz.ratio(t, u) >= 80 for u in b):
            continue
        n += 1
    return n


def _street(addr):
    """address minus numbers, first comma part (street)"""
    if not addr:
        return ""
    part = addr.split(",")
    # choose the part holding the first number, else the first part
    for p in part:
        if _NUM.search(p):
            return _NUM.sub(" ", p).strip()
    return part[0].strip()


def extra_features(n1, a1, n2, a2):
    h1, h2 = _NUM.findall(a1 or ""), _NUM.findall(a2 or "")
    f = []
    if h1 and h2:
        x, y = h1[0], h2[0]
        xi, yi = int(x[:9]), int(y[:9])
        f += [float(x == y), float(min(abs(xi - yi), 10000)), float(x in y or y in x),
              float(Levenshtein.distance(x, y)), float(bool(set(h1) & set(h2))),
              float(set(h1) <= set(h2)), 1.0, min(xi, yi) / max(xi, yi, 1)]
    else:
        f += [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0 if (h1 or h2) else -1.0, -1.0]
    t1, t2 = _tok(n1), _tok(n2)
    c1, c2 = _core(t1), _core(t2)
    s1, s2 = set(c1), set(c2)
    j1, j2 = " ".join(c1), " ".join(c2)
    f += [float(len(s2 - s1)), float(len(s1 - s2)),
          float(_fuzzy_unmatched(c2, s1)), float(_fuzzy_unmatched(c1, s2)),
          fuzz.token_set_ratio(j1, j2) / 100.0, fuzz.ratio(j1, j2) / 100.0,
          float(j1 == j2 and j1 != ""),
          float(bool(re.search(r"\.(com|in|fr|net|org)\b|www", n2 or ""))),
          float(len(t2)),
          float(bool(re.search(r"formerly|a/k/a|aka|dba|fka|known as", n2 or "")))]
    st1, st2 = _street(a1), _street(a2)
    f += [fuzz.token_set_ratio(st1, st2) / 100.0 if st1 and st2 else -1.0,
          float(st1 == st2 and st1 != "")]
    return f


def _chunk(chunk):
    out = np.empty((len(chunk), len(EXTRA_NAMES)), np.float32)
    for i, (n1, a1, n2, a2) in enumerate(chunk):
        out[i] = extra_features(n1, a1, n2, a2)
    return out


def compute_extra(data, workers=12):
    if not data:
        return np.zeros((0, len(EXTRA_NAMES)), np.float32)
    step = max(5000, len(data) // (workers * 8) + 1)
    with ProcessPoolExecutor(workers) as ex:
        return np.vstack(list(ex.map(_chunk, [data[i:i + step] for i in range(0, len(data), step)])))
