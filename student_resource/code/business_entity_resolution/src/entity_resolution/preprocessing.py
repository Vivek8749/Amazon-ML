"""Vectorised text normalisation for business names and addresses."""
import re

import pandas as pd
from anyascii import anyascii

from .config import TRANSLITERATE

# Bumped whenever preprocessing output changes, so stale parquet caches are
# not reused ("" keeps the fingerprint of caches built before versioning).
PREPROCESS_VERSION = "translit1" if TRANSLITERATE else ""


def transliterate(s: pd.Series) -> pd.Series:
    """Map non-ASCII strings to ASCII (Devanagari/Kannada -> Latin, é -> e).

    Only non-ASCII rows go through anyascii, so mostly-Latin data stays cheap.
    """
    s = s.fillna("")
    non_ascii = ~s.map(str.isascii).astype(bool)
    if non_ascii.any():
        s = s.copy()
        s[non_ascii] = s[non_ascii].map(anyascii)
    return s

# ===== PRE-COMPILED MEGA-REGEX (compiled ONCE at import time) =================
# Instead of 40+ separate .str.replace() calls (each re-iterating the Series),
# we compile 4 mega-patterns and do 4 single-pass substitutions.

# --- Name: dotted legal abbreviations (order: longest first) ---
_NAME_DOTTED_MAP = {
    "s.a.r.l.": "sarl", "s.a.r.l": "sarl",
    "s.a.s.": "sas",   "s.a.s": "sas",
    "s.c.i.": "sci",   "s.c.i": "sci",
    "l.l.c.": "llc",   "l.l.c": "llc",
    "l.l.p.": "llp",   "l.l.p": "llp",
    "p.l.c.": "plc",   "p.l.c": "plc",
    "l.p.": "lp",      "l.p": "lp",
    "n.a.": "na",      "n.a": "na",
}
_RE_NAME_DOTTED = re.compile(
    "|".join(re.escape(k) for k in sorted(_NAME_DOTTED_MAP, key=len, reverse=True))
)
def _repl_name_dotted(m): return _NAME_DOTTED_MAP[m.group()]

# --- Name: full words + trailing-dot suffixes (single pass) ---
_NAME_WORD_MAP = {
    "incorporated": "inc", "corporation": "corp", "limited": "ltd",
    "company": "co", "private": "pvt",
    "inc.": "inc", "corp.": "corp", "ltd.": "ltd", "pvt.": "pvt",
}
_RE_NAME_WORDS = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_NAME_WORD_MAP, key=len, reverse=True)) + r")\b"
)
def _repl_name_words(m): return _NAME_WORD_MAP[m.group()]

# --- Address: full words + trailing-dot abbreviations (single pass) ---
_ADDR_MAP = {
    # Full words
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "highway": "hwy", "parkway": "pkwy",
    "terrace": "ter", "apartment": "apt", "suite": "ste", "building": "bldg",
    "floor": "fl", "district": "dist", "nagar": "ngr", "sector": "sec",
    "colony": "col",
    # Trailing-dot abbreviations
    "st.": "st", "rd.": "rd", "ave.": "ave", "blvd.": "blvd",
    "dr.": "dr", "apt.": "apt", "ste.": "ste", "bldg.": "bldg",
    "fl.": "fl", "no.": "no",
}
_RE_ADDR = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ADDR_MAP, key=len, reverse=True)) + r")\b"
)
def _repl_addr(m): return _ADDR_MAP[m.group()]

# Common patterns compiled once
_RE_BRACKETS  = re.compile(r"[\[\](){}]")
_RE_DOMAIN    = re.compile(r"\.(com|org|net|in|fr|co\.in)$")
_RE_CO_DOT    = re.compile(r"\bco\.(?=\s|$)")
_RE_MULTI_WS  = re.compile(r"\s+")
_RE_DIGITS    = re.compile(r"\d+")


# ===== FAST VECTORISED PREPROCESSING ==========================================

def fast_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised string cleaning with entity-resolution normalisation.

    Uses pre-compiled mega-regex patterns for single-pass substitutions
    instead of 40+ chained .str.replace() calls.

    Applies:
      0. Transliteration to ASCII (if TRANSLITERATE)
      1. Basic cleaning (lowercase, bracket removal, &/+ expansion)
      2. Dotted legal abbreviation normalisation (L.L.C. → llc, S.A.R.L. → sarl)
      3. Full-word legal suffix normalisation (Corporation → corp, Private → pvt)
      4. Address abbreviation normalisation (Street → st, Boulevard → blvd)
      5. Whitespace collapse + strip
    """
    df = df.copy()
    raw_name, raw_addr = df["business_name"].fillna(""), df["business_address"].fillna("")
    if TRANSLITERATE:
        raw_name, raw_addr = transliterate(raw_name), transliterate(raw_addr)

    # --- Name cleaning (4 passes instead of 22) ---
    name = (raw_name
            .str.lower()
            .str.replace(_RE_BRACKETS, " ", regex=True)
            .str.replace("&", " and ", regex=False)
            .str.replace("+", " and ", regex=False))
    # Pass 1: dotted abbreviations (s.a.r.l. → sarl, l.l.c. → llc, etc.)
    name = name.str.replace(_RE_NAME_DOTTED, _repl_name_dotted, regex=True)
    # Pass 2: full-word + trailing-dot suffixes (corporation → corp, inc. → inc)
    name = name.str.replace(_RE_NAME_WORDS, _repl_name_words, regex=True)
    # Pass 3: domain suffixes, co. edge case
    name = name.str.replace(_RE_DOMAIN, "", regex=True)
    name = name.str.replace(_RE_CO_DOT, "co", regex=True)
    df["name_clean"] = name.str.replace(_RE_MULTI_WS, " ", regex=True).str.strip()

    # --- Address cleaning (2 passes instead of 27) ---
    addr = (raw_addr
            .str.lower()
            .str.replace(_RE_BRACKETS, " ", regex=True)
            .str.replace("&", " and ", regex=False)
            .str.replace("+", " and ", regex=False))
    # Single pass: full words + trailing-dot abbreviations
    addr = addr.str.replace(_RE_ADDR, _repl_addr, regex=True)
    df["addr_clean"] = addr.str.replace(_RE_MULTI_WS, " ", regex=True).str.strip()

    df["country_norm"] = df["country"].fillna("").str.lower().str.strip()
    df["combined"]     = df["name_clean"] + " " + df["addr_clean"]
    # Numeric tokens from address (for blocking key) — vectorised extraction
    df["addr_nums_str"] = (df["addr_clean"]
                           .str.findall(_RE_DIGITS)
                           .apply(lambda xs: " ".join(sorted(set(xs))[:5])
                                  if isinstance(xs, list) else ""))
    # Sorted-token blocking key: sort name tokens, take first 3
    df["name_sorted_3tok"] = (df["name_clean"]
                              .str.split()
                              .apply(lambda xs: " ".join(sorted(xs)[:3])
                                     if isinstance(xs, list) and len(xs) >= 2 else ""))
    return df
