"""
Text preprocessing utilities for entity resolution.
Handles name normalisation, address canonicalisation, and tokenisation.
"""
import re
import unicodedata
from functools import lru_cache

# ─── Legal-suffix canonicalisation ────────────────────────────────────────────
LEGAL_SUFFIXES = {
    # English
    "llc": "llc", "l.l.c.": "llc", "l.l.c": "llc",
    "inc": "inc", "inc.": "inc", "incorporated": "inc",
    "corp": "corp", "corp.": "corp", "corporation": "corp",
    "ltd": "ltd", "ltd.": "ltd", "limited": "ltd",
    "co": "co", "co.": "co", "company": "co",
    "llp": "llp", "l.l.p.": "llp", "l.l.p": "llp",
    "plc": "plc", "p.l.c.": "plc",
    "lp": "lp", "l.p.": "lp",
    "pllc": "pllc",
    "na": "na", "n.a.": "na",
    # Indian
    "pvt": "pvt", "pvt.": "pvt", "private": "pvt",
    "nidhi": "nidhi",
    # French
    "sarl": "sarl", "s.a.r.l.": "sarl", "s.a.r.l": "sarl",
    "sas": "sas", "s.a.s.": "sas", "s.a.s": "sas",
    "sa": "sa", "s.a.": "sa",
    "eurl": "eurl",
    "sci": "sci", "s.c.i.": "sci",
    "scp": "scp",
    "snc": "snc",
    "gie": "gie",
}

# ─── Address abbreviations ────────────────────────────────────────────────────
ADDRESS_ABBREVS = {
    "street": "st", "st.": "st", "st": "st",
    "road": "rd", "rd.": "rd", "rd": "rd",
    "avenue": "ave", "ave.": "ave", "av": "ave", "av.": "ave",
    "boulevard": "blvd", "blvd.": "blvd",
    "drive": "dr", "dr.": "dr",
    "lane": "ln", "ln.": "ln",
    "court": "ct", "ct.": "ct",
    "place": "pl", "pl.": "pl",
    "circle": "cir", "cir.": "cir",
    "highway": "hwy", "hwy.": "hwy",
    "parkway": "pkwy", "pkwy.": "pkwy",
    "terrace": "ter", "ter.": "ter",
    "trail": "trl", "trl.": "trl",
    "way": "way",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw",
    "apartment": "apt", "apt.": "apt", "apt": "apt",
    "suite": "ste", "ste.": "ste",
    "floor": "fl", "flr": "fl", "fl.": "fl",
    "building": "bldg", "bldg.": "bldg",
    "unit": "unit",
    "number": "no", "no.": "no",
    # Indian
    "nagar": "ngr", "ngr": "ngr",
    "district": "dist", "dist.": "dist",
    "colony": "colony",
    "sector": "sec", "sec.": "sec",
    "block": "blk", "blk.": "blk",
    # French
    "rue": "rue",
    "avenue": "ave",
    "boulevard": "blvd",
    "place": "pl",
    "impasse": "imp",
    "chemin": "ch",
    "allée": "all",
    "passage": "pass",
}

# Indian state abbreviations
INDIAN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as",
    "bihar": "br", "chhattisgarh": "cg", "goa": "ga",
    "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "punjab": "pb", "rajasthan": "rj",
    "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "chandigarh": "ch", "puducherry": "py",
}

# US state name->abbreviation (only common ones — the rest are already 2-letter)
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct",
    "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}

# ─── Core normalisation functions ─────────────────────────────────────────────

_BRACKET_RE = re.compile(r"[\[\]\(\)\{\}]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalise_unicode(text: str) -> str:
    """NFKD normalise, strip accents for Latin text but keep Devanagari etc."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    # Keep characters that are either non-Mark or are part of non-Latin scripts
    chars = []
    for ch in nfkd:
        cat = unicodedata.category(ch)
        if cat.startswith("M"):
            # Combining mark — skip only for Latin base
            # Simple heuristic: if the previous char was Latin, skip
            if chars and ord(chars[-1]) < 0x0300:
                continue
        chars.append(ch)
    return "".join(chars)


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip brackets, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = _BRACKET_RE.sub(" ", text)
    text = text.replace("&", " and ")
    text = text.replace("+", " and ")
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def normalise_name(name: str) -> str:
    """Normalise a business name for comparison."""
    name = clean_text(name)
    name = normalise_unicode(name)

    # Remove common web suffixes
    name = re.sub(r"\.com$|\.org$|\.net$|\.in$|\.fr$|\.co\.in$", "", name)

    # Normalise legal suffixes
    tokens = name.split()
    normalised = []
    for tok in tokens:
        clean_tok = _PUNCT_RE.sub("", tok)
        if clean_tok in LEGAL_SUFFIXES:
            normalised.append(LEGAL_SUFFIXES[clean_tok])
        else:
            normalised.append(tok)

    return " ".join(normalised)


def normalise_address(address: str, country: str = "") -> str:
    """Normalise an address for comparison."""
    address = clean_text(address)
    address = normalise_unicode(address)

    # Normalise address abbreviations
    tokens = address.split()
    normalised = []
    for tok in tokens:
        clean_tok = _PUNCT_RE.sub("", tok)
        if clean_tok in ADDRESS_ABBREVS:
            normalised.append(ADDRESS_ABBREVS[clean_tok])
        elif country.lower() == "india" and clean_tok in INDIAN_STATES:
            normalised.append(INDIAN_STATES[clean_tok])
        elif country.lower() == "us" and clean_tok in US_STATES:
            normalised.append(US_STATES[clean_tok])
        else:
            normalised.append(tok)

    return " ".join(normalised)


def extract_tokens(text: str) -> set:
    """Extract a set of alphanumeric tokens from normalised text."""
    if not text:
        return set()
    return set(re.findall(r"\w+", text.lower(), re.UNICODE))


def extract_numeric_tokens(text: str) -> set:
    """Extract numeric substrings (addresses, PIN codes, etc.)."""
    if not text:
        return set()
    return set(re.findall(r"\d+", text))


def extract_alpha_tokens(text: str) -> set:
    """Extract purely alphabetic tokens (words)."""
    if not text:
        return set()
    return set(re.findall(r"[a-zA-Z\u0900-\u097F\u00C0-\u024F]+", text))


def get_name_key(name: str) -> str:
    """
    Generate a blocking key from the business name.
    Uses first 3 consonants + first numeric token if present.
    """
    name = normalise_name(name)
    # Extract consonants
    consonants = re.findall(r"[^aeiou\s\d\W]", name, re.UNICODE)
    key = "".join(consonants[:4])
    # Add first number if present
    nums = re.findall(r"\d+", name)
    if nums:
        key += "_" + nums[0]
    return key


def get_address_key(address: str, country: str = "") -> str:
    """
    Generate a blocking key from the address.
    Uses numeric components (street numbers, PIN codes).
    """
    address = normalise_address(address, country)
    nums = sorted(set(re.findall(r"\d+", address)))
    return "_".join(nums[:3]) if nums else ""


def create_combined_text(name: str, address: str, country: str = "") -> str:
    """Create a combined normalised text for TF-IDF vectorisation."""
    n = normalise_name(name)
    a = normalise_address(address, country)
    return f"{n} {a}".strip()
