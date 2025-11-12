# app/fuzzy_utils.py
from difflib import SequenceMatcher

def fuzzy_ratio(a: str, b: str) -> float:
    """Return fuzzy similarity between two strings (0–100)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100
