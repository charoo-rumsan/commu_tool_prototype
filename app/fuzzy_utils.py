# app/fuzzy_utils.py
from difflib import SequenceMatcher
from typing import Optional

def fuzzy_ratio(a: Optional[str], b: Optional[str]) -> float:
    """Return fuzzy similarity between two strings (0–100)."""
    # Handle None values
    if a is None or b is None:
        return 0.0
    # Convert to strings and handle empty strings
    a_str = str(a).lower() if a else ""
    b_str = str(b).lower() if b else ""
    if not a_str or not b_str:
        return 0.0
    return SequenceMatcher(None, a_str, b_str).ratio() * 100
