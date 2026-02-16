"""Response parsers for VLM evaluation.

Each parser takes a raw model response string and extracts a structured answer.
"""

import re
from typing import Optional

PARSER_REGISTRY: dict[str, callable] = {}


def register_parser(name: str):
    def decorator(fn):
        PARSER_REGISTRY[name] = fn
        return fn
    return decorator


@register_parser("integer")
def parse_integer(response: str) -> Optional[str]:
    """Extract an integer from response. Handles {N}, plain numbers, or spelled out."""
    # Try {N} format first
    m = re.search(r"\{(\d+)\}", response)
    if m:
        return m.group(1)
    # Try plain number (last number in response as fallback)
    numbers = re.findall(r"\b(\d+)\b", response)
    if numbers:
        return numbers[-1]
    return None


@register_parser("yes_no")
def parse_yes_no(response: str) -> Optional[str]:
    """Extract Yes or No from response."""
    lower = response.lower().strip()
    if lower.startswith("yes") or "yes" in lower.split()[:3]:
        return "Yes"
    if lower.startswith("no") or "no" in lower.split()[:3]:
        return "No"
    # Check anywhere in response
    if re.search(r"\byes\b", lower):
        return "Yes"
    if re.search(r"\bno\b", lower):
        return "No"
    return None


@register_parser("letter")
def parse_letter(response: str) -> Optional[str]:
    """Extract a single uppercase letter from response."""
    # Try {X} format first (most reliable)
    m = re.search(r"\{([A-Za-z])\}", response)
    if m:
        return m.group(1).upper()
    # Try "answer is X" / "is X" / "box X" / "reach X" patterns (last match wins)
    matches = re.findall(r"(?:answer\s+is|is|box|reach)\s+([A-Za-z])\b", response, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    # Search from end of response for a standalone uppercase letter (skip I/A)
    # This avoids grabbing "I" from "I think..." or "A" from "A box..."
    words = response.split()
    for word in reversed(words):
        if re.fullmatch(r"[A-Za-z]\.?", word) and word.rstrip(".").upper() not in ("I", "A"):
            return word.rstrip(".").upper()
    # Last resort: any standalone letter (including I/A)
    m = re.search(r"\b([A-Za-z])\b", response)
    if m:
        return m.group(1).upper()
    return None


@register_parser("row_col")
def parse_row_col(response: str) -> Optional[str]:
    """Extract rows and columns, return as 'R,C' string."""
    # Try rows=N or rows={N} columns=M or columns={M} format
    row_m = re.search(r"rows?\s*[=:]\s*\{?(\d+)\}?", response, re.IGNORECASE)
    col_m = re.search(r"col(?:umn)?s?\s*[=:]\s*\{?(\d+)\}?", response, re.IGNORECASE)
    if row_m and col_m:
        return f"{row_m.group(1)},{col_m.group(1)}"
    # Try (N,M) or NxM format
    m = re.search(r"(\d+)\s*[x×,]\s*(\d+)", response)
    if m:
        return f"{m.group(1)},{m.group(2)}"
    return None


@register_parser("mc4")
def parse_mc4(response: str) -> Optional[str]:
    """Extract a multiple-choice letter A/B/C/D from response."""
    # Try {A} format
    m = re.search(r"\{([A-Da-d])\}", response)
    if m:
        return m.group(1).upper()
    # Try (A) format
    m = re.search(r"\(([A-Da-d])\)", response)
    if m:
        return m.group(1).upper()
    # Try "answer is A" / "answer: A" patterns
    m = re.search(r"(?:answer\s*(?:is|:)\s*)([A-Da-d])\b", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Try **A** bold markdown format
    m = re.search(r"\*\*([A-Da-d])\*\*", response)
    if m:
        return m.group(1).upper()
    # Try standalone A/B/C/D (search from end to get final answer)
    words = response.split()
    for word in reversed(words):
        clean = word.strip(".,;:!?()[]{}\"'*")
        if re.fullmatch(r"[A-Da-d]", clean):
            return clean.upper()
    return None


@register_parser("exact_string")
def parse_exact_string(response: str) -> Optional[str]:
    """Extract an exact string answer from response.

    Tries structured formats first, then strips common preamble.
    """
    if not response or not response.strip():
        return None
    # Try {answer} format (curly brackets)
    m = re.search(r"\{([^}]+)\}", response)
    if m:
        return m.group(1).strip()
    # Try "answer" or 'answer' (quoted)
    m = re.search(r'["\u201c]([^"\u201d]+)["\u201d]', response)
    if m:
        return m.group(1).strip()
    # Strip common preamble patterns and return the rest
    stripped = response.strip()
    for prefix in [
        r"(?:the\s+)?(?:answer|value|text|number|word)\s+is\s*:?\s*",
        r"(?:it\s+(?:says?|reads?)\s*:?\s*)",
    ]:
        m = re.match(prefix, stripped, re.IGNORECASE)
        if m:
            return stripped[m.end():].strip().rstrip(".")
    # If response is short (likely just the answer), return as-is
    if len(stripped.split()) <= 5:
        return stripped.rstrip(".")
    # For longer responses, return first line stripped
    first_line = stripped.split("\n")[0].strip()
    return first_line.rstrip(".")


@register_parser("csv_words")
def parse_csv_words(response: str) -> Optional[str]:
    """Extract sorted comma-separated words from response.

    For tasks where the answer is a set of word names (not single letters).
    Returns "" for explicitly empty answers like {}, {None}, {N/A}.
    """
    # Try {word1, word2} or {} format first
    m = re.search(r"\{([^}]*)\}", response)
    if m:
        inner = m.group(1).strip()
        # Treat {}, {None}, {none}, {N/A} as empty set
        if not inner or inner.lower() in ("none", "n/a", "empty", "no items"):
            return ""
        words = [w.strip() for w in inner.split(",") if w.strip()]
        if words:
            return ",".join(sorted(words, key=str.lower))
    # Try comma-separated list in the response
    # Look for a line/section with comma-separated capitalized words
    m = re.search(r"(?:^|:)\s*([A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+)+)", response, re.MULTILINE)
    if m:
        words = [w.strip() for w in m.group(1).split(",") if w.strip()]
        if words:
            return ",".join(sorted(words, key=str.lower))
    return None


@register_parser("csv_letters")
def parse_csv_letters(response: str) -> Optional[str]:
    """Extract sorted comma-separated uppercase letters."""
    # Try {A, C, E} or {A,C,E} format first
    m = re.search(r"\{([A-Za-z](?:\s*,\s*[A-Za-z])*)\}", response)
    if m:
        letters = re.findall(r"[A-Za-z]", m.group(1))
        return ",".join(sorted(set(l.upper() for l in letters)))
    # Try comma-separated letter list pattern: A, C, E or A,C,E
    m = re.search(r"\b([A-Za-z]\s*(?:,\s*[A-Za-z]\s*)+)\b", response)
    if m:
        letters = re.findall(r"[A-Za-z]", m.group(1))
        if letters:
            return ",".join(sorted(set(l.upper() for l in letters)))
    # Last resort: standalone uppercase option-like letters (skip common words)
    skip = {"I", "A", "IN", "OR", "TO", "IS", "IT", "AS", "AT", "IF", "OF"}
    letters = []
    for word in response.split():
        clean = word.strip(".,;:!?()[]")
        if re.fullmatch(r"[A-Za-z]", clean) and clean.upper() not in skip:
            letters.append(clean.upper())
    if letters:
        return ",".join(sorted(set(letters)))
    return None


@register_parser("csv_cell_labels")
def parse_csv_cell_labels(response: str) -> Optional[str]:
    """Extract sorted comma-separated cell labels like A1, B2, C3."""
    # Try {A1, B2, C3} format first
    m = re.search(r"\{([^}]+)\}", response)
    if m:
        labels = re.findall(r"[A-Za-z]\d+", m.group(1))
        if labels:
            return ",".join(sorted(set(l.upper() for l in labels)))
    # Try comma-separated list outside braces
    labels = re.findall(r"\b([A-Za-z]\d+)\b", response)
    if labels:
        return ",".join(sorted(set(l.upper() for l in labels)))
    return None
