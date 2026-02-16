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
    # Try {X} format first
    m = re.search(r"\{([A-Za-z])\}", response)
    if m:
        return m.group(1).upper()
    # Try standalone letter
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


@register_parser("csv_letters")
def parse_csv_letters(response: str) -> Optional[str]:
    """Extract sorted comma-separated uppercase letters."""
    # Find all single letters that appear to be option labels
    letters = re.findall(r"\b([A-Za-z])\b", response)
    if not letters:
        return None
    # Deduplicate and sort
    unique = sorted(set(l.upper() for l in letters))
    return ",".join(unique)
