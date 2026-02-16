"""Utilities for multiple-choice (MC4) question generation.

Handles distractor selection, option shuffling, and prompt formatting.
"""

import random


def generate_distractors(
    correct_value: float,
    other_values: list[float],
    n: int = 3,
    min_spacing_pct: float = 0.15,
    rng: random.Random | None = None,
) -> list[float]:
    """Generate plausible MC4 distractors for a numeric value.

    Strategy:
    1. Use values of other elements in the image (adjacent bars, neighboring points)
    2. Fill remaining slots with offset values (±15-50% of correct)

    Args:
        correct_value: The correct answer.
        other_values: Values of other visible elements (e.g., heights of other bars).
        n: Number of distractors to generate.
        min_spacing_pct: Minimum spacing between any two options as fraction of range.
        rng: Random instance for reproducibility.

    Returns:
        List of n distractor values (does not include correct_value).
    """
    if rng is None:
        rng = random.Random()

    # Compute value range for spacing checks
    all_candidates = [correct_value] + list(other_values)
    val_range = max(all_candidates) - min(all_candidates) if len(all_candidates) > 1 else abs(correct_value) or 10
    min_gap = val_range * min_spacing_pct

    def too_close(v: float, existing: list[float]) -> bool:
        return any(abs(v - e) < max(min_gap, 1) for e in existing)

    selected: list[float] = []

    # Pick from other_values first (most plausible distractors)
    candidates = [v for v in other_values if v != correct_value]
    rng.shuffle(candidates)
    for v in candidates:
        if len(selected) >= n:
            break
        if not too_close(v, [correct_value] + selected):
            selected.append(v)

    # Fill remaining with offset values
    offsets = [0.2, -0.2, 0.35, -0.35, 0.5, -0.5, 0.15, -0.15]
    rng.shuffle(offsets)
    for offset in offsets:
        if len(selected) >= n:
            break
        v = correct_value * (1 + offset) if correct_value != 0 else offset * 10
        v = round(v)
        if v != correct_value and not too_close(v, [correct_value] + selected):
            selected.append(v)

    # Last resort: sequential offsets
    step = max(int(min_gap), 2)
    for delta in range(1, 20):
        if len(selected) >= n:
            break
        for sign in [1, -1]:
            v = correct_value + sign * delta * step
            if v != correct_value and not too_close(v, [correct_value] + selected):
                selected.append(v)
                if len(selected) >= n:
                    break

    return selected[:n]


def format_mc4_prompt(
    question: str,
    correct_value,
    distractors: list,
    value_format: str = "{}",
    rng: random.Random | None = None,
) -> tuple[str, str]:
    """Build an MC4 prompt with shuffled options.

    Args:
        question: The question text (without options).
        correct_value: The correct answer value.
        distractors: List of distractor values.
        value_format: Format string for displaying values (e.g., "{}" or "${:,.0f}").
        rng: Random instance for reproducibility.

    Returns:
        (full_prompt, correct_letter) where correct_letter is "A", "B", "C", or "D".
    """
    if rng is None:
        rng = random.Random()

    options = [(correct_value, True)] + [(d, False) for d in distractors[:3]]
    rng.shuffle(options)

    letters = ["A", "B", "C", "D"]
    correct_letter = None
    lines = [question]

    for letter, (value, is_correct) in zip(letters, options):
        lines.append(f"({letter}) {value_format.format(value)}")
        if is_correct:
            correct_letter = letter

    lines.append("Answer with only the letter.")
    return "\n".join(lines), correct_letter
