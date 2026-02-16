"""Scoring functions for VLM evaluation.

Each scorer takes (parsed_answer, ground_truth) and returns a dict with
at minimum {correct: bool, score: float}.
"""

from typing import Optional

SCORER_REGISTRY: dict[str, callable] = {}


def register_scorer(name: str):
    def decorator(fn):
        SCORER_REGISTRY[name] = fn
        return fn
    return decorator


@register_scorer("exact_match")
def score_exact_match(parsed: Optional[str], ground_truth: str) -> dict:
    """Case-insensitive string match."""
    if parsed is None:
        return {"correct": False, "score": 0.0}
    correct = parsed.strip().lower() == ground_truth.strip().lower()
    return {"correct": correct, "score": 1.0 if correct else 0.0}


@register_scorer("set_member")
def score_set_member(parsed: Optional[str], ground_truth: str) -> dict:
    """Check if parsed answer is one of the valid answers (comma-separated GT)."""
    if parsed is None:
        return {"correct": False, "score": 0.0}
    valid = {x.strip().upper() for x in ground_truth.split(",") if x.strip()}
    correct = parsed.strip().upper() in valid
    return {"correct": correct, "score": 1.0 if correct else 0.0}


@register_scorer("integer_distance")
def score_integer_distance(parsed: Optional[str], ground_truth: str) -> dict:
    """Exact match + signed error (positive = overcount)."""
    if parsed is None:
        return {"correct": False, "score": 0.0, "error": None, "abs_error": None}
    try:
        pred = int(parsed)
        gt = int(ground_truth)
    except (ValueError, TypeError):
        return {"correct": False, "score": 0.0, "error": None, "abs_error": None}
    error = pred - gt  # positive = overcount
    correct = error == 0
    abs_error = abs(error)
    # Score: 1.0 for exact, decays with distance
    score = 1.0 / (1.0 + abs_error)
    return {"correct": correct, "score": score, "error": error, "abs_error": abs_error}


@register_scorer("row_col")
def score_row_col(parsed: Optional[str], ground_truth: str) -> dict:
    """Score rows and columns independently."""
    if parsed is None:
        return {"correct": False, "score": 0.0, "row_correct": False, "col_correct": False}
    try:
        pred_r, pred_c = parsed.split(",")
        gt_r, gt_c = ground_truth.split(",")
        row_correct = int(pred_r) == int(gt_r)
        col_correct = int(pred_c) == int(gt_c)
    except (ValueError, TypeError):
        return {"correct": False, "score": 0.0, "row_correct": False, "col_correct": False}
    correct = row_correct and col_correct
    score = (int(row_correct) + int(col_correct)) / 2.0
    return {"correct": correct, "score": score, "row_correct": row_correct, "col_correct": col_correct}


@register_scorer("set_match")
def score_set_match(parsed: Optional[str], ground_truth: str) -> dict:
    """Unordered set comparison with precision/recall."""
    if parsed is None:
        return {"correct": False, "score": 0.0, "precision": 0.0, "recall": 0.0}
    pred_set = set(x.strip().upper() for x in parsed.split(",") if x.strip())
    gt_set = set(x.strip().upper() for x in ground_truth.split(",") if x.strip())
    if not gt_set:
        return {"correct": len(pred_set) == 0, "score": 1.0 if len(pred_set) == 0 else 0.0,
                "precision": 1.0 if len(pred_set) == 0 else 0.0, "recall": 1.0}
    if not pred_set:
        return {"correct": False, "score": 0.0, "precision": 0.0, "recall": 0.0}
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    correct = pred_set == gt_set
    score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"correct": correct, "score": score, "precision": precision, "recall": recall}
