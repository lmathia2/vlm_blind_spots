"""Inference-time strategies for improving VLM accuracy without retraining.

Each strategy wraps the client.query() call with additional inference-time
compute: majority voting, crop-zoom-reask, or answer verification.

Strategies are composable and pluggable via the CLI --strategy flag.
"""

import base64
import io
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional

from PIL import Image

from parsers import PARSER_REGISTRY


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, callable] = {}


def register_strategy(name: str):
    def decorator(fn):
        STRATEGY_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_answer(raw_response: str, parser_name: str) -> Optional[str]:
    """Parse a raw response using the named parser."""
    parser_fn = PARSER_REGISTRY.get(parser_name)
    if parser_fn:
        return parser_fn(raw_response)
    return None


def _majority_vote(answers: list[Optional[str]]) -> Optional[str]:
    """Return the most common non-None answer, or None if all failed."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counter = Counter(valid)
    return counter.most_common(1)[0][0]


def _save_temp_image(img: Image.Image) -> str:
    """Save a PIL image to a temporary PNG and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp, format="PNG")
    tmp.close()
    return tmp.name


def _crop_image(image_path: str, bbox: tuple[float, float, float, float]) -> Image.Image:
    """Crop an image to a bounding box (left, upper, right, lower) as fractions [0,1]."""
    img = Image.open(image_path)
    w, h = img.size
    left = int(bbox[0] * w)
    upper = int(bbox[1] * h)
    right = int(bbox[2] * w)
    lower = int(bbox[3] * h)
    cropped = img.crop((left, upper, right, lower))
    # Upscale small crops to at least 512px on the short side
    cw, ch = cropped.size
    min_side = min(cw, ch)
    if min_side < 512:
        scale = 512 / min_side
        cropped = cropped.resize(
            (int(cw * scale), int(ch * scale)), Image.LANCZOS
        )
    return cropped


def _tile_image(image_path: str, grid: tuple[int, int] = (2, 2)) -> list[Image.Image]:
    """Split an image into a grid of tiles. Returns list of PIL images."""
    img = Image.open(image_path)
    w, h = img.size
    rows, cols = grid
    tile_w = w // cols
    tile_h = h // rows
    tiles = []
    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            upper = r * tile_h
            right = left + tile_w
            lower = upper + tile_h
            tile = img.crop((left, upper, right, lower))
            # Upscale tiles to reasonable size
            tw, th = tile.size
            min_side = min(tw, th)
            if min_side < 384:
                scale = 384 / min_side
                tile = tile.resize(
                    (int(tw * scale), int(th * scale)), Image.LANCZOS
                )
            tiles.append(tile)
    return tiles


# ---------------------------------------------------------------------------
# Strategy: baseline (current behavior, no-op wrapper)
# ---------------------------------------------------------------------------

@register_strategy("baseline")
def strategy_baseline(client, sample: dict, **kwargs) -> dict:
    """Single-pass evaluation. Same as current behavior."""
    response = client.query(sample["image_path"], sample["prompt"])
    result = dict(sample)
    result.update(response)
    result["parsed_answer"] = _parse_answer(response["raw_response"], sample["parser"])
    result["strategy"] = "baseline"
    return result


# ---------------------------------------------------------------------------
# Strategy: best_of_n (majority voting)
# ---------------------------------------------------------------------------

@register_strategy("best_of_n")
def strategy_best_of_n(client, sample: dict, n: int = 5, **kwargs) -> dict:
    """Sample N responses and take majority vote on parsed answers.

    Uses temperature > 0 for diversity. The original client temperature
    is temporarily overridden.
    """
    original_temp = getattr(client, 'temperature', 0.0)
    voting_temp = max(original_temp, 0.7)  # Need diversity for voting

    parsed_answers = []
    raw_responses = []
    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for i in range(n):
        # Override temperature for diversity
        client.temperature = voting_temp
        try:
            response = client.query(sample["image_path"], sample["prompt"])
        finally:
            client.temperature = original_temp

        raw_responses.append(response["raw_response"])
        parsed = _parse_answer(response["raw_response"], sample["parser"])
        parsed_answers.append(parsed)
        total_latency += response.get("latency_s", 0)
        total_input_tokens += response.get("input_tokens", 0)
        total_output_tokens += response.get("output_tokens", 0)

    winner = _majority_vote(parsed_answers)

    # Find the raw response that produced the winning answer
    winning_raw = raw_responses[0]
    for raw, parsed in zip(raw_responses, parsed_answers):
        if parsed == winner:
            winning_raw = raw
            break

    result = dict(sample)
    result.update({
        "raw_response": winning_raw,
        "latency_s": round(total_latency, 2),
        "model": getattr(client, 'model', 'unknown'),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "reasoning_mode": getattr(client, 'reasoning', False),
        "parsed_answer": winner,
        "strategy": "best_of_n",
        "strategy_n": n,
        "strategy_votes": dict(Counter(a for a in parsed_answers if a is not None)),
        "strategy_all_answers": parsed_answers,
    })
    return result


# ---------------------------------------------------------------------------
# Strategy: crop_zoom (tile or center-crop, reask, aggregate)
# ---------------------------------------------------------------------------

# Task-specific crop configurations
_CROP_CONFIGS = {
    # Counting tasks: tile into quadrants
    "counting_grid": {"method": "tile", "grid": (2, 2), "aggregate": "sum_rows_cols"},
    # Nested shapes: zoom into center
    "nested_squares": {"method": "center_zoom", "zoom": 2.0},
    # Charts: full + center crop for detail
    "pie_chart": {"method": "center_zoom", "zoom": 1.5},
    # Hierarchy: tile vertically to see each level
    "hierarchy_depth": {"method": "tile", "grid": (2, 1), "aggregate": "max"},
    # Text: crop and upscale
    "text_degradation": {"method": "center_zoom", "zoom": 2.0},
    # Paths: full image is needed, use with verify instead
    "colored_paths": {"method": "center_zoom", "zoom": 1.5},
}


@register_strategy("crop_zoom")
def strategy_crop_zoom(client, sample: dict, **kwargs) -> dict:
    """Crop/zoom regions of the image and reask, then aggregate.

    Uses task-specific crop configurations for known tasks,
    falls back to center-zoom for unknown tasks.
    """
    task_name = sample.get("task_name", "")
    config = _CROP_CONFIGS.get(task_name, {"method": "center_zoom", "zoom": 1.5})
    method = config["method"]

    # Pass 1: normal full-image query
    response_full = client.query(sample["image_path"], sample["prompt"])
    parsed_full = _parse_answer(response_full["raw_response"], sample["parser"])

    total_latency = response_full.get("latency_s", 0)
    total_input = response_full.get("input_tokens", 0)
    total_output = response_full.get("output_tokens", 0)

    # Pass 2: crop/zoom queries
    crop_answers = []
    temp_files = []

    try:
        if method == "center_zoom":
            zoom = config.get("zoom", 2.0)
            # Crop center portion
            margin = 1.0 / (2.0 * zoom)
            center_bbox = (0.5 - margin, 0.5 - margin, 0.5 + margin, 0.5 + margin)
            cropped = _crop_image(sample["image_path"], center_bbox)
            tmp_path = _save_temp_image(cropped)
            temp_files.append(tmp_path)

            zoom_prompt = (
                "Look very carefully at this zoomed-in view. "
                + sample["prompt"]
            )
            resp = client.query(tmp_path, zoom_prompt)
            total_latency += resp.get("latency_s", 0)
            total_input += resp.get("input_tokens", 0)
            total_output += resp.get("output_tokens", 0)
            crop_answers.append(_parse_answer(resp["raw_response"], sample["parser"]))

        elif method == "tile":
            grid = config.get("grid", (2, 2))
            tiles = _tile_image(sample["image_path"], grid)
            tile_labels = []
            for idx, (r, c) in enumerate(
                (r, c) for r in range(grid[0]) for c in range(grid[1])
            ):
                tile_labels.append(f"row {r+1}, column {c+1}")

            for idx, tile in enumerate(tiles):
                tmp_path = _save_temp_image(tile)
                temp_files.append(tmp_path)

                tile_prompt = (
                    f"This is a zoomed-in view of {tile_labels[idx]} of the image. "
                    + sample["prompt"]
                )
                resp = client.query(tmp_path, tile_prompt)
                total_latency += resp.get("latency_s", 0)
                total_input += resp.get("input_tokens", 0)
                total_output += resp.get("output_tokens", 0)
                crop_answers.append(
                    _parse_answer(resp["raw_response"], sample["parser"])
                )
    finally:
        # Clean up temp files
        for tf in temp_files:
            try:
                Path(tf).unlink()
            except OSError:
                pass

    # Aggregate: combine full-image and crop answers
    all_answers = [parsed_full] + crop_answers
    aggregate = config.get("aggregate", "vote")

    if aggregate == "vote":
        # Majority vote across all passes
        final_answer = _majority_vote(all_answers)
    elif aggregate == "max":
        # Take the maximum numeric answer (for counting/depth tasks)
        nums = []
        for a in all_answers:
            if a is not None:
                try:
                    nums.append(int(a))
                except ValueError:
                    pass
        final_answer = str(max(nums)) if nums else parsed_full
    elif aggregate == "sum_rows_cols":
        # For counting_grid: full-image answer is likely best, but verify
        # by checking tile consistency
        final_answer = _majority_vote(all_answers)
    else:
        final_answer = _majority_vote(all_answers)

    result = dict(sample)
    result.update({
        "raw_response": response_full["raw_response"],
        "latency_s": round(total_latency, 2),
        "model": getattr(client, 'model', 'unknown'),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "reasoning_mode": getattr(client, 'reasoning', False),
        "parsed_answer": final_answer,
        "strategy": "crop_zoom",
        "strategy_full_answer": parsed_full,
        "strategy_crop_answers": crop_answers,
        "strategy_all_answers": all_answers,
    })
    return result


# ---------------------------------------------------------------------------
# Strategy: verify (answer → verification pass → final)
# ---------------------------------------------------------------------------

@register_strategy("verify")
def strategy_verify(client, sample: dict, **kwargs) -> dict:
    """Two-pass strategy: get initial answer, then verify it.

    Pass 1: Normal query
    Pass 2: Show the image again and ask model to verify its previous answer
    If the model changes its answer, take the new one.
    """
    # Pass 1: initial answer
    resp1 = client.query(sample["image_path"], sample["prompt"])
    parsed1 = _parse_answer(resp1["raw_response"], sample["parser"])

    total_latency = resp1.get("latency_s", 0)
    total_input = resp1.get("input_tokens", 0)
    total_output = resp1.get("output_tokens", 0)

    if parsed1 is None:
        # Can't verify a failed parse, return as-is
        result = dict(sample)
        result.update(resp1)
        result["parsed_answer"] = None
        result["strategy"] = "verify"
        return result

    # Pass 2: verification
    verify_prompt = (
        f"You previously answered this question about the image:\n\n"
        f"Question: {sample['prompt']}\n"
        f"Your answer: {parsed1}\n\n"
        f"Look at the image again very carefully. "
        f"Is your answer correct? If not, provide the correct answer. "
        f"Respond with ONLY the corrected answer in the same format as before."
    )
    resp2 = client.query(sample["image_path"], verify_prompt)
    total_latency += resp2.get("latency_s", 0)
    total_input += resp2.get("input_tokens", 0)
    total_output += resp2.get("output_tokens", 0)

    # Check if the model confirmed or changed its answer
    raw2 = resp2["raw_response"].strip().lower()
    confirmed = any(phrase in raw2 for phrase in [
        "correct", "yes", "confirmed", "right", "accurate",
    ])

    parsed2 = _parse_answer(resp2["raw_response"], sample["parser"])

    # If verification produced a valid new answer, use it
    # If it confirmed or failed to parse, keep original
    if parsed2 is not None and not confirmed:
        final_answer = parsed2
    else:
        final_answer = parsed1

    result = dict(sample)
    result.update({
        "raw_response": resp1["raw_response"],
        "latency_s": round(total_latency, 2),
        "model": getattr(client, 'model', 'unknown'),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "reasoning_mode": getattr(client, 'reasoning', False),
        "parsed_answer": final_answer,
        "strategy": "verify",
        "strategy_initial_answer": parsed1,
        "strategy_verify_response": resp2["raw_response"],
        "strategy_verify_confirmed": confirmed,
        "strategy_final_changed": final_answer != parsed1,
    })
    return result


# ---------------------------------------------------------------------------
# Strategy: composite (chain multiple strategies)
# ---------------------------------------------------------------------------

@register_strategy("best_of_n_verify")
def strategy_best_of_n_verify(client, sample: dict, n: int = 5, **kwargs) -> dict:
    """Best-of-N followed by verification of the winning answer.

    Combines majority voting for noise reduction with a verification
    pass to catch systematic errors.
    """
    # Phase 1: best-of-n to get consensus
    bon_result = strategy_best_of_n(client, sample, n=n)
    consensus = bon_result["parsed_answer"]

    if consensus is None:
        bon_result["strategy"] = "best_of_n_verify"
        return bon_result

    # Phase 2: verify the consensus answer
    verify_prompt = (
        f"Look at this image very carefully.\n\n"
        f"Question: {sample['prompt']}\n"
        f"A previous analysis concluded the answer is: {consensus}\n\n"
        f"Examine the image closely. Is this answer correct? "
        f"If not, what is the correct answer? "
        f"Respond with ONLY the answer in the same format."
    )
    resp_v = client.query(sample["image_path"], verify_prompt)

    raw_v = resp_v["raw_response"].strip().lower()
    confirmed = any(phrase in raw_v for phrase in [
        "correct", "yes", "confirmed", "right", "accurate",
    ])
    parsed_v = _parse_answer(resp_v["raw_response"], sample["parser"])

    if parsed_v is not None and not confirmed:
        final_answer = parsed_v
    else:
        final_answer = consensus

    result = dict(bon_result)
    result.update({
        "latency_s": round(
            bon_result["latency_s"] + resp_v.get("latency_s", 0), 2
        ),
        "input_tokens": bon_result["input_tokens"] + resp_v.get("input_tokens", 0),
        "output_tokens": bon_result["output_tokens"] + resp_v.get("output_tokens", 0),
        "parsed_answer": final_answer,
        "strategy": "best_of_n_verify",
        "strategy_consensus": consensus,
        "strategy_verify_response": resp_v["raw_response"],
        "strategy_verify_confirmed": confirmed,
        "strategy_final_changed": final_answer != consensus,
    })
    return result
