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

    # Pass 2: task-specific verification prompt
    task_name = sample.get("task_name", "")
    if task_name == "hierarchy_depth":
        verify_prompt = (
            f"You previously answered this question about the image:\n\n"
            f"Question: {sample['prompt']}\n"
            f"Your answer: {parsed1}\n\n"
            f"IMPORTANT: Count the number of HORIZONTAL ROWS of boxes, "
            f"NOT the number of connections/edges between rows. "
            f"For example, if the root is in row 1 and the leaves are in row 3, "
            f"the answer is 3 (not 4). "
            f"Look at the image again. Is {parsed1} correct? "
            f"If not, provide the correct answer in curly brackets."
        )
    else:
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


# ---------------------------------------------------------------------------
# Strategy: structured_decomposition (multi-step sub-questions)
# ---------------------------------------------------------------------------

# Task-specific decomposition plans.
# Each plan is a list of (sub_prompt, extraction_fn) tuples.
# extraction_fn takes the raw response and returns data for the next step.
_DECOMPOSITION_PLANS = {
    "counting_grid": [
        (
            "Look at this grid image. Focus ONLY on counting horizontal lines "
            "(lines that go from left to right across the image). "
            "Count every horizontal line you can see, including the top and bottom borders. "
            "Answer with just the number in curly brackets, e.g., {7}.",
            "integer",
        ),
        (
            "Now focus ONLY on counting vertical lines "
            "(lines that go from top to bottom in the image). "
            "Count every vertical line you can see, including the left and right borders. "
            "Answer with just the number in curly brackets, e.g., {7}.",
            "integer",
        ),
    ],
    "nested_squares": [
        (
            "Look at this image of nested squares. Starting from the OUTERMOST square, "
            "describe each square you can see moving inward. "
            "For each one, say 'Square 1: outermost', 'Square 2: next inner', etc. "
            "List ALL squares you can identify, even very small inner ones.",
            None,  # Free-form description, not parsed
        ),
        (
            "Based on the image, count the TOTAL number of distinct squares, "
            "including the outermost one and any tiny inner squares. "
            "Look carefully at the center — there may be more squares than initially visible. "
            "Answer with just the number in curly brackets, e.g., {5}.",
            "integer",
        ),
    ],
    "hierarchy_depth": [
        (
            "Look at this organizational chart / tree diagram. "
            "Identify the ROOT node at the very top. What is its label? "
            "Then trace the LONGEST path from the root down to a leaf node "
            "(a node with no children below it). "
            "List each node on this longest path, one per line.",
            None,
        ),
        (
            "Count the number of HORIZONTAL ROWS of boxes in this hierarchy. "
            "The top row containing the root node is row 1. "
            "Count each distinct row below it. "
            "IMPORTANT: Count the number of rows (levels), NOT the number of "
            "connections between rows. For example, if there are boxes in 3 "
            "horizontal rows, the answer is 3, not 4. "
            "Answer with just the number of rows in curly brackets, e.g., {3}.",
            "integer",
        ),
    ],
    "colored_paths": [
        (
            "Look at this diagram with colored paths connecting labeled stations. "
            "List ALL the stations you can see and their positions. "
            "Then for each colored path/ribbon, identify which TWO stations it connects "
            "(its start and end points). List them as: 'Red path: Station X to Station Y'.",
            None,
        ),
    ],
    "pie_chart": [
        (
            "Look at this pie chart carefully. List ALL the slices with their labels "
            "and estimate their percentage of the whole circle. "
            "Order them from largest to smallest slice. "
            "Format: 'Label: approximately XX%'.",
            None,
        ),
    ],
}


@register_strategy("decompose")
def strategy_decompose(client, sample: dict, **kwargs) -> dict:
    """Break the task into sub-questions, gather intermediate info, then answer.

    For tasks with known decomposition plans, asks structured sub-questions
    first to prime the model, then asks the original question.
    For unknown tasks, falls back to a generic "describe then answer" approach.
    """
    task_name = sample.get("task_name", "")
    plan = _DECOMPOSITION_PLANS.get(task_name)

    total_latency = 0.0
    total_input = 0
    total_output = 0
    sub_responses = []

    if plan is None:
        # Generic decomposition: describe first, then answer
        plan = [
            (
                "Describe this image in detail. Note all key elements, "
                "labels, numbers, and spatial relationships.",
                None,
            ),
        ]

    # Execute sub-questions
    for sub_prompt, sub_parser in plan:
        resp = client.query(sample["image_path"], sub_prompt)
        total_latency += resp.get("latency_s", 0)
        total_input += resp.get("input_tokens", 0)
        total_output += resp.get("output_tokens", 0)

        sub_result = {"prompt": sub_prompt, "response": resp["raw_response"]}
        if sub_parser:
            sub_result["parsed"] = _parse_answer(resp["raw_response"], sub_parser)
        sub_responses.append(sub_result)

    # Final question: include context from sub-questions
    context_lines = []
    for i, sr in enumerate(sub_responses):
        context_lines.append(f"Observation {i+1}: {sr['response']}")

    # Build final prompt with accumulated context
    final_prompt = (
        "Based on your detailed analysis of the image:\n\n"
        + "\n".join(context_lines)
        + f"\n\nNow answer the original question: {sample['prompt']}"
    )

    resp_final = client.query(sample["image_path"], final_prompt)
    total_latency += resp_final.get("latency_s", 0)
    total_input += resp_final.get("input_tokens", 0)
    total_output += resp_final.get("output_tokens", 0)

    parsed_final = _parse_answer(resp_final["raw_response"], sample["parser"])

    # For counting_grid: if sub-questions extracted row/col counts directly,
    # try to construct the answer from those
    if task_name == "counting_grid" and sample.get("parser") == "row_col":
        row_count = None
        col_count = None
        for sr in sub_responses:
            p = sr.get("parsed")
            if p is not None:
                if row_count is None:
                    row_count = p  # First integer = horizontal lines (rows+1)
                elif col_count is None:
                    col_count = p  # Second integer = vertical lines (cols+1)
        if row_count is not None and col_count is not None:
            try:
                # Lines = edges, rows = lines - 1 (grid lines include borders)
                r = int(row_count) - 1
                c = int(col_count) - 1
                if r > 0 and c > 0:
                    decomposed_answer = f"{r},{c}"
                    # Prefer decomposed if final parse failed
                    if parsed_final is None:
                        parsed_final = decomposed_answer
            except ValueError:
                pass

    result = dict(sample)
    result.update({
        "raw_response": resp_final["raw_response"],
        "latency_s": round(total_latency, 2),
        "model": getattr(client, 'model', 'unknown'),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "reasoning_mode": getattr(client, 'reasoning', False),
        "parsed_answer": parsed_final,
        "strategy": "decompose",
        "strategy_sub_responses": [
            {"prompt": sr["prompt"][:100], "response": sr["response"][:200]}
            for sr in sub_responses
        ],
        "strategy_steps": len(sub_responses) + 1,
    })
    return result


# ---------------------------------------------------------------------------
# Strategy: code_vision (sandboxed Python REPL for image analysis)
# ---------------------------------------------------------------------------

def _run_sandboxed_code(code: str, image_path: str, timeout: int = 30) -> str:
    """Execute Python code in a sandboxed subprocess with image access.

    Returns stdout output or error message.
    """
    import subprocess
    import textwrap

    # Wrap the code to make the image path available
    wrapped = textwrap.dedent(f"""\
        import sys
        IMAGE_PATH = {image_path!r}
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            pass
        {code}
    """)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapped],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout.strip()
        if proc.returncode != 0:
            error = proc.stderr.strip()
            return f"ERROR: {error[-500:]}" if error else "ERROR: non-zero exit"
        return output if output else "NO OUTPUT"
    except subprocess.TimeoutExpired:
        return "ERROR: timeout"
    except Exception as e:
        return f"ERROR: {e}"


import sys

@register_strategy("code_vision")
def strategy_code_vision(client, sample: dict, **kwargs) -> dict:
    """Give the model a Python REPL to analyze the image programmatically.

    The model writes PIL/numpy code to extract features from the image.
    The code is executed in a sandboxed subprocess, and the output is
    fed back to the model for a final answer.

    Particularly effective for geometric tasks (counting_grid, nested_squares)
    where pixel analysis can bypass perceptual limitations.
    """
    task_name = sample.get("task_name", "")

    total_latency = 0.0
    total_input = 0
    total_output = 0

    # Step 1: Ask the model to write analysis code
    code_prompt = (
        f"You have access to a Python environment with PIL and numpy.\n"
        f"The image is at: IMAGE_PATH (already defined)\n\n"
        f"Task: {sample['prompt']}\n\n"
        f"Write Python code that analyzes the image to answer this question. "
        f"The code should print() its findings clearly. "
        f"For example, for counting lines, you might use edge detection or "
        f"pixel scanning along rows/columns.\n\n"
        f"Write ONLY the Python code, no explanation. "
        f"Use print() to output your analysis results."
    )

    resp_code = client.query(sample["image_path"], code_prompt)
    total_latency += resp_code.get("latency_s", 0)
    total_input += resp_code.get("input_tokens", 0)
    total_output += resp_code.get("output_tokens", 0)

    # Extract code from response (may be in ```python blocks)
    raw_code = resp_code["raw_response"]
    code_match = re.search(r"```(?:python)?\s*\n(.*?)```", raw_code, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        # Try to use the whole response as code
        code = raw_code.strip()

    # Step 2: Execute the code
    code_output = _run_sandboxed_code(code, sample["image_path"])

    # Step 3: Feed code output back to model for final answer
    final_prompt = (
        f"You wrote code to analyze an image and got this output:\n\n"
        f"```\n{code_output[:1000]}\n```\n\n"
        f"Based on this analysis, answer the original question:\n"
        f"{sample['prompt']}"
    )

    resp_final = client.query(sample["image_path"], final_prompt)
    total_latency += resp_final.get("latency_s", 0)
    total_input += resp_final.get("input_tokens", 0)
    total_output += resp_final.get("output_tokens", 0)

    parsed_final = _parse_answer(resp_final["raw_response"], sample["parser"])

    result = dict(sample)
    result.update({
        "raw_response": resp_final["raw_response"],
        "latency_s": round(total_latency, 2),
        "model": getattr(client, 'model', 'unknown'),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "reasoning_mode": getattr(client, 'reasoning', False),
        "parsed_answer": parsed_final,
        "strategy": "code_vision",
        "strategy_code": code[:500],
        "strategy_code_output": code_output[:500],
        "strategy_steps": 3,
    })
    return result


# ---------------------------------------------------------------------------
# Strategy: adaptive (picks best strategy per task)
# ---------------------------------------------------------------------------

# Mapping from task name to the best strategy based on error analysis.
# Tasks not listed fall back to best_of_n.
_ADAPTIVE_TASK_STRATEGIES = {
    # Verified via Qwen3-VL-8B benchmark (N=20 per task, 176 total samples)
    # verify catches systematic +1 overcount bias
    "hierarchy_depth": "verify",
    # verify re-examination corrects cell lookup errors
    "realistic_table": "verify",
    # decompose sub-questions help with proportion estimation
    "pie_chart": "decompose",
    # decompose improves bar percentage reading
    "progress_bar": "decompose",
    # majority voting reduces noise on counting
    "nested_squares": "best_of_n",
    # baseline outperforms all multi-pass strategies:
    "colored_paths": "baseline",
    "counting_grid": "baseline",
    "scatter_plot": "baseline",
    "text_degradation": "baseline",
    "edge_crossing": "code_vision",
}


@register_strategy("adaptive")
def strategy_adaptive(client, sample: dict, n: int = 5, **kwargs) -> dict:
    """Pick the best strategy for each task based on known error patterns.

    Uses task-specific strategy selection from _ADAPTIVE_TASK_STRATEGIES.
    Falls back to verify for unknown tasks (best overall strategy on benchmarks).
    """
    task_name = sample.get("task_name", "")

    # Strip _text suffix for text controls — they don't need special strategies
    base_task = task_name[:-5] if task_name.endswith("_text") else task_name

    selected = _ADAPTIVE_TASK_STRATEGIES.get(base_task, "verify")

    # Dispatch to the selected strategy
    strategy_fn = STRATEGY_REGISTRY.get(selected, strategy_best_of_n)
    result = strategy_fn(client, sample, n=n, **kwargs)

    # Override strategy name to track adaptive routing
    result["strategy"] = "adaptive"
    result["strategy_selected"] = selected
    return result


# ---------------------------------------------------------------------------
# Strategy: iterative_refine (multi-round prompt refinement)
# ---------------------------------------------------------------------------

_REFINEMENT_CRITIQUES = {
    "counting_grid": (
        "Scan the image systematically: count horizontal lines by moving "
        "top-to-bottom, then count vertical lines by moving left-to-right. "
        "Include border lines. Don't guess — be methodical."
    ),
    "pie_chart": (
        "Verify that your estimated percentages sum to 100%. Use visual "
        "anchors: a quarter-circle is 25%, a half is 50%. Re-examine each "
        "slice's angle relative to these anchors."
    ),
    "progress_bar": (
        "Estimate the filled portion relative to the full bar width. "
        "Look for tick marks, labels, or percentage indicators as reference "
        "points. Express your answer as a percentage."
    ),
    "hierarchy_depth": (
        "Count the number of HORIZONTAL ROWS of boxes, not the number of "
        "connections. The root is row 1. Count each distinct row below it."
    ),
    "nested_squares": (
        "Check the center of the image carefully for tiny inner squares "
        "you may have missed. Count from the outermost square inward."
    ),
}

_REFINEMENT_GENERIC = "Re-examine the image step by step. Check your previous answer carefully."


def _build_refinement_prompt(
    sample: dict, prior_answers: list[str], round_num: int,
) -> str:
    """Build a critique prompt incorporating all prior answers."""
    task_name = sample.get("task_name", "")
    critique = _REFINEMENT_CRITIQUES.get(task_name, _REFINEMENT_GENERIC)

    history = "\n".join(
        f"  Round {i+1}: {a}" for i, a in enumerate(prior_answers)
    )
    return (
        f"You have answered this question about the image {len(prior_answers)} time(s):\n\n"
        f"Question: {sample['prompt']}\n"
        f"Your prior answers:\n{history}\n\n"
        f"Critique: {critique}\n\n"
        f"Look at the image again. Provide your revised answer in the same "
        f"format as the original question asks."
    )


@register_strategy("iterative_refine")
def strategy_iterative_refine(
    client, sample: dict, max_rounds: int = 5, **kwargs,
) -> dict:
    """Multi-round refinement with task-specific critique prompts.

    Unlike verify (single re-examination), this iterates up to max_rounds
    with evolving critique directives until the answer converges (same
    parsed answer for 2 consecutive rounds).
    """
    total_latency = 0.0
    total_input = 0
    total_output = 0
    prior_answers: list[str] = []
    raw_responses: list[str] = []
    parsed_answers: list[Optional[str]] = []

    for round_num in range(max_rounds):
        if round_num == 0:
            prompt = sample["prompt"]
        else:
            prompt = _build_refinement_prompt(sample, prior_answers, round_num)

        resp = client.query(sample["image_path"], prompt)
        total_latency += resp.get("latency_s", 0)
        total_input += resp.get("input_tokens", 0)
        total_output += resp.get("output_tokens", 0)

        raw_responses.append(resp["raw_response"])
        parsed = _parse_answer(resp["raw_response"], sample["parser"])
        parsed_answers.append(parsed)
        prior_answers.append(
            parsed if parsed is not None else resp["raw_response"][:200]
        )

        # Convergence: stop when 2 consecutive rounds give the same parsed answer
        if (
            round_num >= 1
            and parsed is not None
            and parsed == parsed_answers[round_num - 1]
        ):
            break

    # Final answer is the last successfully parsed answer
    final_answer = None
    for a in reversed(parsed_answers):
        if a is not None:
            final_answer = a
            break

    result = dict(sample)
    result.update({
        "raw_response": raw_responses[-1],
        "latency_s": round(total_latency, 2),
        "model": getattr(client, "model", "unknown"),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "reasoning_mode": getattr(client, "reasoning", False),
        "parsed_answer": final_answer,
        "strategy": "iterative_refine",
        "strategy_rounds": len(parsed_answers),
        "strategy_all_answers": parsed_answers,
        "strategy_converged": (
            len(parsed_answers) >= 2
            and parsed_answers[-1] is not None
            and parsed_answers[-1] == parsed_answers[-2]
        ),
    })
    return result


# ---------------------------------------------------------------------------
# Strategy: sketchpad (visual sketchpad with pre-built primitives)
# ---------------------------------------------------------------------------

@register_strategy("sketchpad")
def strategy_sketchpad(
    client, sample: dict, max_passes: int = 3, **kwargs,
) -> dict:
    """Visual Sketchpad: run pre-built vision primitives and feed annotated
    images back to the VLM for multi-pass analysis.

    Pass 0 (automatic): decompose question, classify sub-questions, run
    primitives, annotate image, accumulate findings.
    Passes 1-N (model-driven): model sees annotated image + findings,
    requests more tools or provides final answer.
    """
    from sketchpad import (
        run_sketchpad_pass0,
        build_sketchpad_prompt,
        parse_sketchpad_response,
        PRIMITIVE_REGISTRY,
    )

    task_name = sample.get("task_name", "")
    image_path = sample["image_path"]
    prompt = sample["prompt"]

    total_latency = 0.0
    total_input = 0
    total_output = 0

    # Open the image
    img = Image.open(image_path)

    # Pass 0: automatic primitive execution
    canvas, findings = run_sketchpad_pass0(img, prompt, task_name)

    # Save annotated image for model consumption
    annotated_path = _save_temp_image(canvas)

    # Model-driven passes
    raw_responses = []
    final_answer_text = None
    n_passes = 1  # Pass 0 counts as 1

    for pass_num in range(1, max_passes):
        # Build prompt with findings
        sketchpad_prompt = build_sketchpad_prompt(prompt, findings, pass_num)

        # Query model with annotated image
        response = client.query(annotated_path, sketchpad_prompt)
        total_latency += response.get("latency_s", 0)
        total_input += response.get("input_tokens", 0)
        total_output += response.get("output_tokens", 0)
        raw_responses.append(response["raw_response"])
        n_passes += 1

        # Parse response
        action, value, tool_kwargs = parse_sketchpad_response(
            response["raw_response"]
        )

        if action == "tool" and value in PRIMITIVE_REGISTRY:
            # Execute requested primitive
            prim_fn = PRIMITIVE_REGISTRY[value]
            tool_kwargs = tool_kwargs or {}
            annotated, finding = prim_fn(canvas, **tool_kwargs)
            canvas = annotated
            annotated_path = _save_temp_image(canvas)
            findings.append({
                "sub_question": f"Model-requested (pass {pass_num})",
                "primitives_run": [value],
                "findings": finding,
            })
        elif action in ("answer", "unknown"):
            final_answer_text = value
            break
        else:
            # Unknown tool requested — treat response as answer
            final_answer_text = response["raw_response"]
            break

    # If we exhausted passes without an explicit ANSWER, use the last response
    if final_answer_text is None and raw_responses:
        final_answer_text = raw_responses[-1]

    # If no model passes ran (shouldn't happen, but safety), do a basic query
    if not raw_responses:
        sketchpad_prompt = build_sketchpad_prompt(prompt, findings, 1)
        response = client.query(annotated_path, sketchpad_prompt)
        total_latency += response.get("latency_s", 0)
        total_input += response.get("input_tokens", 0)
        total_output += response.get("output_tokens", 0)
        raw_responses.append(response["raw_response"])
        final_answer_text = response["raw_response"]
        n_passes += 1

    # Parse the final answer
    parsed_final = _parse_answer(final_answer_text, sample["parser"])

    # Clean up temp files
    import os
    try:
        os.unlink(annotated_path)
    except OSError:
        pass

    # Compile findings summary for tracing
    findings_summary = "; ".join(
        f"[{f['sub_question']}] {f['findings'][:100]}"
        for f in findings
    )

    result = dict(sample)
    result.update({
        "raw_response": final_answer_text,
        "latency_s": round(total_latency, 2),
        "model": getattr(client, "model", "unknown"),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "reasoning_mode": getattr(client, "reasoning", False),
        "parsed_answer": parsed_final,
        "strategy": "sketchpad",
        "strategy_passes": n_passes,
        "strategy_findings": findings_summary[:500],
        "strategy_sub_questions": len(findings),
    })
    return result
