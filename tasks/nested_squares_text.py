"""Task: Text-only nested squares — perception vs reasoning diagnostic.

Provides square corner coordinates as text instead of an image.
If text accuracy >> image accuracy, the failure is perceptual.
"""

from random import Random

from PIL import Image

TASK_CONFIG = {
    "task_name": "nested_squares_text",
    "prompt_template": "",  # filled dynamically
    "prompt_template_v2": "",
    "parser": "integer",
    "scorer": "integer_distance",
    "default_params": {
        "depth": 3,
        "reduction_factor": 0.6,
    },
    "sweep_axes": {
        "depth": [2, 3, 4, 5, 6, 7, 8],
        "reduction_factor": [0.4, 0.5, 0.6, 0.7, 0.8],
    },
}

_call_counter = 0


def render(
    depth: int = 3,
    reduction_factor: float = 0.6,
    seed: int | None = None,
    prompt_variant: int = 1,
) -> tuple[Image.Image, str, dict]:
    """Return a tiny placeholder image with square coordinates as text."""
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    # Cap depth so smallest square side >= 5% of canvas (mirrors image task)
    effective_depth = depth
    size_check = 0.85
    for d in range(1, depth + 1):
        if size_check < 0.05:
            effective_depth = d - 1
            break
        size_check *= reduction_factor
    effective_depth = max(1, effective_depth)

    # Generate square coordinates (same logic as nested_squares.py)
    squares = []
    size = 0.85
    cx, cy = 0.5, 0.5

    for i in range(effective_depth):
        half = size / 2
        x0, y0 = cx - half, cy - half
        x1, y1 = cx + half, cy + half
        squares.append((round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)))

        size *= reduction_factor
        parent_half = half
        max_jitter = min(0.03, parent_half * 0.2)
        cx += rng.uniform(-max_jitter, max_jitter)
        cy += rng.uniform(-max_jitter, max_jitter)
        half_new = size / 2
        cx = max(half_new + 0.02, min(1 - half_new - 0.02, cx))
        cy = max(half_new + 0.02, min(1 - half_new - 0.02, cy))

    # Format coordinates as text
    square_descriptions = []
    for i, (x0, y0, x1, y1) in enumerate(squares):
        square_descriptions.append(
            f"  Square {i + 1}: corners at ({x0}, {y0}), ({x1}, {y0}), ({x1}, {y1}), ({x0}, {y1})"
        )
    squares_text = "\n".join(square_descriptions)

    if prompt_variant == 2:
        prompt = (
            f"The following squares are drawn on a 1×1 canvas:\n{squares_text}\n"
            f"How many squares are there in total? "
            f"Put your answer in curly brackets, e.g., {{3}}."
        )
    else:
        prompt = (
            f"An image contains squares defined by these corner coordinates:\n{squares_text}\n"
            f"Some squares are nested inside others.\n"
            f"Count the total number of squares. "
            f"Answer with only the number in curly brackets, e.g., {{3}}."
        )

    # Tiny placeholder image (API requires an image)
    img = Image.new("RGB", (64, 64), "white")

    ground_truth = str(effective_depth)
    metadata = {
        "depth": effective_depth,
        "requested_depth": depth,
        "reduction_factor": reduction_factor,
        "squares": squares,
        "prompt": prompt,
        "mode": "text_only",
    }
    return img, ground_truth, metadata
