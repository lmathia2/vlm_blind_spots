"""Task: Text-only line intersection — perception vs reasoning diagnostic.

Provides the same line coordinates as text instead of an image.
If text accuracy >> image accuracy, the failure is perceptual.
"""

from random import Random

from PIL import Image

from tasks.line_intersection import (
    _generate_path, _count_intersections, _paths_are_valid,
)

TASK_CONFIG = {
    "task_name": "line_intersection_text",
    "prompt_template": "",  # filled dynamically per sample
    "parser": "integer",
    "scorer": "integer_distance",
    "default_params": {
        "grid_size": 6,
        "target_intersections": None,
    },
    "sweep_axes": {
        "target_intersections": [0, 1, 2],
    },
}

_call_counter = 0


def render(grid_size: int = 6,
           target_intersections: int | None = None) -> tuple[Image.Image, str, dict]:
    """Return a tiny placeholder image with line coords as text in the prompt."""
    global _call_counter
    _call_counter += 1
    rng = Random(_call_counter)

    if target_intersections is None:
        target_intersections = rng.choice([0, 1, 2])

    blue, red = [], []
    n_intersections = -1
    for _ in range(5000):
        attempt_rng = Random(rng.randint(0, 2**31))
        blue = _generate_path(attempt_rng, grid_size)
        red = _generate_path(attempt_rng, grid_size)
        if not _paths_are_valid(blue, red):
            continue
        n_intersections = _count_intersections(blue, red)
        if n_intersections == target_intersections:
            break

    blue_str = " → ".join(f"({x:.1f}, {y:.1f})" for x, y in blue)
    red_str = " → ".join(f"({x:.1f}, {y:.1f})" for x, y in red)

    prompt = (
        f"A blue line goes through these points: {blue_str}\n"
        f"A red line goes through these points: {red_str}\n"
        f"Both lines are drawn on a coordinate grid from (0,0) to ({grid_size},{grid_size}).\n"
        f"How many times do the blue and red lines intersect? "
        f"Put your answer in curly brackets, e.g., {{2}}."
    )

    # Tiny placeholder image (API requires an image)
    img = Image.new("RGB", (64, 64), "white")

    ground_truth = str(n_intersections)
    metadata = {
        "grid_size": grid_size,
        "path_blue": blue,
        "path_red": red,
        "intersections": n_intersections,
        "target_intersections": target_intersections,
        "prompt": prompt,
        "mode": "text_only",
    }
    return img, ground_truth, metadata
