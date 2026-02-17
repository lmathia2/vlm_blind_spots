"""Task: Text-only line chart crossing — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image
from tasks.line_chart_crossing import _build_curves, _count_sign_changes

import numpy as np

TASK_CONFIG = {
    "task_name": "line_chart_crossing_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "integer",
    "scorer": "exact_match",
    "default_params": {
        "n_points": 20,
        "target_crossings": None,
    },
    "sweep_axes": {
        "target_crossings": [0, 1, 2, 3],
    },
}

_call_counter = 0


def render(
    n_points: int = 20,
    target_crossings: int | None = None,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    if target_crossings is None:
        target_crossings = rng.choice([0, 1, 2, 3])

    # Use parent's curve generation
    best_y1, best_y2 = None, None
    actual_crossings = -1
    for _ in range(2000):
        attempt_rng = Random(rng.randint(0, 2**31))
        x, y1, y2 = _build_curves(attempt_rng, n_points, target_crossings)
        diff = y1 - y2
        crossings = _count_sign_changes(diff)
        if crossings == target_crossings:
            best_y1, best_y2 = y1, y2
            actual_crossings = crossings
            break

    if best_y1 is None:
        best_y1, best_y2 = y1, y2
        actual_crossings = crossings

    # Sample ~10 representative points for text description
    step = max(1, n_points // 10)
    indices = list(range(0, n_points, step))
    if indices[-1] != n_points - 1:
        indices.append(n_points - 1)

    a_pts = ", ".join(f"({i}, {best_y1[i]:.1f})" for i in indices)
    b_pts = ", ".join(f"({i}, {best_y2[i]:.1f})" for i in indices)

    prompt = (
        f"Two line series are plotted:\n"
        f"  Series A (Revenue, blue) data points: {a_pts}\n"
        f"  Series B (Cost, red) data points: {b_pts}\n\n"
        f"How many times do the two lines cross each other? "
        f"Answer with a number in curly brackets, e.g., {{2}}."
    )

    ground_truth = str(actual_crossings)
    metadata = {
        "prompt": prompt,
        "n_points": n_points,
        "target_crossings": target_crossings,
        "actual_crossings": actual_crossings,
        "mode": "text_only",
    }
    return placeholder_image(), ground_truth, metadata
