"""Task: Text-only touching circles — perception vs reasoning diagnostic."""

import math
from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

TASK_CONFIG = {
    "task_name": "touching_circles_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "yes_no",
    "scorer": "exact_match",
    "default_params": {
        "distance": 0.0,
        "diameter": 0.2,
    },
    "sweep_axes": {
        "distance": [-0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25],
        "diameter": [0.08, 0.12, 0.15, 0.2],
    },
}

_call_counter = 0


def render(
    distance: float = 0.0,
    diameter: float = 0.2,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    radius = diameter / 2.0
    gap = 2 * radius * (1 + distance)

    # Place circles horizontally centered
    c1 = (0.5 - gap / 2, 0.5)
    c2 = (0.5 + gap / 2, 0.5)

    center_dist = math.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)

    prompt = (
        f"Circle 1: center ({c1[0]:.3f}, {c1[1]:.3f}), radius {radius:.3f}\n"
        f"Circle 2: center ({c2[0]:.3f}, {c2[1]:.3f}), radius {radius:.3f}\n"
        f"The distance between centers is {center_dist:.3f}.\n"
        f"The sum of their radii is {2*radius:.3f}.\n\n"
        f"Are the two circles touching or overlapping? Answer Yes or No."
    )

    ground_truth = "Yes" if distance <= 0 else "No"
    metadata = {
        "prompt": prompt,
        "distance": distance,
        "diameter": diameter,
        "center_dist": center_dist,
        "sum_radii": 2 * radius,
        "mode": "text_only",
    }
    return placeholder_image(), ground_truth, metadata
