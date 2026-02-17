"""Task: Count nested squares in an image."""

import io
import math
from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

TASK_CONFIG = {
    "task_name": "nested_squares",
    "prompt_template": (
        "Count total number of squares in the image. "
        "Answer with only the number in numerical format in curly brackets e.g. {3}."
    ),
    "prompt_template_v2": (
        "How many squares are shown in total, including squares inside other squares? "
        "Answer in curly brackets, e.g., {3}."
    ),
    "parser": "integer",
    "scorer": "exact_match",
    "default_params": {
        "depth": 3,
        "resolution": 512,
        "line_thickness": 2,
        "reduction_factor": 0.6,
    },
    "sweep_axes": {
        "depth": [2, 3, 4, 5, 6, 7, 8],
        "reduction_factor": [0.4, 0.5, 0.6, 0.7, 0.8],
        "line_thickness": [1, 2, 3],
    },
}

_call_counter = 0


def render(depth: int = 3, resolution: int = 512, line_thickness: int = 2,
           reduction_factor: float = 0.6) -> tuple[Image.Image, str, dict]:
    """Render nested squares with random center offsets."""
    global _call_counter
    _call_counter += 1
    rng = Random(_call_counter)

    # Cap depth so all squares are visible (smallest side >= 5px)
    min_square_px = 5
    effective_depth = depth
    size_check = 0.85
    for d in range(1, depth + 1):
        if size_check * resolution < min_square_px:
            effective_depth = d - 1
            break
        size_check *= reduction_factor
    effective_depth = max(1, effective_depth)

    dpi = 100
    fig_size = resolution / dpi
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    lw = line_thickness
    size = 0.85  # initial square size as fraction of canvas
    cx, cy = 0.5, 0.5

    for i in range(effective_depth):
        half = size / 2
        rect = patches.Rectangle(
            (cx - half, cy - half), size, size,
            linewidth=lw, edgecolor="black", facecolor="none"
        )
        ax.add_patch(rect)

        # Shrink and offset for next square
        size *= reduction_factor
        max_offset = (1 - size) / 2 - 0.05  # keep within bounds
        # Small random offset, constrained to stay inside parent
        parent_half = half * reduction_factor / reduction_factor
        max_jitter = min(0.03, parent_half * 0.2)
        cx += rng.uniform(-max_jitter, max_jitter)
        cy += rng.uniform(-max_jitter, max_jitter)
        # Clamp to keep square visible
        half_new = size / 2
        cx = max(half_new + 0.02, min(1 - half_new - 0.02, cx))
        cy = max(half_new + 0.02, min(1 - half_new - 0.02, cy))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    ground_truth = str(effective_depth)
    metadata = {
        "depth": effective_depth,
        "requested_depth": depth,
        "resolution": resolution,
        "line_thickness": line_thickness,
        "reduction_factor": reduction_factor,
    }
    return img, ground_truth, metadata
