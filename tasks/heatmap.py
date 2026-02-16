"""Task: Heatmap cell value reading (MC4)."""

from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mc4_utils import generate_distractors, format_mc4_prompt

TASK_CONFIG = {
    "task_name": "heatmap",
    "prompt_template": None,  # dynamic MC4 prompt
    "prompt_template_v2": None,
    "parser": "mc4",
    "scorer": "exact_match",
    "default_params": {
        "grid_size": 4,
        "colormap": "Blues",
        "resolution": 512,
    },
    "sweep_axes": {
        "grid_size": [3, 4, 5, 6],
        "colormap": ["Blues", "YlOrRd", "viridis"],
    },
}

_ROW_LABELS = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
_COL_LABELS = ["Q1", "Q2", "Q3", "Q4", "H1", "H2"]

_call_counter = 0


def render(
    grid_size: int = 4,
    colormap: str = "Blues",
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    rows = _ROW_LABELS[:grid_size]
    cols = _COL_LABELS[:grid_size]

    # Generate grid values (0-100)
    data = [[rng.randint(0, 100) for _ in range(grid_size)] for _ in range(grid_size)]

    # Pick target cell
    target_row = rng.randint(0, grid_size - 1)
    target_col = rng.randint(0, grid_size - 1)
    correct_value = data[target_row][target_col]

    # Gather neighboring cell values for distractors
    other_values = []
    for r in range(grid_size):
        for c in range(grid_size):
            if r != target_row or c != target_col:
                other_values.append(data[r][c])

    distractors = generate_distractors(
        correct_value, other_values, n=3,
        rng=Random(rng.randint(0, 2**31)),
    )

    question = f"What is the approximate value in row '{rows[target_row]}', column '{cols[target_col]}'?"
    prompt, correct_letter = format_mc4_prompt(
        question, correct_value, distractors,
        rng=Random(rng.randint(0, 2**31)),
    )

    # Render heatmap
    dpi = 100
    fig_size = resolution / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)

    arr = np.array(data)
    im = ax.imshow(arr, cmap=colormap, vmin=0, vmax=100, aspect="equal")

    ax.set_xticks(range(grid_size))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(grid_size))
    ax.set_yticklabels(rows)
    ax.set_title("Performance Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = Image.frombytes("RGBA", (w, h), buf).convert("RGB")
    plt.close(fig)

    ground_truth = correct_letter
    metadata = {
        "prompt": prompt,
        "grid_size": grid_size,
        "colormap": colormap,
        "data": data,
        "target_row_label": rows[target_row],
        "target_col_label": cols[target_col],
        "target_value": correct_value,
        "correct_letter": correct_letter,
        "distractors": distractors,
        "resolution": resolution,
    }
    return img, ground_truth, metadata
