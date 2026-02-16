"""Task: Scatter plot value reading (MC4)."""

from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mc4_utils import generate_distractors, format_mc4_prompt

TASK_CONFIG = {
    "task_name": "scatter_plot",
    "prompt_template": None,  # dynamic MC4 prompt
    "prompt_template_v2": None,
    "parser": "mc4",
    "scorer": "exact_match",
    "default_params": {
        "n_points": 8,
        "marker_size": 5,
        "n_series": 2,
        "resolution": 512,
    },
    "sweep_axes": {
        "n_points": [5, 8, 12],
        "marker_size": [3, 5, 8],
        "n_series": [1, 2],
    },
}

_SERIES_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
_SERIES_NAMES = ["Blue", "Red", "Green", "Orange"]
_MARKERS = ["o", "s", "^", "D"]

_call_counter = 0


def render(
    n_points: int = 8,
    marker_size: int = 5,
    n_series: int = 2,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    # Generate data for each series
    all_points = []  # (series_idx, x, y)
    for si in range(n_series):
        for _ in range(n_points):
            x = rng.randint(1, 50)
            y = rng.randint(5, 95)
            all_points.append((si, x, y))

    # Pick a target point
    target_idx = rng.randint(0, len(all_points) - 1)
    target_series, target_x, target_y = all_points[target_idx]
    series_name = _SERIES_NAMES[target_series]

    # Gather other y-values for distractors
    other_ys = [y for i, (_, _, y) in enumerate(all_points) if i != target_idx]
    distractors = generate_distractors(
        target_y, other_ys, n=3,
        rng=Random(rng.randint(0, 2**31)),
    )

    question = f"What is the approximate y-value of the {series_name} point at x={target_x}?"
    prompt, correct_letter = format_mc4_prompt(
        question, target_y, distractors,
        rng=Random(rng.randint(0, 2**31)),
    )

    # Render chart
    dpi = 100
    fig_size = resolution / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)

    for si in range(n_series):
        xs = [x for s, x, _ in all_points if s == si]
        ys = [y for s, _, y in all_points if s == si]
        ax.scatter(
            xs, ys,
            c=_SERIES_COLORS[si],
            marker=_MARKERS[si],
            s=marker_size ** 2 * 5,
            label=_SERIES_NAMES[si],
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Scatter Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = Image.frombytes("RGBA", (w, h), buf).convert("RGB")
    plt.close(fig)

    ground_truth = correct_letter
    metadata = {
        "prompt": prompt,
        "n_points": n_points,
        "marker_size": marker_size,
        "n_series": n_series,
        "target_series": series_name,
        "target_x": target_x,
        "target_y": target_y,
        "correct_letter": correct_letter,
        "distractors": distractors,
        "resolution": resolution,
    }
    return img, ground_truth, metadata
