"""Task T2.3: Line chart point value reading (MC4)."""

from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mc4_utils import generate_distractors, format_mc4_prompt

TASK_CONFIG = {
    "task_name": "line_chart_point",
    "prompt_template": None,  # dynamic MC4 prompt
    "prompt_template_v2": None,
    "parser": "mc4",
    "scorer": "exact_match",
    "default_params": {
        "n_points": 8,
        "marker_size": 6,
        "gridlines": True,
        "resolution": 512,
    },
    "sweep_axes": {
        "n_points": [5, 8, 12, 15],
        "marker_size": [3, 5, 6, 8],
        "gridlines": [True, False],
    },
}

_call_counter = 0


def render(
    n_points: int = 8,
    marker_size: int = 6,
    gridlines: bool = True,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)
    np_rng = np.random.RandomState(seed if seed is not None else _call_counter)

    # Generate data points
    start_year = rng.choice([2010, 2012, 2015, 2016])
    x = list(range(start_year, start_year + n_points))
    y = np.cumsum(np_rng.normal(5, 8, n_points)) + np_rng.uniform(20, 50)
    y = np.round(y, 1)

    # Pick a target point (not first or last for better visibility)
    target_idx = rng.randint(1, n_points - 2)
    target_x = x[target_idx]
    correct_value = float(y[target_idx])

    # Distractors from neighboring points
    neighbor_values = [float(y[i]) for i in range(n_points) if i != target_idx]
    distractors = generate_distractors(
        round(correct_value, 1),
        [round(v, 1) for v in neighbor_values],
        n=3, rng=Random(rng.randint(0, 2**31)),
    )

    question = f"What is the y-value at the data point marked at x={target_x}?"
    prompt, correct_letter = format_mc4_prompt(
        question, round(correct_value, 1), [round(d, 1) for d in distractors],
        rng=Random(rng.randint(0, 2**31)),
    )

    # Render chart
    dpi = 100
    fig_size = resolution / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.plot(x, y, "b-o", markersize=marker_size, linewidth=1.5, label="Metric")
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.set_title("Trend Analysis")
    if gridlines:
        ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.tick_params(axis="x", rotation=45)
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
        "gridlines": gridlines,
        "resolution": resolution,
        "target_x": target_x,
        "target_y": round(correct_value, 1),
        "correct_letter": correct_letter,
        "all_x": x,
        "all_y": [round(float(v), 1) for v in y],
    }
    return img, ground_truth, metadata
