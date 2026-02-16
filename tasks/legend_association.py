"""Task T2.4: Legend-data association — which series has the highest peak."""

from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

TASK_CONFIG = {
    "task_name": "legend_association",
    "prompt_template": None,  # dynamic per sample
    "prompt_template_v2": None,
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "n_series": 3,
        "color_mode": "distinct",
        "legend_position": "upper right",
        "resolution": 512,
    },
    "sweep_axes": {
        "n_series": [2, 3, 4],
        "color_mode": ["distinct", "similar"],
        "legend_position": ["upper right", "lower right", "center left"],
    },
}

_SERIES_NAMES = ["Revenue", "Cost", "Profit", "Tax", "Expenses", "Savings"]
_DISTINCT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
_SIMILAR_COLORS = ["#1f77b4", "#4a9bd9", "#2ca02c", "#5cb85c", "#17a2b8", "#20c997"]

_call_counter = 0


def render(
    n_series: int = 3,
    color_mode: str = "distinct",
    legend_position: str = "upper right",
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)
    np_rng = np.random.RandomState(seed if seed is not None else _call_counter)

    names = _SERIES_NAMES[:n_series]
    colors = (_DISTINCT_COLORS if color_mode == "distinct" else _SIMILAR_COLORS)[:n_series]

    n_points = rng.randint(8, 15)
    x = np.arange(2015, 2015 + n_points)

    # Generate series data with one clear peak winner
    peaks = []
    all_data = []
    for i in range(n_series):
        base = np_rng.uniform(20, 60)
        data = base + np.cumsum(np_rng.normal(0, 3, n_points))
        # Add a peak
        peak_idx = rng.randint(2, n_points - 3)
        peak_boost = np_rng.uniform(15, 35)
        data[peak_idx] += peak_boost
        peaks.append(float(np.max(data)))
        all_data.append(data)

    # Ensure one clear winner by boosting the designated winner
    winner_idx = rng.randint(0, n_series - 1)
    gap = max(peaks) - peaks[winner_idx] + np_rng.uniform(5, 15)
    peak_pos = int(np.argmax(all_data[winner_idx]))
    all_data[winner_idx][peak_pos] += gap

    # Render chart
    dpi = 100
    fig_size = resolution / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)

    for i in range(n_series):
        ax.plot(x, all_data[i], color=colors[i], label=names[i], linewidth=2, marker="o", markersize=3)

    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.set_title("Multi-Series Comparison")
    ax.legend(loc=legend_position)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = Image.frombytes("RGBA", (w, h), buf).convert("RGB")
    plt.close(fig)

    ground_truth = names[winner_idx]
    series_names_str = ", ".join(names[:-1]) + f", or {names[-1]}" if n_series > 2 else " or ".join(names)
    prompt = (
        f"Which series has the highest peak — {series_names_str}? "
        f"Put your answer in curly brackets, e.g., {{Revenue}}."
    )

    metadata = {
        "prompt": prompt,
        "n_series": n_series,
        "color_mode": color_mode,
        "legend_position": legend_position,
        "resolution": resolution,
        "winner": ground_truth,
        "series_names": names,
    }
    return img, ground_truth, metadata
