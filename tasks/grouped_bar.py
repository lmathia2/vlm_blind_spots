"""Task: Grouped bar chart value reading (MC4)."""

from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mc4_utils import generate_distractors, format_mc4_prompt

TASK_CONFIG = {
    "task_name": "grouped_bar",
    "prompt_template": None,  # dynamic MC4 prompt
    "prompt_template_v2": None,
    "parser": "mc4",
    "scorer": "exact_match",
    "default_params": {
        "n_groups": 4,
        "n_series": 2,
        "resolution": 512,
    },
    "sweep_axes": {
        "n_groups": [3, 4, 5],
        "n_series": [2, 3],
    },
}

_GROUP_LABELS = ["Q1", "Q2", "Q3", "Q4", "H1", "H2"]
_SERIES_NAMES = ["Product A", "Product B", "Product C"]
_SERIES_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
_SERIES_DISPLAY = ["Blue", "Orange", "Green"]

_call_counter = 0


def render(
    n_groups: int = 4,
    n_series: int = 2,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    group_labels = _GROUP_LABELS[:n_groups]
    series_names = _SERIES_NAMES[:n_series]
    colors = _SERIES_COLORS[:n_series]
    display_colors = _SERIES_DISPLAY[:n_series]

    # Generate values for each group × series
    data = []
    for _ in range(n_groups):
        row = [rng.randint(10, 90) for _ in range(n_series)]
        data.append(row)

    # Pick target
    target_group_idx = rng.randint(0, n_groups - 1)
    target_series_idx = rng.randint(0, n_series - 1)
    target_group = group_labels[target_group_idx]
    target_color = display_colors[target_series_idx]
    correct_value = data[target_group_idx][target_series_idx]

    # Gather other bar values for distractors
    other_values = []
    for gi in range(n_groups):
        for si in range(n_series):
            if gi != target_group_idx or si != target_series_idx:
                other_values.append(data[gi][si])

    distractors = generate_distractors(
        correct_value, other_values, n=3,
        rng=Random(rng.randint(0, 2**31)),
    )

    question = f"What is the approximate value of the {target_color} bar in the '{target_group}' group?"
    prompt, correct_letter = format_mc4_prompt(
        question, correct_value, distractors,
        rng=Random(rng.randint(0, 2**31)),
    )

    # Render grouped bar chart
    dpi = 100
    fig_size = resolution / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)

    x = np.arange(n_groups)
    bar_width = 0.8 / n_series
    offsets = np.linspace(-(n_series - 1) * bar_width / 2, (n_series - 1) * bar_width / 2, n_series)

    for si in range(n_series):
        vals = [data[gi][si] for gi in range(n_groups)]
        ax.bar(
            x + offsets[si], vals, bar_width,
            color=colors[si], label=series_names[si],
            edgecolor="black", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Value")
    ax.set_title("Grouped Comparison")
    ax.legend()
    ax.set_ylim(0, 100)
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = Image.frombytes("RGBA", (w, h), buf).convert("RGB")
    plt.close(fig)

    ground_truth = correct_letter
    metadata = {
        "prompt": prompt,
        "n_groups": n_groups,
        "n_series": n_series,
        "data": data,
        "target_group": target_group,
        "target_series": series_names[target_series_idx],
        "target_color": target_color,
        "target_value": correct_value,
        "correct_letter": correct_letter,
        "distractors": distractors,
        "resolution": resolution,
    }
    return img, ground_truth, metadata
