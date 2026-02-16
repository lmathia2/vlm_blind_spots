"""Task T2.2: Bar chart value reading (MC4)."""

from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mc4_utils import generate_distractors, format_mc4_prompt

TASK_CONFIG = {
    "task_name": "bar_chart_value",
    "prompt_template": None,  # dynamic MC4 prompt
    "prompt_template_v2": None,
    "parser": "mc4",
    "scorer": "exact_match",
    "default_params": {
        "n_bars": 5,
        "value_range_lo": 10,
        "value_range_hi": 90,
        "resolution": 512,
    },
    "sweep_axes": {
        "n_bars": [3, 5, 7],
        "value_range_hi": [50, 90],
    },
}

_BAR_LABELS = ["Q1", "Q2", "Q3", "Q4", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_call_counter = 0


def render(
    n_bars: int = 5,
    value_range_lo: int = 10,
    value_range_hi: int = 90,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    labels = _BAR_LABELS[:n_bars]
    values = [rng.randint(value_range_lo, value_range_hi) for _ in range(n_bars)]
    target_idx = rng.randint(0, n_bars - 1)
    target_label = labels[target_idx]
    correct_value = values[target_idx]

    other_values = [v for i, v in enumerate(values) if i != target_idx]
    distractors = generate_distractors(correct_value, other_values, n=3, rng=Random(rng.randint(0, 2**31)))

    question = f"What is the value of the bar labeled '{target_label}'?"
    prompt, correct_letter = format_mc4_prompt(
        question, correct_value, distractors,
        rng=Random(rng.randint(0, 2**31)),
    )

    # Render chart
    dpi = 100
    fig_size = resolution / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.bar(labels, values, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Value")
    ax.set_title("Quarterly Results")
    ax.set_ylim(0, value_range_hi + 10)
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = Image.frombytes("RGBA", (w, h), buf).convert("RGB")
    plt.close(fig)

    ground_truth = correct_letter
    metadata = {
        "prompt": prompt,
        "n_bars": n_bars,
        "values": values,
        "target_label": target_label,
        "target_value": correct_value,
        "correct_letter": correct_letter,
        "distractors": distractors,
        "resolution": resolution,
    }
    return img, ground_truth, metadata
