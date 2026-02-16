"""Task T2.6: Stacked bar segment reading (MC4)."""

from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mc4_utils import generate_distractors, format_mc4_prompt

TASK_CONFIG = {
    "task_name": "stacked_bar",
    "prompt_template": None,  # dynamic MC4 prompt
    "prompt_template_v2": None,
    "parser": "mc4",
    "scorer": "exact_match",
    "default_params": {
        "n_bars": 4,
        "n_segments": 3,
        "resolution": 512,
    },
    "sweep_axes": {
        "n_bars": [3, 4, 5, 6],
        "n_segments": [2, 3],
    },
}

_BAR_LABELS = ["2020", "2021", "2022", "2023", "2024", "2025"]
_SEGMENT_NAMES = ["Product A", "Product B", "Product C", "Product D"]
_SEGMENT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

_call_counter = 0


def render(
    n_bars: int = 4,
    n_segments: int = 3,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    bar_labels = _BAR_LABELS[:n_bars]
    segment_names = _SEGMENT_NAMES[:n_segments]
    colors = _SEGMENT_COLORS[:n_segments]

    # Generate segment values
    data = []
    for _ in range(n_bars):
        segments = [rng.randint(10, 50) for _ in range(n_segments)]
        data.append(segments)

    # Pick target
    target_bar_idx = rng.randint(0, n_bars - 1)
    target_seg_idx = rng.randint(0, n_segments - 1)
    target_bar = bar_labels[target_bar_idx]
    target_seg = segment_names[target_seg_idx]
    correct_value = data[target_bar_idx][target_seg_idx]

    # Distractors: total bar height, other segment values, cumulative heights
    other_values = []
    other_values.append(sum(data[target_bar_idx]))  # total bar height
    for si in range(n_segments):
        if si != target_seg_idx:
            other_values.append(data[target_bar_idx][si])
    # Cumulative up to target segment
    cum = sum(data[target_bar_idx][:target_seg_idx + 1])
    other_values.append(cum)

    distractors = generate_distractors(correct_value, other_values, n=3, rng=Random(rng.randint(0, 2**31)))
    question = f"In the '{target_bar}' bar, what is the value of the {target_seg} segment?"
    prompt, correct_letter = format_mc4_prompt(
        question, correct_value, distractors,
        rng=Random(rng.randint(0, 2**31)),
    )

    # Render chart
    dpi = 100
    fig_size = resolution / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)

    x = np.arange(n_bars)
    bottoms = np.zeros(n_bars)
    for si in range(n_segments):
        vals = [data[bi][si] for bi in range(n_bars)]
        ax.bar(x, vals, bottom=bottoms, color=colors[si], label=segment_names[si],
               edgecolor="black", linewidth=0.5)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel("Value")
    ax.set_title("Stacked Results")
    ax.legend()
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
        "n_segments": n_segments,
        "data": data,
        "target_bar": target_bar,
        "target_segment": target_seg,
        "target_value": correct_value,
        "correct_letter": correct_letter,
        "resolution": resolution,
    }
    return img, ground_truth, metadata
