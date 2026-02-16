"""Task T2.5: Pie chart relative comparison (MC4)."""

from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mc4_utils import format_mc4_prompt

TASK_CONFIG = {
    "task_name": "pie_chart",
    "prompt_template": None,  # dynamic MC4 prompt
    "prompt_template_v2": None,
    "parser": "mc4",
    "scorer": "exact_match",
    "default_params": {
        "n_slices": 5,
        "resolution": 512,
    },
    "sweep_axes": {
        "n_slices": [3, 4, 5, 6, 7],
    },
}

_SLICE_LABELS = ["Marketing", "Engineering", "Sales", "Operations", "HR",
                 "Finance", "Support", "R&D", "Legal", "Admin"]

_call_counter = 0


def render(
    n_slices: int = 5,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    labels = _SLICE_LABELS[:n_slices]

    # Generate percentages that sum to 100, with minimum 5% each
    raw = [rng.randint(5, 40) for _ in range(n_slices)]
    total = sum(raw)
    percentages = [round(v / total * 100) for v in raw]
    # Fix rounding to sum to 100
    diff = 100 - sum(percentages)
    percentages[0] += diff

    target_idx = rng.randint(0, n_slices - 1)
    correct_pct = percentages[target_idx]

    # Generate distractors: other slice percentages + offsets, spaced ≥7% apart
    other_pcts = [p for i, p in enumerate(percentages) if i != target_idx]
    distractors = []
    candidates = sorted(set(other_pcts), key=lambda x: abs(x - correct_pct))

    for c in candidates:
        if len(distractors) >= 3:
            break
        if all(abs(c - d) >= 7 for d in distractors) and abs(c - correct_pct) >= 7:
            distractors.append(c)

    # Fill remaining with offsets
    for offset in [12, -12, 20, -20, 8, -8, 15, -15]:
        if len(distractors) >= 3:
            break
        v = correct_pct + offset
        if 1 <= v <= 80 and v != correct_pct and all(abs(v - d) >= 7 for d in distractors) and abs(v - correct_pct) >= 7:
            distractors.append(v)

    question = f"What approximate percentage does the '{labels[target_idx]}' slice represent?"
    prompt, correct_letter = format_mc4_prompt(
        question, f"{correct_pct}%", [f"{d}%" for d in distractors[:3]],
        rng=Random(rng.randint(0, 2**31)),
    )

    # Render chart
    dpi = 100
    fig_size = resolution / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    colors = plt.cm.Set3(np.linspace(0, 1, n_slices))
    ax.pie(percentages, labels=labels, colors=colors, startangle=rng.randint(0, 360))
    ax.set_title("Budget Allocation")
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = Image.frombytes("RGBA", (w, h), buf).convert("RGB")
    plt.close(fig)

    ground_truth = correct_letter
    metadata = {
        "prompt": prompt,
        "n_slices": n_slices,
        "percentages": percentages,
        "labels": labels,
        "target_label": labels[target_idx],
        "target_pct": correct_pct,
        "correct_letter": correct_letter,
        "resolution": resolution,
    }
    return img, ground_truth, metadata
