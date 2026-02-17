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

_MIN_GAP = 8  # minimum percentage-point gap between any two slices

_call_counter = 0


def _generate_slices(rng: Random, n_slices: int, max_attempts: int = 200) -> list[int]:
    """Generate n_slices percentages summing to 100, with ≥ _MIN_GAP pp between any pair."""
    for _ in range(max_attempts):
        raw = [rng.randint(5, 40) for _ in range(n_slices)]
        total = sum(raw)
        percentages = [round(v / total * 100) for v in raw]
        diff = 100 - sum(percentages)
        percentages[0] += diff

        # Check minimum gap between all pairs
        ok = True
        for i in range(len(percentages)):
            for j in range(i + 1, len(percentages)):
                if abs(percentages[i] - percentages[j]) < _MIN_GAP:
                    ok = False
                    break
            if not ok:
                break
        if ok and all(p >= 3 for p in percentages):
            return percentages

    # Fallback: reduce gap requirement to 6
    for _ in range(max_attempts):
        raw = [rng.randint(5, 40) for _ in range(n_slices)]
        total = sum(raw)
        percentages = [round(v / total * 100) for v in raw]
        diff = 100 - sum(percentages)
        percentages[0] += diff
        ok = True
        for i in range(len(percentages)):
            for j in range(i + 1, len(percentages)):
                if abs(percentages[i] - percentages[j]) < 6:
                    ok = False
                    break
            if not ok:
                break
        if ok and all(p >= 3 for p in percentages):
            return percentages

    return percentages  # last attempt, best-effort


def render(
    n_slices: int = 5,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    labels = _SLICE_LABELS[:n_slices]

    # Generate well-spaced percentages
    percentages = _generate_slices(rng, n_slices)

    target_idx = rng.randint(0, n_slices - 1)
    correct_pct = percentages[target_idx]

    # Distractors: prefer actual slice percentages (guaranteed well-spaced after _generate_slices)
    other_pcts = [p for i, p in enumerate(percentages) if i != target_idx]
    distractors = []

    # Use actual slice values first (already ≥ _MIN_GAP apart from each other and target)
    rng.shuffle(other_pcts)
    for c in other_pcts:
        if len(distractors) >= 3:
            break
        if all(abs(c - d) >= 7 for d in distractors) and abs(c - correct_pct) >= 7:
            distractors.append(c)

    # Fill remaining with synthetic offsets only if needed
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
