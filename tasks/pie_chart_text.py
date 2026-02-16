"""Task: Text-only pie chart — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

TASK_CONFIG = {
    "task_name": "pie_chart_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "mc4",
    "scorer": "exact_match",
    "default_params": {
        "n_slices": 5,
    },
    "sweep_axes": {
        "n_slices": [3, 4, 5, 6, 7],
    },
}

_SLICE_LABELS = ["Marketing", "Sales", "Engineering", "Support", "Admin", "R&D", "Operations"]

_call_counter = 0


def render(
    n_slices: int = 5,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    labels = _SLICE_LABELS[:n_slices]
    raw = [rng.randint(5, 40) for _ in range(n_slices)]
    total = sum(raw)
    percentages = [round(v / total * 100) for v in raw]
    # Adjust last to sum to 100
    percentages[-1] = 100 - sum(percentages[:-1])

    target_idx = rng.randint(0, n_slices - 1)
    target_label = labels[target_idx]
    target_pct = percentages[target_idx]

    distractors = set()
    while len(distractors) < 3:
        d = target_pct + rng.choice([-10, -5, -3, 3, 5, 10])
        if d != target_pct and 1 <= d <= 60:
            distractors.add(d)

    options = [target_pct] + sorted(distractors)
    rng.shuffle(options)
    correct_letter = chr(65 + options.index(target_pct))
    options_text = "\n".join(f"  {chr(65+i)}) {v}%" for i, v in enumerate(options))

    slices_str = ", ".join(f"{labels[i]}={percentages[i]}%" for i in range(n_slices))

    prompt = (
        f"Pie chart slices: {slices_str}\n\n"
        f"What percentage does \"{target_label}\" represent?\n"
        f"{options_text}\n\n"
        f"Answer with just the letter (A, B, C, or D)."
    )

    metadata = {
        "prompt": prompt,
        "n_slices": n_slices,
        "percentages": percentages,
        "labels": labels,
        "target_label": target_label,
        "target_pct": target_pct,
        "correct_letter": correct_letter,
        "mode": "text_only",
    }
    return placeholder_image(), correct_letter, metadata
