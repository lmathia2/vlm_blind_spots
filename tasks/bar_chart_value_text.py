"""Task: Text-only bar chart value — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

TASK_CONFIG = {
    "task_name": "bar_chart_value_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "mc4",
    "scorer": "exact_match",
    "default_params": {
        "n_bars": 5,
        "value_range_hi": 90,
    },
    "sweep_axes": {
        "n_bars": [3, 5, 7],
        "value_range_hi": [50, 90],
    },
}

_BAR_LABELS = ["Q1", "Q2", "Q3", "Q4", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]

_call_counter = 0


def render(
    n_bars: int = 5,
    value_range_hi: int = 90,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    labels = _BAR_LABELS[:n_bars]
    values = [rng.randint(10, value_range_hi) for _ in range(n_bars)]
    target_idx = rng.randint(0, n_bars - 1)
    target_label = labels[target_idx]
    target_value = values[target_idx]

    # Generate MC4 distractors
    distractors = set()
    while len(distractors) < 3:
        d = target_value + rng.choice([-15, -10, -5, 5, 10, 15])
        if d != target_value and 0 < d <= 100:
            distractors.add(d)
    distractors = sorted(distractors)

    options = [target_value] + distractors
    rng.shuffle(options)
    correct_letter = chr(65 + options.index(target_value))
    options_text = "\n".join(f"  {chr(65+i)}) {v}" for i, v in enumerate(options))

    bar_data = ", ".join(f"{labels[i]}={values[i]}" for i in range(n_bars))

    prompt = (
        f"Bar chart data: {bar_data}\n\n"
        f"What is the value of the \"{target_label}\" bar?\n"
        f"{options_text}\n\n"
        f"Answer with just the letter (A, B, C, or D)."
    )

    metadata = {
        "prompt": prompt,
        "n_bars": n_bars,
        "values": values,
        "target_label": target_label,
        "target_value": target_value,
        "correct_letter": correct_letter,
        "mode": "text_only",
    }
    return placeholder_image(), correct_letter, metadata
