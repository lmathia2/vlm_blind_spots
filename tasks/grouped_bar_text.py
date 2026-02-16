"""Task: Text-only grouped bar — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

TASK_CONFIG = {
    "task_name": "grouped_bar_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "mc4",
    "scorer": "exact_match",
    "default_params": {
        "n_groups": 4,
        "n_series": 2,
    },
    "sweep_axes": {
        "n_groups": [3, 4, 5],
        "n_series": [2, 3],
    },
}

_call_counter = 0


def render(
    n_groups: int = 4,
    n_series: int = 2,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    group_names = [f"Group {i+1}" for i in range(n_groups)]
    series_names = ["Blue", "Red", "Green"][:n_series]
    data = {}
    for gname in group_names:
        data[gname] = {s: rng.randint(10, 90) for s in series_names}

    target_group = rng.choice(group_names)
    target_series = rng.choice(series_names)
    target_value = data[target_group][target_series]

    distractors = set()
    while len(distractors) < 3:
        d = target_value + rng.choice([-15, -10, -5, 5, 10, 15])
        if d != target_value and 0 < d <= 100:
            distractors.add(d)

    options = [target_value] + sorted(distractors)
    rng.shuffle(options)
    correct_letter = chr(65 + options.index(target_value))
    options_text = "\n".join(f"  {chr(65+i)}) {v}" for i, v in enumerate(options))

    data_lines = []
    for gname in group_names:
        vals = ", ".join(f"{s}={data[gname][s]}" for s in series_names)
        data_lines.append(f"  {gname}: {vals}")
    data_text = "\n".join(data_lines)

    prompt = (
        f"Grouped bar chart data:\n{data_text}\n\n"
        f"What is the {target_series} value for {target_group}?\n"
        f"{options_text}\n\n"
        f"Answer with just the letter (A, B, C, or D)."
    )

    metadata = {
        "prompt": prompt,
        "n_groups": n_groups,
        "n_series": n_series,
        "data": data,
        "target_group": target_group,
        "target_series": target_series,
        "target_value": target_value,
        "correct_letter": correct_letter,
        "mode": "text_only",
    }
    return placeholder_image(), correct_letter, metadata
