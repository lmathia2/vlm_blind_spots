"""Task: Text-only scatter plot — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

TASK_CONFIG = {
    "task_name": "scatter_plot_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "mc4",
    "scorer": "exact_match",
    "default_params": {
        "n_points": 8,
        "n_series": 2,
    },
    "sweep_axes": {
        "n_points": [5, 8, 12],
        "n_series": [1, 2],
    },
}

_call_counter = 0


def render(
    n_points: int = 8,
    n_series: int = 2,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    series_names = ["Blue", "Red", "Green"][:n_series]
    all_points = {}
    for sname in series_names:
        pts = [(rng.randint(1, 20), rng.randint(1, 20)) for _ in range(n_points)]
        all_points[sname] = pts

    target_series = rng.choice(series_names)
    target_idx = rng.randint(0, n_points - 1)
    target_x, target_y = all_points[target_series][target_idx]

    # MC4 question about y-value at a specific x
    distractors = set()
    while len(distractors) < 3:
        d = target_y + rng.choice([-5, -3, -2, 2, 3, 5])
        if d != target_y and 0 < d <= 25:
            distractors.add(d)
    distractors = sorted(distractors)

    options = [target_y] + list(distractors)
    rng.shuffle(options)
    correct_letter = chr(65 + options.index(target_y))
    options_text = "\n".join(f"  {chr(65+i)}) {v}" for i, v in enumerate(options))

    data_desc = []
    for sname in series_names:
        pts_str = ", ".join(f"({x},{y})" for x, y in all_points[sname])
        data_desc.append(f"  {sname} series: {pts_str}")
    data_text = "\n".join(data_desc)

    prompt = (
        f"Scatter plot data points:\n{data_text}\n\n"
        f"In the {target_series} series, what is the y-value of the point at x={target_x}?\n"
        f"{options_text}\n\n"
        f"Answer with just the letter (A, B, C, or D)."
    )

    metadata = {
        "prompt": prompt,
        "n_points": n_points,
        "n_series": n_series,
        "target_series": target_series,
        "target_x": target_x,
        "target_y": target_y,
        "correct_letter": correct_letter,
        "mode": "text_only",
    }
    return placeholder_image(), correct_letter, metadata
