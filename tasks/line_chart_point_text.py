"""Task: Text-only line chart point — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

TASK_CONFIG = {
    "task_name": "line_chart_point_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "mc4",
    "scorer": "exact_match",
    "default_params": {
        "n_points": 8,
    },
    "sweep_axes": {
        "n_points": [5, 8, 12, 15],
    },
}

_call_counter = 0


def render(
    n_points: int = 8,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    x_vals = list(range(2020, 2020 + n_points))
    y_vals = [rng.randint(10, 90) for _ in range(n_points)]
    target_idx = rng.randint(0, n_points - 1)
    target_x = x_vals[target_idx]
    target_y = y_vals[target_idx]

    distractors = set()
    while len(distractors) < 3:
        d = target_y + rng.choice([-10, -5, -3, 3, 5, 10])
        if d != target_y and 0 < d <= 100:
            distractors.add(d)

    options = [target_y] + sorted(distractors)
    rng.shuffle(options)
    correct_letter = chr(65 + options.index(target_y))
    options_text = "\n".join(f"  {chr(65+i)}) {v}" for i, v in enumerate(options))

    pts_str = ", ".join(f"({x},{y})" for x, y in zip(x_vals, y_vals))

    prompt = (
        f"Line chart data points: {pts_str}\n\n"
        f"What is the y-value at x={target_x}?\n"
        f"{options_text}\n\n"
        f"Answer with just the letter (A, B, C, or D)."
    )

    metadata = {
        "prompt": prompt,
        "n_points": n_points,
        "target_x": target_x,
        "target_y": target_y,
        "all_x": x_vals,
        "all_y": y_vals,
        "correct_letter": correct_letter,
        "mode": "text_only",
    }
    return placeholder_image(), correct_letter, metadata
