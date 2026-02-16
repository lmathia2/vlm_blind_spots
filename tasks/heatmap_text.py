"""Task: Text-only heatmap — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

TASK_CONFIG = {
    "task_name": "heatmap_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "mc4",
    "scorer": "exact_match",
    "default_params": {
        "grid_size": 4,
    },
    "sweep_axes": {
        "grid_size": [3, 4, 5, 6],
    },
}

_call_counter = 0


def render(
    grid_size: int = 4,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    row_labels = [chr(65 + i) for i in range(grid_size)]
    col_labels = [str(i + 1) for i in range(grid_size)]
    data = [[rng.randint(0, 100) for _ in range(grid_size)] for _ in range(grid_size)]

    target_r = rng.randint(0, grid_size - 1)
    target_c = rng.randint(0, grid_size - 1)
    target_value = data[target_r][target_c]

    distractors = set()
    while len(distractors) < 3:
        d = target_value + rng.choice([-20, -10, -5, 5, 10, 20])
        if d != target_value and 0 <= d <= 100:
            distractors.add(d)
    distractors = sorted(distractors)

    options = [target_value] + list(distractors)
    rng.shuffle(options)
    correct_letter = chr(65 + options.index(target_value))
    options_text = "\n".join(f"  {chr(65+i)}) {v}" for i, v in enumerate(options))

    # Format grid
    header = "     " + "".join(f"{c:>6}" for c in col_labels)
    grid_lines = [header]
    for r in range(grid_size):
        row_data = "".join(f"{data[r][c]:>6}" for c in range(grid_size))
        grid_lines.append(f"  {row_labels[r]}  {row_data}")
    grid_text = "\n".join(grid_lines)

    prompt = (
        f"Heatmap grid values:\n{grid_text}\n\n"
        f"What is the value at row {row_labels[target_r]}, column {col_labels[target_c]}?\n"
        f"{options_text}\n\n"
        f"Answer with just the letter (A, B, C, or D)."
    )

    metadata = {
        "prompt": prompt,
        "grid_size": grid_size,
        "data": data,
        "target_row_label": row_labels[target_r],
        "target_col_label": col_labels[target_c],
        "target_value": target_value,
        "correct_letter": correct_letter,
        "mode": "text_only",
    }
    return placeholder_image(), correct_letter, metadata
