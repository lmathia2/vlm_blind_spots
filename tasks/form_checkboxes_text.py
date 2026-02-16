"""Task: Text-only form checkboxes — perception vs reasoning diagnostic."""

import string
from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

TASK_CONFIG = {
    "task_name": "form_checkboxes_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "csv_letters",
    "scorer": "set_match",
    "default_params": {
        "n_options": 5,
        "n_checked": 2,
    },
    "sweep_axes": {
        "n_options": [4, 6, 8],
        "n_checked": [1, 2, 3, 4],
    },
}

_call_counter = 0


def render(
    n_options: int = 5,
    n_checked: int = 2,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    n_checked = min(n_checked, n_options)
    letters = list(string.ascii_uppercase[:n_options])
    checked_set = set(rng.sample(letters, n_checked))

    checkbox_lines = []
    for letter in letters:
        state = "checked" if letter in checked_set else "unchecked"
        checkbox_lines.append(f"  [{state}] Option {letter}")
    checkboxes_text = "\n".join(checkbox_lines)

    prompt = (
        f"Checkboxes:\n{checkboxes_text}\n\n"
        f"Which options are checked? List only the letters of the "
        f"checked options, separated by commas, in curly brackets. "
        f"For example: {{A, C, E}}."
    )

    ground_truth = ",".join(sorted(checked_set))
    metadata = {
        "prompt": prompt,
        "n_options": n_options,
        "n_checked": n_checked,
        "checked_letters": sorted(checked_set),
        "all_options": letters,
        "mode": "text_only",
    }
    return placeholder_image(), ground_truth, metadata
