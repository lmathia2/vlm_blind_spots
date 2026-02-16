"""Task: Text-only rotated text — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image
from tasks.rotated_text import _LABELS

TASK_CONFIG = {
    "task_name": "rotated_text_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {},
    "sweep_axes": {},
}

_call_counter = 0


def render(seed: int | None = None) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    text = rng.choice(_LABELS)

    prompt = (
        f"The text displayed is: {text}\n"
        f"What does the text say? "
        f"Put your answer in curly brackets, e.g., {{Revenue}}."
    )

    metadata = {
        "prompt": prompt,
        "text": text,
        "mode": "text_only",
    }
    return placeholder_image(), text, metadata
