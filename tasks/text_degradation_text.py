"""Task: Text-only text degradation — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image
from tasks.text_degradation import _TEXT_STRINGS

TASK_CONFIG = {
    "task_name": "text_degradation_text",
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

    text = rng.choice(_TEXT_STRINGS)

    prompt = (
        f"The text displayed is: {text}\n"
        f"What does the text say? "
        f"Put your answer in curly brackets, e.g., {{Total: $500}}."
    )

    metadata = {
        "prompt": prompt,
        "text": text,
        "mode": "text_only",
    }
    return placeholder_image(), text, metadata
