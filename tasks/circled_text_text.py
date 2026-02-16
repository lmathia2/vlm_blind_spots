"""Task: Text-only circled text — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image
from tasks.circled_text import _SENTENCES

TASK_CONFIG = {
    "task_name": "circled_text_text",
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

    sentence = rng.choice(_SENTENCES)
    words = sentence.split()
    target_idx = rng.randint(0, len(words) - 1)
    target_word = words[target_idx]

    prompt = (
        f"Sentence: \"{sentence}\"\n"
        f"The word \"{target_word}\" has a red circle drawn around it.\n"
        f"Which word is circled? "
        f"Put your answer in curly brackets, e.g., {{Revenue}}."
    )

    metadata = {
        "prompt": prompt,
        "sentence": sentence,
        "target_word": target_word,
        "target_index": target_idx,
        "mode": "text_only",
    }
    return placeholder_image(), target_word, metadata
