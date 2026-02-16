"""Task: Text-only arrow annotation — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image
from tasks.arrow_annotation import _WORDS

TASK_CONFIG = {
    "task_name": "arrow_annotation_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "n_words": 4,
    },
    "sweep_axes": {
        "n_words": [3, 4, 5, 6],
    },
}

_call_counter = 0


def render(
    n_words: int = 4,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    words = rng.sample(_WORDS, n_words)
    target_idx = rng.randint(0, n_words - 1)
    target_word = words[target_idx]

    word_list = ", ".join(f"{i+1}. {w}" for i, w in enumerate(words))

    prompt = (
        f"Words arranged vertically: {word_list}\n"
        f"A red arrow points at word {target_idx + 1} ({target_word}).\n\n"
        f"What word does the red arrow point to? "
        f"Put your answer in curly brackets, e.g., {{Revenue}}."
    )

    metadata = {
        "prompt": prompt,
        "words": words,
        "target_word": target_word,
        "target_index": target_idx,
        "mode": "text_only",
    }
    return placeholder_image(), target_word, metadata
