"""Task: Text-only strikethrough — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image
from tasks.strikethrough import _ITEM_LABELS

TASK_CONFIG = {
    "task_name": "strikethrough_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "csv_words",
    "scorer": "set_match",
    "default_params": {
        "n_words": 6,
        "n_struck": 2,
    },
    "sweep_axes": {
        "n_words": [5, 6, 8],
        "n_struck": [1, 2, 3],
    },
}

_call_counter = 0


def render(
    n_words: int = 6,
    n_struck: int = 2,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    words = rng.sample(_ITEM_LABELS, n_words)
    n_struck = min(n_struck, n_words)
    struck_indices = sorted(rng.sample(range(n_words), n_struck))
    labels = [chr(65 + i) for i in range(n_words)]

    word_list = ", ".join(f"{labels[i]}={words[i]}" for i in range(n_words))
    struck_labels = ", ".join(labels[i] for i in struck_indices)

    prompt = (
        f"Words with labels: {word_list}\n"
        f"Words {struck_labels} have a strikethrough line drawn through them.\n"
        f"Which words are struck through? List all struck-through words, "
        f"separated by commas. Put your answer in curly brackets, e.g., {{Revenue, Cost}}."
    )

    struck_words = [words[i] for i in struck_indices]
    ground_truth = ",".join(sorted(struck_words, key=str.lower))

    metadata = {
        "prompt": prompt,
        "words": words,
        "labels": labels,
        "struck_words": struck_words,
        "struck_indices": struck_indices,
        "mode": "text_only",
    }
    return placeholder_image(), ground_truth, metadata
