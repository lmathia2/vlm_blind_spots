"""Task: Text-only highlighted text — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

_WORD_POOL = [
    "Revenue", "Expenses", "Profit", "Loss", "Budget", "Forecast",
    "Target", "Growth", "Margin", "Savings", "Overhead", "Surplus",
]

TASK_CONFIG = {
    "task_name": "highlighted_text_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "csv_words",
    "scorer": "set_match",
    "default_params": {
        "n_words": 6,
        "n_highlighted": 2,
        "highlight_color": "yellow",
    },
    "sweep_axes": {
        "n_words": [5, 6, 8],
        "n_highlighted": [1, 2, 3],
        "highlight_color": ["yellow", "cyan", "lightgreen"],
    },
}

_call_counter = 0


def render(
    n_words: int = 6,
    n_highlighted: int = 2,
    highlight_color: str = "yellow",
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    words = rng.sample(_WORD_POOL, min(n_words, len(_WORD_POOL)))
    n_highlighted = min(n_highlighted, len(words))
    highlighted_indices = sorted(rng.sample(range(len(words)), n_highlighted))
    highlighted_words = [words[i] for i in highlighted_indices]

    word_list = ", ".join(words)
    highlighted_list = ", ".join(highlighted_words)

    prompt = (
        f"Words displayed: {word_list}\n"
        f"The words {highlighted_list} are highlighted in {highlight_color}.\n"
        f"Which words are highlighted? List them separated by commas "
        f"in curly brackets, e.g., {{Revenue, Cost}}."
    )

    ground_truth = ",".join(sorted(highlighted_words, key=str.lower))

    metadata = {
        "prompt": prompt,
        "words": words,
        "highlighted_words": highlighted_words,
        "highlight_color": highlight_color,
        "mode": "text_only",
    }
    return placeholder_image(), ground_truth, metadata
