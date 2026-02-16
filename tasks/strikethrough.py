"""Task T5.3: Strikethrough detection."""

from random import Random

from PIL import Image, ImageDraw, ImageFont

TASK_CONFIG = {
    "task_name": "strikethrough",
    "prompt_template": (
        "Which words are struck through? List all struck-through words, "
        "separated by commas. Put your answer in curly brackets, e.g., {Revenue, Cost}."
    ),
    "prompt_template_v2": (
        "Some words have a line drawn through them. Which ones? "
        "List them separated by commas in curly brackets, e.g., {Revenue, Cost}."
    ),
    "parser": "csv_words",
    "scorer": "set_match",
    "default_params": {
        "n_words": 6,
        "n_struck": 2,
        "line_thickness": 2,
        "font_size": 20,
        "resolution": 512,
    },
    "sweep_axes": {
        "n_words": [5, 6, 8],
        "n_struck": [1, 2, 3],
        "line_thickness": [1, 2, 3],
        "font_size": [12, 16, 20, 24],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_ITEM_LABELS = [
    "Revenue", "Expenses", "Profit", "Loss", "Budget", "Forecast",
    "Target", "Growth", "Margin", "Savings", "Overhead", "Surplus",
]

_call_counter = 0


def _load_font(size: int):
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render(
    n_words: int = 6,
    n_struck: int = 2,
    line_thickness: int = 2,
    font_size: int = 20,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    words = rng.sample(_ITEM_LABELS, n_words)
    struck_indices = sorted(rng.sample(range(n_words), min(n_struck, n_words)))
    labels = [chr(65 + i) for i in range(n_words)]  # A, B, C, ...

    font = _load_font(font_size)
    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)

    # Layout: vertical list with letter labels
    margin = resolution // 8
    spacing = min((resolution - margin * 2) // (n_words + 1), font_size * 3)
    y_start = (resolution - spacing * n_words) // 2

    for i, word in enumerate(words):
        label_text = f"{labels[i]}. {word}"
        y = y_start + spacing * i
        x = margin

        bbox = draw.textbbox((0, 0), label_text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x, y), label_text, fill="black", font=font)

        if i in struck_indices:
            # Draw strikethrough line through the word (not the label letter)
            word_bbox = draw.textbbox((0, 0), f"{labels[i]}. ", font=font)
            word_start_x = x + word_bbox[2] - word_bbox[0]
            strike_y = y + th // 2
            draw.line(
                [(word_start_x, strike_y), (x + tw, strike_y)],
                fill="red", width=line_thickness,
            )

    struck_words = [words[i] for i in struck_indices]
    ground_truth = ",".join(sorted(struck_words, key=str.lower))

    metadata = {
        "prompt": (
            "Which words are struck through? List all struck-through words, "
            "separated by commas. Put your answer in curly brackets, e.g., {Revenue, Cost}."
        ),
        "n_words": n_words,
        "n_struck": n_struck,
        "line_thickness": line_thickness,
        "font_size": font_size,
        "resolution": resolution,
        "words": words,
        "labels": labels,
        "struck_words": struck_words,
        "struck_indices": struck_indices,
    }
    return img, ground_truth, metadata
