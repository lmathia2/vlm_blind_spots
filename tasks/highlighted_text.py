"""Task: Highlighted text detection (csv_words, SET)."""

from random import Random

from PIL import Image, ImageDraw, ImageFont

TASK_CONFIG = {
    "task_name": "highlighted_text",
    "prompt_template": None,  # dynamic per sample
    "prompt_template_v2": None,
    "parser": "csv_words",
    "scorer": "set_match",
    "default_params": {
        "n_words": 6,
        "n_highlighted": 2,
        "font_size": 20,
        "highlight_color": "yellow",
        "resolution": 512,
    },
    "sweep_axes": {
        "n_words": [5, 6, 8],
        "n_highlighted": [1, 2, 3],
        "font_size": [14, 18, 24],
        "highlight_color": ["yellow", "cyan", "lightgreen"],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_WORD_POOL = [
    "Revenue", "Expenses", "Profit", "Growth", "Margin", "Budget",
    "Forecast", "Target", "Savings", "Surplus", "Overhead", "Deficit",
    "Income", "Capital", "Equity", "Assets", "Returns", "Yield",
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
    n_highlighted: int = 2,
    font_size: int = 20,
    highlight_color: str = "yellow",
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    words = rng.sample(_WORD_POOL, min(n_words, len(_WORD_POOL)))
    highlighted_indices = sorted(rng.sample(range(len(words)), min(n_highlighted, len(words))))

    font = _load_font(font_size)
    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)

    # Layout: words in a horizontal flow with wrapping
    margin = resolution // 10
    x = margin
    y = margin + resolution // 8
    line_height = int(font_size * 2.0)
    word_gap = int(font_size * 0.8)

    for i, word in enumerate(words):
        bbox = draw.textbbox((0, 0), word, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Wrap to next line if needed
        if x + tw > resolution - margin:
            x = margin
            y += line_height

        # Draw highlight background if this word is highlighted
        if i in highlighted_indices:
            pad = 3
            draw.rectangle(
                [(x - pad, y - pad), (x + tw + pad, y + th + pad)],
                fill=highlight_color,
            )

        draw.text((x, y), word, fill="black", font=font)
        x += tw + word_gap

    highlighted_words = [words[i] for i in highlighted_indices]
    ground_truth = ",".join(sorted(highlighted_words, key=str.lower))

    color_name = highlight_color
    prompt = (
        f"Which words are highlighted with a {color_name} background? "
        f"List them separated by commas in curly brackets, e.g., {{Revenue, Cost}}."
    )

    metadata = {
        "prompt": prompt,
        "n_words": n_words,
        "n_highlighted": n_highlighted,
        "font_size": font_size,
        "highlight_color": highlight_color,
        "resolution": resolution,
        "words": words,
        "highlighted_words": highlighted_words,
    }
    return img, ground_truth, metadata
