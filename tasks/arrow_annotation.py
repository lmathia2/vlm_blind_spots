"""Task T5.2: Arrow annotation target identification."""

from random import Random

from PIL import Image, ImageDraw, ImageFont

TASK_CONFIG = {
    "task_name": "arrow_annotation",
    "prompt_template": (
        "What word does the red arrow point to? "
        "Put your answer in curly brackets, e.g., {Revenue}."
    ),
    "prompt_template_v2": (
        "A red arrow points to one of the words. Which word? "
        "Put your answer in curly brackets, e.g., {Revenue}."
    ),
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "n_words": 4,
        "arrow_width": 2,
        "resolution": 512,
    },
    "sweep_axes": {
        "n_words": [3, 4, 5, 6],
        "arrow_width": [1, 2, 3, 4],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_WORDS = [
    "Revenue", "Cost", "Profit", "Loss", "Budget", "Forecast",
    "Target", "Actual", "Variance", "Margin", "Growth", "Decline",
    "Subtotal", "Total", "Pending", "Approved",
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
    n_words: int = 4,
    arrow_width: int = 2,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    words = rng.sample(_WORDS, n_words)
    target_idx = rng.randint(0, n_words - 1)

    font = _load_font(max(14, resolution // 30))
    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)

    # Place words vertically spaced
    margin = resolution // 8
    spacing = (resolution - margin * 2) // (n_words + 1)
    word_positions = []

    for i, word in enumerate(words):
        x = resolution // 2
        y = margin + spacing * (i + 1)
        bbox = draw.textbbox((0, 0), word, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x - tw // 2
        ty = y - th // 2
        draw.text((tx, ty), word, fill="black", font=font)
        word_positions.append((tx, ty, tx + tw, ty + th))

    # Draw red arrow from left margin to target word
    target_box = word_positions[target_idx]
    arrow_start_x = margin // 2
    arrow_y = (target_box[1] + target_box[3]) // 2
    arrow_end_x = target_box[0] - 10

    import math
    # Arrow line
    draw.line([(arrow_start_x, arrow_y), (arrow_end_x, arrow_y)], fill="red", width=arrow_width)
    # Arrowhead
    head_len = 12
    for da in [0.4, -0.4]:
        ax = arrow_end_x - int(head_len * math.cos(da))
        ay = arrow_y - int(head_len * math.sin(da))
        draw.line([(arrow_end_x, arrow_y), (ax, ay)], fill="red", width=arrow_width)

    ground_truth = words[target_idx]
    metadata = {
        "n_words": n_words,
        "arrow_width": arrow_width,
        "resolution": resolution,
        "words": words,
        "target_word": ground_truth,
        "target_index": target_idx,
    }
    return img, ground_truth, metadata
