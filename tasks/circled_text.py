"""Task T5.1: Circled text identification."""

from random import Random

from PIL import Image, ImageDraw, ImageFont

TASK_CONFIG = {
    "task_name": "circled_text",
    "prompt_template": (
        "Which word in this image is circled? "
        "Put your answer in curly brackets, e.g., {Revenue}."
    ),
    "prompt_template_v2": (
        "A red circle/ellipse is drawn around one word. Which word is it? "
        "Put your answer in curly brackets, e.g., {Revenue}."
    ),
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "font_size": 24,
        "ellipse_thickness": 2,
        "resolution": 512,
    },
    "sweep_axes": {
        "font_size": [14, 18, 24, 36],
        "ellipse_thickness": [1, 2, 3, 4],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_SENTENCES = [
    "The quarterly revenue exceeded expectations by a significant margin",
    "Please review the attached invoice before the payment deadline",
    "Total expenses were reduced through careful budget management",
    "The marketing campaign generated substantial customer engagement",
    "Annual performance metrics showed consistent growth across divisions",
    "Inventory levels should be adjusted based on seasonal demand patterns",
    "The compliance report requires immediate attention from management",
    "Operating costs remained stable despite increased production volume",
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
    font_size: int = 24,
    ellipse_thickness: int = 2,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    sentence = rng.choice(_SENTENCES)
    words = sentence.split()
    target_idx = rng.randint(0, len(words) - 1)
    target_word = words[target_idx]

    font = _load_font(font_size)
    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)

    # Compute word positions by rendering word-by-word
    # Wrap text into lines that fit
    padding = 20
    max_width = resolution - padding * 2
    lines: list[list[tuple[str, int]]] = []  # list of (word, word_index)
    current_line: list[tuple[str, int]] = []
    current_width = 0
    space_w = draw.textbbox((0, 0), " ", font=font)[2]

    for wi, word in enumerate(words):
        bbox = draw.textbbox((0, 0), word, font=font)
        ww = bbox[2] - bbox[0]
        new_width = current_width + (space_w if current_line else 0) + ww
        if new_width > max_width and current_line:
            lines.append(current_line)
            current_line = [(word, wi)]
            current_width = ww
        else:
            current_line.append((word, wi))
            current_width = new_width
    if current_line:
        lines.append(current_line)

    line_height = font_size + 8
    total_text_height = line_height * len(lines)
    y_start = (resolution - total_text_height) // 2

    # Draw all words and track target position
    target_bbox = None
    for li, line in enumerate(lines):
        # Compute line width for centering
        line_text = " ".join(w for w, _ in line)
        lbbox = draw.textbbox((0, 0), line_text, font=font)
        line_w = lbbox[2] - lbbox[0]
        x = (resolution - line_w) // 2

        for word, wi in line:
            bbox = draw.textbbox((0, 0), word, font=font)
            ww = bbox[2] - bbox[0]
            wh = bbox[3] - bbox[1]
            y = y_start + li * line_height
            draw.text((x - bbox[0], y - bbox[1]), word, fill="black", font=font)

            if wi == target_idx:
                target_bbox = (x - bbox[0] - 4, y - bbox[1] - 4, x - bbox[0] + ww + 4, y - bbox[1] + wh + 4)

            x += ww + space_w

    # Draw red ellipse around target word
    if target_bbox:
        for i in range(ellipse_thickness):
            draw.ellipse(
                [target_bbox[0] - i, target_bbox[1] - i, target_bbox[2] + i, target_bbox[3] + i],
                outline="red",
            )

    ground_truth = target_word
    metadata = {
        "prompt": (
            "Which word in this image is circled? "
            "Put your answer in curly brackets, e.g., {Revenue}."
        ),
        "font_size": font_size,
        "ellipse_thickness": ellipse_thickness,
        "resolution": resolution,
        "sentence": sentence,
        "target_word": target_word,
        "target_index": target_idx,
    }
    return img, ground_truth, metadata
