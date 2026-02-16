"""Task T6.2: Rotated text reading."""

from random import Random

from PIL import Image, ImageDraw, ImageFont

TASK_CONFIG = {
    "task_name": "rotated_text",
    "prompt_template": (
        "What does the rotated text in this image say? "
        "Put your answer in curly brackets, e.g., {Revenue}."
    ),
    "prompt_template_v2": (
        "Read the text shown in this image, ignoring its rotation. "
        "Put your answer in curly brackets, e.g., {Revenue}."
    ),
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "rotation": 45,
        "font_size": 18,
        "resolution": 512,
    },
    "sweep_axes": {
        "rotation": [0, 15, 30, 45, 60, 90],
        "font_size": [10, 14, 18, 24],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_LABELS = [
    "Revenue", "Expenses", "Q1 2024", "Growth Rate",
    "Net Profit", "Total Sales", "Year", "Budget",
    "Forecast", "Variance", "Jan", "Feb",
    "Category A", "Region North", "FY2024",
]

_call_counter = 0


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render(
    rotation: int = 45,
    font_size: int = 18,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    text = rng.choice(_LABELS)
    font = _load_font(font_size)

    # Render text on a large canvas, rotate, then crop to resolution
    canvas_size = resolution * 2
    img = Image.new("RGB", (canvas_size, canvas_size), "white")
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (canvas_size - tw) / 2 - bbox[0]
    ty = (canvas_size - th) / 2 - bbox[1]
    draw.text((tx, ty), text, fill="black", font=font)

    if rotation != 0:
        img = img.rotate(rotation, resample=Image.BICUBIC, expand=False, fillcolor="white")

    # Center-crop to target resolution
    left = (canvas_size - resolution) // 2
    top = (canvas_size - resolution) // 2
    img = img.crop((left, top, left + resolution, top + resolution))

    ground_truth = text
    metadata = {
        "rotation": rotation,
        "font_size": font_size,
        "resolution": resolution,
        "text": text,
    }
    return img, ground_truth, metadata
