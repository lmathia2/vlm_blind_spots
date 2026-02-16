"""Task T6.1: Text readability under degradation (blur, contrast, rotation, noise)."""

from random import Random

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont

TASK_CONFIG = {
    "task_name": "text_degradation",
    "prompt_template": (
        "What does the text in this image say? "
        "Put your answer in curly brackets, e.g., {Total: $500}."
    ),
    "prompt_template_v2": (
        "Read the text shown in this image. "
        "Put your answer in curly brackets, e.g., {Total: $500}."
    ),
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "font_size": 18,
        "blur_radius": 0,
        "rotation": 0,
        "contrast": 1.0,
        "resolution": 512,
    },
    "sweep_axes": {
        "font_size": [8, 12, 16, 22, 28],
        "blur_radius": [0, 1, 2, 3],
        "rotation": [0, 2, 5],
        "contrast": [1.0, 0.6, 0.3],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_TEXT_STRINGS = [
    "Total: $42,387.19",
    "Invoice #INV-2024-0892",
    "Balance Due: $1,256.00",
    "PO-78432-A",
    "Ref: TXN-20240315-0047",
    "Net Amount: $8,914.52",
    "Account #00-4472-8891",
    "Date: 03/15/2024",
    "Qty: 1,250 units",
    "Rate: $12.75/hr",
    "Subtotal: $15,937.50",
    "Tax (8.5%): $1,354.69",
    "Approved by: J. Smith",
    "Contract #C-2024-00341",
    "Wire Ref: SWIFT-ABCD1234",
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
    font_size: int = 18,
    blur_radius: int = 0,
    rotation: int = 0,
    contrast: float = 1.0,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    text = rng.choice(_TEXT_STRINGS)
    font = _load_font(font_size)

    # Render text centered on white background
    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (resolution - tw) / 2 - bbox[0]
    ty = (resolution - th) / 2 - bbox[1]
    draw.text((tx, ty), text, fill="black", font=font)

    # Apply degradation
    if rotation != 0:
        img = img.rotate(rotation, resample=Image.BICUBIC, expand=False, fillcolor="white")
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if contrast < 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)

    ground_truth = text
    metadata = {
        "font_size": font_size,
        "blur_radius": blur_radius,
        "rotation": rotation,
        "contrast": contrast,
        "resolution": resolution,
        "text": text,
    }
    return img, ground_truth, metadata
