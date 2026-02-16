"""Task: Draw a vertical list of labeled checkboxes, some checked and some unchecked."""

import string
from random import Random

from PIL import Image, ImageDraw, ImageFont

TASK_CONFIG = {
    "task_name": "form_checkboxes",
    "prompt_template": (
        "Which options are checked? List only the letters of the "
        "checked options, separated by commas, in curly brackets. "
        "For example: {A, C, E}."
    ),
    "prompt_template_v2": (
        "List the letters of all checked checkboxes, separated by commas, "
        "in curly brackets. For example: {A, C}."
    ),
    "parser": "csv_letters",
    "scorer": "set_match",
    "default_params": {
        "n_options": 5,
        "n_checked": 2,
        "box_size": 16,
        "font_size": 16,
        "resolution": 512,
    },
    "sweep_axes": {
        "n_options": [4, 6, 8],
        "n_checked": [1, 2, 3, 4],
        "box_size": [12, 16, 20],
        "font_size": [12, 16, 20],
    },
}

_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_call_counter = 0


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try system fonts, fall back to PIL default."""
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render(
    n_options: int = 5,
    n_checked: int = 2,
    box_size: int = 16,
    font_size: int = 16,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    """Render a form with labeled checkboxes.

    Returns:
        (image, ground_truth, metadata) where ground_truth is a
        comma-separated string of sorted uppercase checked letters.
    """
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    n_checked = min(n_checked, n_options)
    letters = list(string.ascii_uppercase[:n_options])
    checked_set = set(rng.sample(letters, n_checked))

    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)

    # Layout: evenly space rows vertically with padding
    margin_x = int(resolution * 0.1)
    margin_y = int(resolution * 0.1)
    usable_height = resolution - 2 * margin_y
    row_height = usable_height / n_options

    for i, letter in enumerate(letters):
        y_center = margin_y + row_height * i + row_height / 2
        box_top = int(y_center - box_size / 2)
        box_left = margin_x
        box_right = box_left + box_size
        box_bottom = box_top + box_size

        # Draw the checkbox outline
        draw.rectangle(
            [box_left, box_top, box_right, box_bottom],
            outline="black",
            width=2,
        )

        # Draw checkmark if checked
        if letter in checked_set:
            # Two lines forming a ✓ inside the box
            pad = max(2, int(box_size * 0.15))
            x0 = box_left + pad
            y0 = box_top + int(box_size * 0.55)
            x_mid = box_left + int(box_size * 0.4)
            y_mid = box_bottom - pad
            x1 = box_right - pad
            y1 = box_top + pad
            draw.line([(x0, y0), (x_mid, y_mid)], fill="black", width=2)
            draw.line([(x_mid, y_mid), (x1, y1)], fill="black", width=2)

        # Draw label text
        label = f"Option {letter}"
        text_x = box_right + int(box_size * 0.5)
        text_y = int(y_center - font_size / 2)
        draw.text((text_x, text_y), label, fill="black", font=font)

    ground_truth = ",".join(sorted(checked_set))
    metadata = {
        "n_options": n_options,
        "n_checked": n_checked,
        "box_size": box_size,
        "font_size": font_size,
        "resolution": resolution,
        "checked_letters": sorted(checked_set),
    }
    return img, ground_truth, metadata
