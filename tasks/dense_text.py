"""Task T6.3: Dense small text extraction — read a specific line."""

from random import Random

from PIL import Image, ImageDraw, ImageFont

TASK_CONFIG = {
    "task_name": "dense_text",
    "prompt_template": None,  # dynamic per sample
    "prompt_template_v2": None,
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "font_size": 10,
        "n_lines": 7,
        "line_spacing": 1.2,
        "resolution": 512,
    },
    "sweep_axes": {
        "font_size": [8, 9, 10, 12, 14],
        "n_lines": [5, 7, 10],
        "line_spacing": [1.0, 1.2, 1.5],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_TEXT_LINES = [
    "This agreement is governed by the laws of the State of Delaware.",
    "All disputes shall be resolved through binding arbitration.",
    "The licensee agrees to maintain confidentiality of all proprietary information.",
    "Payment terms are net 30 days from the date of invoice.",
    "Warranty coverage extends for a period of twelve (12) months.",
    "The vendor shall not be liable for indirect or consequential damages.",
    "Force majeure events include but are not limited to natural disasters.",
    "Either party may terminate this agreement with 30 days written notice.",
    "The maximum aggregate liability shall not exceed the total contract value.",
    "All modifications must be in writing and signed by both parties.",
    "Intellectual property rights remain with the original creator.",
    "The indemnifying party shall defend against all third-party claims.",
    "Compliance with applicable export control regulations is required.",
    "Data processing activities are subject to the privacy policy.",
    "Annual escalation shall not exceed three percent (3%) per year.",
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
    font_size: int = 10,
    n_lines: int = 7,
    line_spacing: float = 1.2,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    lines = rng.sample(_TEXT_LINES, min(n_lines, len(_TEXT_LINES)))
    target_line_num = rng.randint(1, len(lines))  # 1-indexed
    target_text = lines[target_line_num - 1]

    font = _load_font(font_size)
    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)

    line_h = int(font_size * line_spacing)
    total_h = line_h * len(lines)
    y_start = (resolution - total_h) // 2
    x_margin = 15

    for i, line in enumerate(lines):
        y = y_start + i * line_h
        draw.text((x_margin, y), line, fill="black", font=font)

    ground_truth = target_text

    ordinal = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
               6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}
    line_ref = ordinal.get(target_line_num, f"{target_line_num}th")

    prompt = (
        f"What does the {line_ref} line of text say? "
        f"Copy it exactly. Put your answer in curly brackets."
    )

    metadata = {
        "prompt": prompt,
        "font_size": font_size,
        "n_lines": len(lines),
        "line_spacing": line_spacing,
        "resolution": resolution,
        "target_line_num": target_line_num,
        "lines": lines,
    }
    return img, ground_truth, metadata
