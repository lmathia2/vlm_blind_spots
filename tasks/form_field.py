"""Task T4.3: Form field value extraction."""

from random import Random

from PIL import Image, ImageDraw, ImageFont

TASK_CONFIG = {
    "task_name": "form_field",
    "prompt_template": None,  # dynamic per sample
    "prompt_template_v2": None,
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "n_fields": 6,
        "font_size": 14,
        "field_style": "boxed",
        "resolution": 768,
    },
    "sweep_axes": {
        "n_fields": [5, 8, 12],
        "font_size": [10, 12, 14, 16],
        "field_style": ["boxed", "underlined"],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_FIELD_POOL = [
    ("Company Name", ["Acme Corp", "Globex Inc", "Initech LLC", "Wayne Enterprises", "Stark Industries"]),
    ("Contact Name", ["Alice Johnson", "Bob Martinez", "Carol Lee", "David Chen", "Eva Patel"]),
    ("Email", ["info@acme.com", "bob@globex.io", "carol@initech.net", "d.chen@mail.com"]),
    ("Phone", ["(555) 123-4567", "(555) 987-6543", "(555) 456-7890", "(555) 321-0987"]),
    ("Address", ["123 Main St", "456 Oak Ave", "789 Pine Rd", "321 Elm Blvd"]),
    ("City", ["New York", "Chicago", "San Francisco", "Austin", "Seattle"]),
    ("State", ["NY", "IL", "CA", "TX", "WA"]),
    ("ZIP Code", ["10001", "60601", "94105", "73301", "98101"]),
    ("Invoice #", ["INV-2024-001", "INV-2024-042", "INV-2024-187", "INV-2024-093"]),
    ("Date", ["01/15/2024", "03/22/2024", "06/10/2024", "11/05/2024"]),
    ("Amount Due", ["$1,250.00", "$3,475.50", "$890.25", "$12,340.00"]),
    ("Payment Method", ["Credit Card", "Wire Transfer", "Check", "ACH"]),
    ("Account #", ["4472-8891", "3301-5567", "7789-0012", "6654-3321"]),
    ("PO Number", ["PO-78432", "PO-91205", "PO-44831", "PO-60127"]),
    ("Tax ID", ["12-3456789", "98-7654321", "45-6789012", "67-8901234"]),
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
    n_fields: int = 6,
    font_size: int = 14,
    field_style: str = "boxed",
    resolution: int = 768,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    # Select fields
    selected = rng.sample(_FIELD_POOL, min(n_fields, len(_FIELD_POOL)))
    fields = [(label, rng.choice(values)) for label, values in selected]

    font = _load_font(font_size)
    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)

    # Layout
    padding = 20
    label_x = padding
    row_height = int(font_size * 2.8)
    y_start = padding + 30

    # Title
    title_font = _load_font(font_size + 4)
    draw.text((padding, padding), "Form", fill="black", font=title_font)

    # Compute max label width for alignment
    max_label_w = 0
    for label, _ in fields:
        bbox = draw.textbbox((0, 0), label + ":", font=font)
        max_label_w = max(max_label_w, bbox[2] - bbox[0])

    value_x = label_x + max_label_w + 15
    value_w = resolution - value_x - padding

    for i, (label, value) in enumerate(fields):
        y = y_start + i * row_height

        # Draw label
        draw.text((label_x, y + 4), label + ":", fill="#333333", font=font)

        # Draw value area
        if field_style == "boxed":
            draw.rectangle(
                [(value_x, y), (value_x + value_w, y + row_height - 6)],
                outline="#999999", width=1,
            )
            draw.text((value_x + 6, y + 4), value, fill="black", font=font)
        else:  # underlined
            draw.text((value_x + 2, y + 4), value, fill="black", font=font)
            line_y = y + row_height - 8
            draw.line([(value_x, line_y), (value_x + value_w, line_y)], fill="#999999", width=1)

    # Pick target field
    target_idx = rng.randint(0, len(fields) - 1)
    target_label, target_value = fields[target_idx]
    ground_truth = target_value

    prompt = (
        f"What is the value in the '{target_label}' field? "
        f"Put your answer in curly brackets, e.g., {{Acme Corp}}."
    )

    metadata = {
        "prompt": prompt,
        "n_fields": n_fields,
        "font_size": font_size,
        "field_style": field_style,
        "resolution": resolution,
        "target_label": target_label,
        "fields": [(l, v) for l, v in fields],
    }
    return img, ground_truth, metadata
