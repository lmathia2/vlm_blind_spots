"""Task T4.2: Radio button selection detection."""

from random import Random

from PIL import Image, ImageDraw, ImageFont

TASK_CONFIG = {
    "task_name": "radio_button",
    "prompt_template": None,  # dynamic per sample
    "prompt_template_v2": None,
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "n_groups": 2,
        "options_per_group": 3,
        "circle_size": 14,
        "resolution": 512,
    },
    "sweep_axes": {
        "n_groups": [1, 2, 3],
        "options_per_group": [3, 4],
        "circle_size": [8, 12, 14, 20],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_GROUP_DATA = [
    ("Payment Method", ["Credit Card", "Wire Transfer", "Check", "ACH"]),
    ("Shipping", ["Standard", "Express", "Overnight", "Pickup"]),
    ("Priority", ["Low", "Medium", "High", "Critical"]),
    ("Frequency", ["Daily", "Weekly", "Monthly", "Quarterly"]),
    ("Format", ["PDF", "Excel", "CSV", "JSON"]),
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
    n_groups: int = 2,
    options_per_group: int = 3,
    circle_size: int = 14,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    selected_groups = rng.sample(_GROUP_DATA, min(n_groups, len(_GROUP_DATA)))
    groups = []
    for name, options in selected_groups:
        opts = options[:options_per_group]
        selected_idx = rng.randint(0, len(opts) - 1)
        groups.append({"name": name, "options": opts, "selected": selected_idx})

    font = _load_font(max(10, resolution // 40))
    title_font = _load_font(max(12, resolution // 35))
    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)

    margin = resolution // 10
    y = margin
    r = circle_size // 2
    fill_r = max(r - 4, r * 5 // 10)
    option_spacing = circle_size + 12

    for group in groups:
        # Group title
        draw.text((margin, y), group["name"], fill="#333333", font=title_font)
        y += int(title_font.size * 1.8) if hasattr(title_font, 'size') else 24

        for oi, option in enumerate(group["options"]):
            cx = margin + r + 5
            cy = y + r
            # Outer circle
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline="black", width=2)
            # Fill if selected
            if oi == group["selected"]:
                draw.ellipse([(cx - fill_r, cy - fill_r), (cx + fill_r, cy + fill_r)], fill="black")
            # Label
            draw.text((cx + r + 10, y + 2), option, fill="black", font=font)
            y += option_spacing

        y += 20  # gap between groups

    # Pick a target group to ask about
    target_group_idx = rng.randint(0, len(groups) - 1)
    target = groups[target_group_idx]
    ground_truth = target["options"][target["selected"]]

    prompt = (
        f"In the '{target['name']}' group, which option is selected? "
        f"Put your answer in curly brackets, e.g., {{Credit Card}}."
    )

    metadata = {
        "prompt": prompt,
        "n_groups": n_groups,
        "options_per_group": options_per_group,
        "circle_size": circle_size,
        "resolution": resolution,
        "target_group": target["name"],
        "selected_option": ground_truth,
    }
    return img, ground_truth, metadata
