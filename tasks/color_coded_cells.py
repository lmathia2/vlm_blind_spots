"""Task: Color-coded table cells identification (csv_letters, SET)."""

from random import Random

from PIL import Image, ImageDraw, ImageFont

TASK_CONFIG = {
    "task_name": "color_coded_cells",
    "prompt_template": None,  # dynamic per sample
    "prompt_template_v2": None,
    "parser": "csv_cell_labels",
    "scorer": "set_match",
    "default_params": {
        "rows": 4,
        "cols": 4,
        "n_colored": 3,
        "target_color": "red",
        "font_size": 14,
        "resolution": 512,
    },
    "sweep_axes": {
        "rows": [3, 4, 5],
        "cols": [3, 4, 5],
        "n_colored": [2, 3, 4],
        "target_color": ["red", "green", "yellow"],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_COLOR_MAP = {
    "red": "#FFCCCC",
    "green": "#CCFFCC",
    "yellow": "#FFFFCC",
}

# Non-target background colors (used as distractors in the image)
_OTHER_COLORS = {
    "red": ["#CCFFCC", "#CCCCFF"],      # green, blue
    "green": ["#FFCCCC", "#FFFFCC"],     # red, yellow
    "yellow": ["#FFCCCC", "#CCCCFF"],    # red, blue
}

_call_counter = 0


def _load_font(size: int):
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render(
    rows: int = 4,
    cols: int = 4,
    n_colored: int = 3,
    target_color: str = "red",
    font_size: int = 14,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    total_cells = rows * cols
    # Generate cell labels: A1, A2, ..., B1, B2, ...
    cell_labels = []
    for r in range(rows):
        for c in range(cols):
            cell_labels.append(f"{chr(65 + r)}{c + 1}")

    # Pick cells for target color
    n_target = min(n_colored, total_cells)
    target_indices = sorted(rng.sample(range(total_cells), n_target))

    # Also color some cells with other (non-target) colors for distraction
    remaining = [i for i in range(total_cells) if i not in target_indices]
    n_distractor = min(rng.randint(1, 3), len(remaining))
    distractor_indices = sorted(rng.sample(remaining, n_distractor))
    other_colors = _OTHER_COLORS[target_color]

    # Assign background color for each cell
    cell_bg = {}
    for idx in target_indices:
        cell_bg[idx] = _COLOR_MAP[target_color]
    for idx in distractor_indices:
        cell_bg[idx] = rng.choice(other_colors)

    font = _load_font(font_size)
    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)

    margin = resolution // 10
    table_w = resolution - 2 * margin
    table_h = resolution - 2 * margin - 30  # leave room for title
    cell_w = table_w // cols
    cell_h = table_h // rows
    y_offset = margin + 30

    # Title
    title_font = _load_font(font_size + 4)
    draw.text((margin, margin), "Data Table", fill="black", font=title_font)

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            x0 = margin + c * cell_w
            y0 = y_offset + r * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h

            # Fill background
            bg = cell_bg.get(idx, "white")
            draw.rectangle([(x0, y0), (x1, y1)], fill=bg, outline="#999999", width=1)

            # Draw cell label centered
            label = cell_labels[idx]
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x0 + (cell_w - tw) // 2
            ty = y0 + (cell_h - th) // 2
            draw.text((tx, ty), label, fill="black", font=font)

    target_labels = sorted([cell_labels[i] for i in target_indices])
    ground_truth = ",".join(target_labels)

    prompt = (
        f"Which cells have a {target_color} background? "
        f"List the cell labels separated by commas in curly brackets, e.g., {{A1, B2, C3}}."
    )

    metadata = {
        "prompt": prompt,
        "rows": rows,
        "cols": cols,
        "n_colored": n_colored,
        "target_color": target_color,
        "font_size": font_size,
        "resolution": resolution,
        "target_cells": target_labels,
        "all_colored_cells": {cell_labels[i]: cell_bg[i] for i in cell_bg},
    }
    return img, ground_truth, metadata
