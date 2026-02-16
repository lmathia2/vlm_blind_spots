"""Task T1.3: Realistic table with headers, mixed data types, cell lookup."""

from random import Random

from PIL import Image, ImageDraw, ImageFont

TASK_CONFIG = {
    "task_name": "realistic_table",
    "prompt_template": None,  # dynamic per sample
    "prompt_template_v2": None,
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "n_rows": 5,
        "n_cols": 4,
        "font_size": 14,
        "resolution": 768,
    },
    "sweep_axes": {
        "n_rows": [4, 6, 8, 12],
        "n_cols": [3, 4, 5, 6],
        "font_size": [9, 12, 14, 18],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_HEADERS_SETS = [
    ["Product", "Q1 Revenue", "Q2 Revenue", "YoY %", "Region", "Units Sold"],
    ["Employee", "Department", "Salary", "Bonus %", "Start Date", "Rating"],
    ["Item", "Price", "Quantity", "Discount %", "Total", "Category"],
]

_PRODUCT_NAMES = ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Part 101",
                  "Part 202", "Module C", "Module D", "Service P", "Service Q",
                  "Alpha", "Beta", "Gamma", "Delta"]
_EMPLOYEE_NAMES = ["J. Smith", "A. Chen", "M. Garcia", "K. Patel", "L. Kim",
                   "R. Jones", "S. Brown", "T. Wilson", "N. Taylor", "D. Lee",
                   "E. Davis", "F. Martinez"]
_ITEM_NAMES = ["Laptop", "Monitor", "Keyboard", "Mouse", "Headset",
               "Webcam", "Dock", "Cable", "Adapter", "Stand",
               "Charger", "Case"]

_call_counter = 0


def _load_font(size: int):
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _gen_cell(rng: Random, col_idx: int, header: str) -> str:
    """Generate a plausible cell value based on header type."""
    h = header.lower()
    if any(k in h for k in ["product", "item"]):
        return rng.choice(_PRODUCT_NAMES)
    if "employee" in h:
        return rng.choice(_EMPLOYEE_NAMES)
    if "department" in h:
        return rng.choice(["Sales", "Eng", "Ops", "HR", "Mkt", "Fin"])
    if "category" in h:
        return rng.choice(["A", "B", "C", "D"])
    if "region" in h:
        return rng.choice(["North", "South", "East", "West"])
    if "date" in h:
        m, d = rng.randint(1, 12), rng.randint(1, 28)
        return f"{m:02d}/{d:02d}/2024"
    if "rating" in h:
        return str(rng.randint(1, 5))
    if any(k in h for k in ["revenue", "salary", "price", "total", "bonus"]):
        if "%" in h:
            return f"{rng.randint(1, 35)}%"
        v = rng.randint(100, 9999)
        return f"${v:,}"
    if "%" in h or "discount" in h or "yoy" in h:
        return f"{rng.randint(-15, 40)}%"
    if any(k in h for k in ["quantity", "units", "qty"]):
        return f"{rng.randint(10, 5000):,}"
    return str(rng.randint(10, 999))


def render(
    n_rows: int = 5,
    n_cols: int = 4,
    font_size: int = 14,
    resolution: int = 768,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    header_set = rng.choice(_HEADERS_SETS)
    headers = header_set[:n_cols]

    # Generate data
    data: list[list[str]] = []
    for _ in range(n_rows):
        row = [_gen_cell(rng, ci, headers[ci]) for ci in range(n_cols)]
        data.append(row)

    # Draw table
    font = _load_font(font_size)
    bold_font = font  # PIL doesn't easily do bold; same font for headers

    padding = max(6, font_size // 2)
    # Compute column widths
    all_rows = [headers] + data
    col_widths = []
    img_tmp = Image.new("RGB", (1, 1))
    draw_tmp = ImageDraw.Draw(img_tmp)
    for ci in range(n_cols):
        max_w = 0
        for row in all_rows:
            bbox = draw_tmp.textbbox((0, 0), row[ci], font=font)
            max_w = max(max_w, bbox[2] - bbox[0])
        col_widths.append(max_w + padding * 2)

    row_height = font_size + padding * 2
    table_w = sum(col_widths)
    table_h = row_height * (n_rows + 1)

    # Scale to fit resolution
    scale = min(resolution / (table_w + 20), resolution / (table_h + 20), 1.5)
    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)

    x_offset = max(10, (resolution - int(table_w * scale)) // 2)
    y_offset = max(10, (resolution - int(table_h * scale)) // 2)

    scaled_col_widths = [int(w * scale) for w in col_widths]
    scaled_row_h = int(row_height * scale)
    scaled_font = _load_font(max(7, int(font_size * scale)))

    # Draw header row (gray background)
    y = y_offset
    x = x_offset
    for ci, header in enumerate(headers):
        cw = scaled_col_widths[ci]
        draw.rectangle([(x, y), (x + cw, y + scaled_row_h)], fill="#E0E0E0", outline="black")
        bbox = draw.textbbox((0, 0), header, font=scaled_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x + (cw - tw) // 2, y + (scaled_row_h - th) // 2), header, fill="black", font=scaled_font)
        x += cw

    # Draw data rows
    for ri, row in enumerate(data):
        y = y_offset + scaled_row_h * (ri + 1)
        x = x_offset
        bg = "white" if ri % 2 == 0 else "#F5F5F5"
        for ci, cell in enumerate(row):
            cw = scaled_col_widths[ci]
            draw.rectangle([(x, y), (x + cw, y + scaled_row_h)], fill=bg, outline="black")
            bbox = draw.textbbox((0, 0), cell, font=scaled_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x + (cw - tw) // 2, y + (scaled_row_h - th) // 2), cell, fill="black", font=scaled_font)
            x += cw

    # Pick a random target cell
    target_row = rng.randint(0, n_rows - 1)
    target_col = rng.randint(0, n_cols - 1)
    target_header = headers[target_col]
    # Use first-column value as row identifier
    row_id = data[target_row][0]
    ground_truth = data[target_row][target_col]

    prompt = (
        f"What is the {target_header} for {row_id}? "
        f"Put your answer in curly brackets, e.g., {{$1,456}}."
    )

    metadata = {
        "prompt": prompt,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "font_size": font_size,
        "resolution": resolution,
        "target_header": target_header,
        "row_id": row_id,
        "target_row": target_row,
        "target_col": target_col,
    }
    return img, ground_truth, metadata
