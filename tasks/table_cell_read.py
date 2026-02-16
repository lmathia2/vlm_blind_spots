"""Task: Read a 2-digit number from a specific cell of a drawn table."""

from random import Random

from PIL import Image, ImageDraw, ImageFont

TASK_CONFIG = {
    "task_name": "table_cell_read",
    "prompt_template": None,  # filled dynamically per sample via render()
    "parser": "integer",
    "scorer": "exact_match",
    "default_params": {
        "rows": 4,
        "cols": 4,
        "font_size": 16,
        "line_width": 2,
        "resolution": 512,
    },
    "sweep_axes": {
        "rows": [3, 5, 8],
        "cols": [3, 5, 8],
        "font_size": [10, 12, 16, 20],
        "line_width": [1, 2, 3],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_call_counter = 0


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try known font paths, fall back to PIL default."""
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render(
    rows: int = 4,
    cols: int = 4,
    font_size: int = 16,
    line_width: int = 2,
    resolution: int = 512,
) -> tuple[Image.Image, str, dict]:
    """Render a table grid with 2-digit numbers and query one cell."""
    global _call_counter
    _call_counter += 1
    rng = Random(_call_counter)

    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)

    row_h = resolution / rows
    col_w = resolution / cols

    # Generate the table of 2-digit numbers
    table = [[rng.randint(10, 99) for _ in range(cols)] for _ in range(rows)]

    # Draw numbers centered in each cell
    for r in range(rows):
        for c in range(cols):
            text = str(table[r][c])
            cx = c * col_w + col_w / 2
            cy = r * row_h + row_h / 2
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            # Offset accounts for bbox origin not being (0,0)
            tx = cx - tw / 2 - bbox[0]
            ty = cy - th / 2 - bbox[1]
            draw.text((tx, ty), text, fill="black", font=font)

    # Draw grid lines on top of text so they're always visible
    for r in range(rows + 1):
        y = int(r * row_h)
        draw.line([(0, y), (resolution, y)], fill="black", width=line_width)
    for c in range(cols + 1):
        x = int(c * col_w)
        draw.line([(x, 0), (x, resolution)], fill="black", width=line_width)

    # Pick a random target cell (1-indexed for the prompt)
    target_row = rng.randint(1, rows)
    target_col = rng.randint(1, cols)
    ground_truth = str(table[target_row - 1][target_col - 1])

    prompt = (
        f"What number is in row {target_row}, column {target_col} "
        f"of this table? Answer with just the number."
    )

    metadata = {
        "rows": rows,
        "cols": cols,
        "font_size": font_size,
        "line_width": line_width,
        "resolution": resolution,
        "target_row": target_row,
        "target_col": target_col,
        "prompt": prompt,
        "table": table,
    }
    return img, ground_truth, metadata
