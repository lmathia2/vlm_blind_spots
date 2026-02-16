"""Task: Count rows and columns in a grid image."""

from PIL import Image, ImageDraw

TASK_CONFIG = {
    "task_name": "counting_grid",
    "prompt_template": (
        "Count the number of rows and columns in this grid. "
        "Reply in the format: rows=N columns=M"
    ),
    "prompt_template_v2": (
        "This image shows a grid. How many rows and columns does it have? "
        "Reply as: rows=N columns=M"
    ),
    "parser": "row_col",
    "scorer": "row_col",
    "default_params": {"rows": 5, "cols": 6, "resolution": 512, "line_width": 2},
    "sweep_axes": {
        "rows": [3, 5, 7, 10, 15],
        "cols": [3, 5, 7, 10, 15],
        "line_width": [1, 2, 3, 5, 10],
        "resolution": [256, 512, 768, 1024],
    },
}


def render(rows: int = 5, cols: int = 6, resolution: int = 512,
           line_width: int = 2) -> tuple[Image.Image, str, dict]:
    """Render a grid image. Returns (image, ground_truth, metadata)."""
    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)
    row_h = resolution / rows
    col_w = resolution / cols
    for r in range(rows + 1):
        y = int(r * row_h)
        draw.line([(0, y), (resolution, y)], fill="black", width=line_width)
    for c in range(cols + 1):
        x = int(c * col_w)
        draw.line([(x, 0), (x, resolution)], fill="black", width=line_width)

    ground_truth = f"{rows},{cols}"
    metadata = {"rows": rows, "cols": cols, "resolution": resolution, "line_width": line_width}
    return img, ground_truth, metadata
