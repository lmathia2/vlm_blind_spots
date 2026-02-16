"""Task: Text-only table cell read — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

TASK_CONFIG = {
    "task_name": "table_cell_read_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "integer",
    "scorer": "exact_match",
    "default_params": {
        "rows": 4,
        "cols": 4,
    },
    "sweep_axes": {
        "rows": [3, 5, 8],
        "cols": [3, 5, 8],
    },
}

_call_counter = 0


def render(
    rows: int = 4,
    cols: int = 4,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    # Generate table data (same as parent: 2-digit numbers)
    table = [[rng.randint(10, 99) for _ in range(cols)] for _ in range(rows)]
    target_row = rng.randint(1, rows)
    target_col = rng.randint(1, cols)
    target_value = table[target_row - 1][target_col - 1]

    # Format as text table
    col_headers = "".join(f"{'C' + str(c+1):>6}" for c in range(cols))
    table_lines = [f"      {col_headers}"]
    for r in range(rows):
        row_data = "".join(f"{table[r][c]:>6}" for c in range(cols))
        table_lines.append(f"  R{r+1:<3}{row_data}")
    table_text = "\n".join(table_lines)

    prompt = (
        f"Table:\n{table_text}\n\n"
        f"What number is in row {target_row}, column {target_col}? "
        f"Answer with just the number in curly brackets, e.g., {{42}}."
    )

    ground_truth = str(target_value)
    metadata = {
        "prompt": prompt,
        "rows": rows,
        "cols": cols,
        "target_row": target_row,
        "target_col": target_col,
        "table": table,
        "mode": "text_only",
    }
    return placeholder_image(), ground_truth, metadata
