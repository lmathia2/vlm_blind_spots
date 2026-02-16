"""Task: Count grid properties — rows/columns, total cells, merged regions."""

from random import Random

from PIL import Image, ImageDraw

TASK_CONFIG = {
    "task_name": "counting_grid",
    "prompt_template": None,  # dynamic per question_type
    "prompt_template_v2": None,
    "parser": "row_col",  # default fallback; overridden per sample in metadata
    "scorer": "row_col",
    "default_params": {
        "rows": 8,
        "cols": 8,
        "resolution": 512,
        "line_width": 2,
        "n_merged": 3,
        "question_type": "total_cells",
    },
    "sweep_axes": {
        "rows": [4, 8, 12, 18, 25],
        "cols": [4, 8, 12, 18, 25],
        "n_merged": [0, 3, 6],
        "question_type": ["grid_size", "total_cells", "merged_count"],
    },
}

_call_counter = 0

_PROMPTS = {
    "grid_size": (
        "Count the number of rows and columns in this grid. "
        "Reply in the format: rows=N columns=M"
    ),
    "total_cells": (
        "Count the total number of cells in this grid. "
        "Merged cells count as a single cell. "
        "Answer with just the number in curly brackets, e.g., {24}."
    ),
    "merged_count": (
        "How many merged (multi-cell) regions are in this grid? "
        "A merged region is where internal borders have been removed to combine cells. "
        "Answer with just the number in curly brackets, e.g., {3}."
    ),
}

_PARSER_SCORER = {
    "grid_size": ("row_col", "row_col"),
    "total_cells": ("integer", "integer_distance"),
    "merged_count": ("integer", "integer_distance"),
}


def _generate_merges(
    rows: int, cols: int, n_merged: int, rng: Random
) -> list[tuple[int, int, int, int]]:
    """Generate non-overlapping rectangular merged regions.

    Returns list of (row, col, span_rows, span_cols).
    """
    occupied: set[tuple[int, int]] = set()
    merges = []

    for _ in range(n_merged):
        for _attempt in range(50):
            r = rng.randint(0, rows - 1)
            c = rng.randint(0, cols - 1)
            if (r, c) in occupied:
                continue

            max_sr = min(3, rows - r)
            max_sc = min(3, cols - c)

            sr = rng.randint(1, max_sr)
            sc = rng.randint(1, max_sc)

            # Must span >1 cell in at least one dimension
            if sr == 1 and sc == 1:
                if max_sr >= 2:
                    sr = 2
                elif max_sc >= 2:
                    sc = 2
                else:
                    continue

            cells = {(r + dr, c + dc) for dr in range(sr) for dc in range(sc)}
            if cells & occupied:
                continue

            occupied |= cells
            merges.append((r, c, sr, sc))
            break

    return merges


def render(
    rows: int = 8,
    cols: int = 8,
    resolution: int = 512,
    line_width: int = 2,
    n_merged: int = 3,
    question_type: str = "total_cells",
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    """Render a grid with optional merged cells.

    Returns (image, ground_truth, metadata).
    """
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    merges = _generate_merges(rows, cols, n_merged, rng) if n_merged > 0 else []

    # Build sets of internal edges to skip for merged regions
    skip_h: set[tuple[int, int]] = set()  # (h, c): skip horizontal seg at row h, col c→c+1
    skip_v: set[tuple[int, int]] = set()  # (r, v): skip vertical seg at col v, row r→r+1

    for mr, mc, sr, sc in merges:
        for h in range(mr + 1, mr + sr):
            for c in range(mc, mc + sc):
                skip_h.add((h, c))
        for v in range(mc + 1, mc + sc):
            for r in range(mr, mr + sr):
                skip_v.add((r, v))

    # Draw
    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)
    cell_h = resolution / rows
    cell_w = resolution / cols

    # Horizontal lines as per-cell segments
    for h in range(rows + 1):
        y = round(h * cell_h)
        for c in range(cols):
            if (h, c) in skip_h:
                continue
            x1 = round(c * cell_w)
            x2 = round((c + 1) * cell_w)
            draw.line([(x1, y), (x2, y)], fill="black", width=line_width)

    # Vertical lines as per-cell segments
    for v in range(cols + 1):
        x = round(v * cell_w)
        for r in range(rows):
            if (r, v) in skip_v:
                continue
            y1 = round(r * cell_h)
            y2 = round((r + 1) * cell_h)
            draw.line([(x, y1), (x, y2)], fill="black", width=line_width)

    # Ground truth
    cells_absorbed = sum(sr * sc - 1 for _, _, sr, sc in merges)
    total_cells = rows * cols - cells_absorbed

    if question_type == "grid_size":
        ground_truth = f"{rows},{cols}"
    elif question_type == "total_cells":
        ground_truth = str(total_cells)
    elif question_type == "merged_count":
        ground_truth = str(len(merges))
    else:
        raise ValueError(f"Unknown question_type: {question_type}")

    parser, scorer = _PARSER_SCORER[question_type]
    prompt = _PROMPTS[question_type]

    metadata = {
        "prompt": prompt,
        "parser": parser,
        "scorer": scorer,
        "rows": rows,
        "cols": cols,
        "resolution": resolution,
        "line_width": line_width,
        "n_merged": len(merges),
        "question_type": question_type,
        "total_cells": total_cells,
        "merges": [(r, c, sr, sc) for r, c, sr, sc in merges],
    }
    return img, ground_truth, metadata
