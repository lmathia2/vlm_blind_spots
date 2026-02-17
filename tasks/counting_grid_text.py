"""Task: Text-only counting grid — perception vs reasoning diagnostic.

Provides grid line positions as coordinates instead of an image.
If text accuracy >> image accuracy, the failure is perceptual.
"""

from random import Random

from PIL import Image

from tasks.counting_grid import _generate_merges, _PROMPTS, _PARSER_SCORER

TASK_CONFIG = {
    "task_name": "counting_grid_text",
    "prompt_template": "",  # filled dynamically
    "prompt_template_v2": "",
    "parser": "row_col",  # default; overridden per sample
    "scorer": "exact_match",
    "default_params": {
        "rows": 8,
        "cols": 8,
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


def render(
    rows: int = 8,
    cols: int = 8,
    n_merged: int = 3,
    question_type: str = "total_cells",
    seed: int | None = None,
    prompt_variant: int = 1,
) -> tuple[Image.Image, str, dict]:
    """Return a tiny placeholder image with grid structure as text."""
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    merges = _generate_merges(rows, cols, n_merged, rng) if n_merged > 0 else []

    # Describe grid structure as text
    grid_desc = (
        f"A grid has {rows} rows and {cols} columns.\n"
        f"Horizontal lines are drawn at y-positions: {', '.join(str(i) for i in range(rows + 1))}\n"
        f"Vertical lines are drawn at x-positions: {', '.join(str(i) for i in range(cols + 1))}\n"
    )

    if merges:
        merge_descs = []
        for mr, mc, sr, sc in merges:
            cells = []
            for dr in range(sr):
                for dc in range(sc):
                    cells.append(f"({mr + dr},{mc + dc})")
            merge_descs.append(
                f"  Cells {', '.join(cells)} are merged into one region"
            )
        grid_desc += "Merged regions (internal borders removed):\n" + "\n".join(merge_descs) + "\n"
    else:
        grid_desc += "No cells are merged.\n"

    # Ground truth
    cells_absorbed = sum(sr * sc - 1 for _, _, sr, sc in merges)
    total_cells = rows * cols - cells_absorbed

    if question_type == "grid_size":
        ground_truth = f"{rows},{cols}"
        question = "How many rows and columns does this grid have? Reply in the format: rows=N columns=M"
    elif question_type == "total_cells":
        ground_truth = str(total_cells)
        question = (
            "Count the total number of cells in this grid. "
            "Merged cells count as a single cell. "
            "Answer with just the number in curly brackets, e.g., {24}."
        )
    elif question_type == "merged_count":
        ground_truth = str(len(merges))
        question = (
            "How many merged (multi-cell) regions are in this grid? "
            "Answer with just the number in curly brackets, e.g., {3}."
        )
    else:
        raise ValueError(f"Unknown question_type: {question_type}")

    parser, scorer = _PARSER_SCORER[question_type]
    prompt = f"{grid_desc}\n{question}"

    # Tiny placeholder image
    img = Image.new("RGB", (64, 64), "white")

    metadata = {
        "prompt": prompt,
        "parser": parser,
        "scorer": scorer,
        "rows": rows,
        "cols": cols,
        "n_merged": len(merges),
        "question_type": question_type,
        "total_cells": total_cells,
        "merges": [(r, c, sr, sc) for r, c, sr, sc in merges],
        "mode": "text_only",
    }
    return img, ground_truth, metadata
