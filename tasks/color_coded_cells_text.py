"""Task: Text-only color coded cells — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

TASK_CONFIG = {
    "task_name": "color_coded_cells_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "csv_cell_labels",
    "scorer": "set_match",
    "default_params": {
        "rows": 4,
        "cols": 4,
        "n_colored": 3,
        "target_color": "red",
    },
    "sweep_axes": {
        "rows": [3, 4, 5],
        "cols": [3, 4, 5],
        "n_colored": [2, 3, 4],
        "target_color": ["red", "green", "yellow"],
    },
}

_call_counter = 0


def render(
    rows: int = 4,
    cols: int = 4,
    n_colored: int = 3,
    target_color: str = "red",
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    colors = ["red", "green", "yellow", "blue"]
    other_colors = [c for c in colors if c != target_color]

    # Assign colors to some cells
    n_total_colored = min(n_colored * 2, len(all_cells))
    colored_cells = rng.sample(all_cells, n_total_colored)
    cell_colors = {}
    target_cells = []

    for i, cell in enumerate(colored_cells):
        if i < n_colored:
            cell_colors[cell] = target_color
            target_cells.append(cell)
        else:
            cell_colors[cell] = rng.choice(other_colors)

    # Format grid description
    desc_lines = []
    for r in range(rows):
        for c in range(cols):
            label = f"{chr(65 + r)}{c + 1}"
            if (r, c) in cell_colors:
                desc_lines.append(f"  {label} = {cell_colors[(r, c)]}")
            else:
                desc_lines.append(f"  {label} = white")
    cell_desc = "\n".join(desc_lines)

    target_labels = sorted(f"{chr(65 + r)}{c + 1}" for r, c in target_cells)
    ground_truth = ",".join(target_labels)

    prompt = (
        f"Grid cells and their background colors:\n{cell_desc}\n\n"
        f"Which cells have a {target_color} background? "
        f"List the cell labels separated by commas in curly brackets, e.g., {{A1, B3}}."
    )

    metadata = {
        "prompt": prompt,
        "rows": rows,
        "cols": cols,
        "n_colored": n_colored,
        "target_color": target_color,
        "target_cells": target_labels,
        "all_colored_cells": {f"{chr(65+r)}{c+1}": color for (r, c), color in cell_colors.items()},
        "mode": "text_only",
    }
    return placeholder_image(), ground_truth, metadata
