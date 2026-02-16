"""Task: Text-only merged cell read — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

TASK_CONFIG = {
    "task_name": "merged_cell_read_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "rows": 5,
        "cols": 5,
        "n_merged": 2,
    },
    "sweep_axes": {
        "rows": [4, 5, 6, 8],
        "cols": [4, 5, 6],
        "n_merged": [1, 2, 3],
    },
}

_call_counter = 0


def render(
    rows: int = 5,
    cols: int = 5,
    n_merged: int = 2,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    # Generate cell values
    values = [[f"{chr(65 + r)}{c+1}" for c in range(cols)] for r in range(rows)]

    # Generate merged regions
    merges = []
    occupied = set()
    for _ in range(n_merged):
        for attempt in range(100):
            mr = rng.randint(0, rows - 2)
            mc = rng.randint(0, cols - 2)
            sr = rng.randint(2, min(3, rows - mr))
            sc = rng.randint(2, min(3, cols - mc))
            cells = {(mr + dr, mc + dc) for dr in range(sr) for dc in range(sc)}
            if not cells & occupied:
                occupied |= cells
                merge_val = f"M{len(merges)+1}"
                for r, c in cells:
                    values[r][c] = merge_val
                merges.append((mr, mc, sr, sc, merge_val))
                break

    # Pick a target cell
    target_r = rng.randint(0, rows - 1)
    target_c = rng.randint(0, cols - 1)
    target_value = values[target_r][target_c]

    # Format grid as text
    col_headers = "     " + "".join(f"{'C' + str(c+1):>6}" for c in range(cols))
    grid_lines = [col_headers]
    for r in range(rows):
        row_data = "".join(f"{values[r][c]:>6}" for c in range(cols))
        grid_lines.append(f"  R{r+1:<2}{row_data}")
    grid_text = "\n".join(grid_lines)

    merge_desc = ""
    if merges:
        merge_lines = []
        for mr, mc, sr, sc, mv in merges:
            cells_str = ", ".join(f"(R{mr+dr+1},C{mc+dc+1})" for dr in range(sr) for dc in range(sc))
            merge_lines.append(f"  Cells {cells_str} are merged, containing \"{mv}\"")
        merge_desc = "\nMerged regions:\n" + "\n".join(merge_lines) + "\n"

    prompt = (
        f"Table grid:\n{grid_text}\n{merge_desc}\n"
        f"What value is in cell (R{target_r+1}, C{target_c+1})? "
        f"Put your answer in curly brackets, e.g., {{A1}}."
    )

    metadata = {
        "prompt": prompt,
        "rows": rows,
        "cols": cols,
        "n_merged": len(merges),
        "merges": [(mr, mc, sr, sc) for mr, mc, sr, sc, _ in merges],
        "target_row": target_r,
        "target_col": target_c,
        "mode": "text_only",
    }
    return placeholder_image(), target_value, metadata
