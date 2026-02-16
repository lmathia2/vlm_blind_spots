"""Task: Text-only realistic table — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

_HEADERS = ["Name", "Revenue", "Cost", "Profit", "Region", "Quarter", "Units", "Growth"]
_NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]

TASK_CONFIG = {
    "task_name": "realistic_table_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "n_rows": 5,
        "n_cols": 4,
    },
    "sweep_axes": {
        "n_rows": [4, 6, 8, 12],
        "n_cols": [3, 4, 5, 6],
    },
}

_call_counter = 0


def render(
    n_rows: int = 5,
    n_cols: int = 4,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    headers = ["ID"] + rng.sample(_HEADERS, min(n_cols - 1, len(_HEADERS)))
    names = rng.sample(_NAMES, min(n_rows, len(_NAMES)))

    rows_data = []
    for i, name in enumerate(names):
        row = [name]
        for h in headers[1:]:
            if h in ("Revenue", "Cost", "Profit"):
                row.append(f"${rng.randint(1000, 9999)}")
            elif h == "Region":
                row.append(rng.choice(["North", "South", "East", "West"]))
            elif h == "Quarter":
                row.append(rng.choice(["Q1", "Q2", "Q3", "Q4"]))
            elif h == "Units":
                row.append(str(rng.randint(50, 999)))
            elif h == "Growth":
                row.append(f"{rng.randint(-20, 40)}%")
            else:
                row.append(str(rng.randint(10, 99)))
        rows_data.append(row)

    target_row_idx = rng.randint(0, len(rows_data) - 1)
    target_col_idx = rng.randint(1, len(headers) - 1)
    target_header = headers[target_col_idx]
    target_value = rows_data[target_row_idx][target_col_idx]
    row_id = rows_data[target_row_idx][0]

    # Format table as text
    col_widths = [max(len(h), max(len(r[i]) for r in rows_data)) + 2 for i, h in enumerate(headers)]
    header_line = "  " + "".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "  " + "".join("-" * w for w in col_widths)
    data_lines = []
    for row in rows_data:
        data_lines.append("  " + "".join(row[i].ljust(col_widths[i]) for i in range(len(headers))))
    table_text = "\n".join([header_line, sep_line] + data_lines)

    prompt = (
        f"Table:\n{table_text}\n\n"
        f"What is the {target_header} value for {row_id}? "
        f"Put your answer in curly brackets, e.g., {{$5000}}."
    )

    metadata = {
        "prompt": prompt,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "target_header": target_header,
        "row_id": row_id,
        "target_row": target_row_idx,
        "target_col": target_col_idx,
        "mode": "text_only",
    }
    return placeholder_image(), target_value, metadata
