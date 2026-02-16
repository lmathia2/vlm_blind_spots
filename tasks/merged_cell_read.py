"""Task T1.4: Merged cell text reading."""

from random import Random

from PIL import Image, ImageDraw, ImageFont

TASK_CONFIG = {
    "task_name": "merged_cell_read",
    "prompt_template": None,  # dynamic per sample
    "prompt_template_v2": None,
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "rows": 5,
        "cols": 5,
        "n_merged": 2,
        "font_size": 14,
        "resolution": 768,
    },
    "sweep_axes": {
        "rows": [4, 5, 6, 8],
        "cols": [4, 5, 6],
        "n_merged": [1, 2, 3],
        "font_size": [10, 12, 14],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_MERGE_LABELS = [
    "Total", "Subtotal", "Grand Total", "Summary", "Combined",
    "Merged Region", "Header", "Category A", "Category B", "All Items",
    "Q1-Q2", "H1 Total", "Full Year", "Net Amount", "Overview",
]

_call_counter = 0


def _load_font(size: int):
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render(
    rows: int = 5,
    cols: int = 5,
    n_merged: int = 2,
    font_size: int = 14,
    resolution: int = 768,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    font = _load_font(font_size)
    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)

    margin = 20
    cell_w = (resolution - margin * 2) // cols
    cell_h = (resolution - margin * 2) // rows

    # Generate merged regions (non-overlapping)
    occupied: set[tuple[int, int]] = set()
    merges = []
    available_labels = list(_MERGE_LABELS)
    rng.shuffle(available_labels)

    for mi in range(n_merged):
        for _attempt in range(50):
            r = rng.randint(0, rows - 1)
            c = rng.randint(0, cols - 1)
            if (r, c) in occupied:
                continue
            max_sr = min(3, rows - r)
            max_sc = min(3, cols - c)
            sr = rng.randint(1, max_sr)
            sc = rng.randint(1, max_sc)
            if sr == 1 and sc == 1:
                if max_sc >= 2:
                    sc = 2
                elif max_sr >= 2:
                    sr = 2
                else:
                    continue
            cells = {(r + dr, c + dc) for dr in range(sr) for dc in range(sc)}
            if cells & occupied:
                continue
            occupied |= cells
            label = available_labels[mi % len(available_labels)]
            merges.append({"r": r, "c": c, "sr": sr, "sc": sc, "label": label})
            break

    # Fill regular cells with numbers
    cell_values: dict[tuple[int, int], str] = {}
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in occupied:
                cell_values[(r, c)] = str(rng.randint(10, 99))

    # Build skip sets for grid lines
    skip_h: set[tuple[int, int]] = set()
    skip_v: set[tuple[int, int]] = set()
    for m in merges:
        for h in range(m["r"] + 1, m["r"] + m["sr"]):
            for c in range(m["c"], m["c"] + m["sc"]):
                skip_h.add((h, c))
        for v in range(m["c"] + 1, m["c"] + m["sc"]):
            for r in range(m["r"], m["r"] + m["sr"]):
                skip_v.add((r, v))

    # Draw cell contents
    for (r, c), val in cell_values.items():
        cx = margin + c * cell_w + cell_w // 2
        cy = margin + r * cell_h + cell_h // 2
        bbox = draw.textbbox((0, 0), val, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw // 2, cy - th // 2), val, fill="black", font=font)

    # Draw merged region contents (centered, with light background)
    for m in merges:
        x0 = margin + m["c"] * cell_w + 1
        y0 = margin + m["r"] * cell_h + 1
        x1 = margin + (m["c"] + m["sc"]) * cell_w - 1
        y1 = margin + (m["r"] + m["sr"]) * cell_h - 1
        draw.rectangle([(x0, y0), (x1, y1)], fill="#E8F0FE")
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        bbox = draw.textbbox((0, 0), m["label"], font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw // 2, cy - th // 2), m["label"], fill="black", font=font)

    # Draw grid lines (with gaps for merges)
    for h in range(rows + 1):
        y = margin + h * cell_h
        for c in range(cols):
            if (h, c) in skip_h:
                continue
            x1 = margin + c * cell_w
            x2 = margin + (c + 1) * cell_w
            draw.line([(x1, y), (x2, y)], fill="black", width=2)

    for v in range(cols + 1):
        x = margin + v * cell_w
        for r in range(rows):
            if (r, v) in skip_v:
                continue
            y1 = margin + r * cell_h
            y2 = margin + (r + 1) * cell_h
            draw.line([(x, y1), (x, y2)], fill="black", width=2)

    # Pick a merged region to ask about
    if not merges:
        # Fallback: ask about a regular cell
        target_r, target_c = rng.choice(list(cell_values.keys()))
        ground_truth = cell_values[(target_r, target_c)]
        prompt = (
            f"What value is in row {target_r + 1}, column {target_c + 1}? "
            f"Put your answer in curly brackets."
        )
    else:
        target = rng.choice(merges)
        ground_truth = target["label"]
        col_start = target["c"] + 1
        col_end = target["c"] + target["sc"]
        row_ref = target["r"] + 1
        if target["sc"] > 1:
            prompt = (
                f"What text is in the cell that spans columns {col_start}-{col_end} "
                f"in row {row_ref}? Put your answer in curly brackets."
            )
        else:
            row_end = target["r"] + target["sr"]
            prompt = (
                f"What text is in the cell that spans rows {row_ref}-{row_end} "
                f"in column {col_start}? Put your answer in curly brackets."
            )

    metadata = {
        "prompt": prompt,
        "rows": rows,
        "cols": cols,
        "n_merged": len(merges),
        "font_size": font_size,
        "resolution": resolution,
        "merges": [{"r": m["r"], "c": m["c"], "sr": m["sr"], "sc": m["sc"], "label": m["label"]} for m in merges],
    }
    return img, ground_truth, metadata
