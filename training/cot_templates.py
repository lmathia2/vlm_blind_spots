"""Chain-of-thought templates for grid counting SFT data.

Three strategy groups, each with 3-5 paraphrase variants:
  - Direct counting (grid sizes 3-12)
  - Intermediate representation (grid sizes 3-15)
  - Tool use (grid sizes 12-25)

Templates use str.format() with named placeholders derived from grid params.
"""

from random import Random

# ---------------------------------------------------------------------------
# Strategy 1: Direct counting
# ---------------------------------------------------------------------------

DIRECT_COT_TEMPLATES = [
    # Variant 1 — canonical
    (
        "I need to count lines, not cells — N+1 lines create N rows.\n"
        "\n"
        "Horizontal: top border + {h_interior} dividers + bottom border = "
        "{h_lines} lines → {rows} rows.\n"
        "Vertical: left border + {v_interior} dividers + right border = "
        "{v_lines} lines → {cols} columns.\n"
        "\n"
        "rows={rows} columns={cols}"
    ),
    # Variant 2 — borders first
    (
        "Let me count the grid lines carefully.\n"
        "\n"
        "There are {h_lines} horizontal lines (2 borders + {h_interior} "
        "internal dividers). Since N+1 lines create N rows, that means "
        "{rows} rows.\n"
        "There are {v_lines} vertical lines (2 borders + {v_interior} "
        "internal dividers). That gives {cols} columns.\n"
        "\n"
        "rows={rows} columns={cols}"
    ),
    # Variant 3 — counting up
    (
        "Counting horizontal lines from top to bottom: I see {h_lines} lines.\n"
        "The number of rows is always one less than the number of horizontal "
        "lines: {h_lines} - 1 = {rows}.\n"
        "\n"
        "Counting vertical lines from left to right: I see {v_lines} lines.\n"
        "Columns = vertical lines minus 1: {v_lines} - 1 = {cols}.\n"
        "\n"
        "rows={rows} columns={cols}"
    ),
    # Variant 4 — rule-first
    (
        "Key rule: a grid with R rows has R+1 horizontal lines.\n"
        "\n"
        "I count {h_lines} horizontal lines, so rows = {h_lines} - 1 = {rows}.\n"
        "I count {v_lines} vertical lines, so columns = {v_lines} - 1 = {cols}.\n"
        "\n"
        "rows={rows} columns={cols}"
    ),
    # Variant 5 — verbose
    (
        "To determine the grid dimensions, I'll count lines and apply the "
        "subtraction rule.\n"
        "\n"
        "Horizontal lines: the top edge is line 1, then there are "
        "{h_interior} internal dividers, plus the bottom edge — {h_lines} "
        "lines total. Each pair of adjacent horizontal lines forms one row, "
        "so {h_lines} lines → {rows} rows.\n"
        "\n"
        "Vertical lines: left edge + {v_interior} dividers + right edge = "
        "{v_lines} lines. Similarly, {v_lines} lines → {cols} columns.\n"
        "\n"
        "rows={rows} columns={cols}"
    ),
]

# Self-correction insert: placed after an intentional wrong step
SELF_CORRECTION_INSERT = (
    "Wait — I almost said {h_lines} rows, but {h_lines} lines means "
    "{h_lines} - 1 = {rows} rows. Lines and rows are not the same thing.\n"
)

# ---------------------------------------------------------------------------
# Strategy 2: Intermediate representation
# ---------------------------------------------------------------------------

INTERMEDIATE_COT_TEMPLATES = [
    # Variant 1 — structured summary
    (
        "Let me build a structured summary first.\n"
        "\n"
        "Line count summary:\n"
        "  Horizontal lines: {h_lines} (top border, {h_interior} dividers, "
        "bottom border)\n"
        "  Vertical lines: {v_lines} (left border, {v_interior} dividers, "
        "right border)\n"
        "\n"
        "Applying the N+1 rule:\n"
        "  {h_lines} horizontal lines → {rows} rows\n"
        "  {v_lines} vertical lines → {cols} columns\n"
        "\n"
        "rows={rows} columns={cols}"
    ),
    # Variant 2 — ASCII sketch then answer
    (
        "I'll sketch the grid structure to make the count explicit.\n"
        "\n"
        "{ascii_sketch}\n"
        "\n"
        "From the sketch:\n"
        "  Horizontal lines = {h_lines} → rows = {rows}\n"
        "  Vertical lines = {v_lines} → columns = {cols}\n"
        "\n"
        "rows={rows} columns={cols}"
    ),
    # Variant 3 — tabular summary
    (
        "Organizing my observations:\n"
        "\n"
        "| Direction  | Border lines | Interior dividers | Total lines | "
        "Cells |\n"
        "|------------|-------------|-------------------|-------------|------|\n"
        "| Horizontal | 2           | {h_interior}              | "
        "{h_lines}           | {rows}   |\n"
        "| Vertical   | 2           | {v_interior}              | "
        "{v_lines}           | {cols}   |\n"
        "\n"
        "The 'Cells' column is total lines minus 1 in each direction.\n"
        "\n"
        "rows={rows} columns={cols}"
    ),
    # Variant 4 — enumerate then derive
    (
        "Step 1: Enumerate what I see.\n"
        "  - {h_lines} horizontal lines running across the grid\n"
        "  - {v_lines} vertical lines running down the grid\n"
        "\n"
        "Step 2: Derive dimensions.\n"
        "  Lines always exceed the cell count by 1 (the fence-post principle).\n"
        "  Rows = {h_lines} - 1 = {rows}\n"
        "  Columns = {v_lines} - 1 = {cols}\n"
        "\n"
        "rows={rows} columns={cols}"
    ),
    # Variant 5 — visual breakdown
    (
        "Breaking down the grid visually:\n"
        "\n"
        "Horizontal structure:\n"
        "  Line 1 (top border)\n"
        "  ... {h_interior} interior dividers ...\n"
        "  Line {h_lines} (bottom border)\n"
        "  Total: {h_lines} lines → {rows} rows\n"
        "\n"
        "Vertical structure:\n"
        "  Line 1 (left border)\n"
        "  ... {v_interior} interior dividers ...\n"
        "  Line {v_lines} (right border)\n"
        "  Total: {v_lines} lines → {cols} columns\n"
        "\n"
        "rows={rows} columns={cols}"
    ),
]

# ---------------------------------------------------------------------------
# Strategy 3: Tool use
# ---------------------------------------------------------------------------

TOOL_USE_COT_TEMPLATES = [
    # Variant 1 — canonical
    (
        "This grid looks too dense to count visually. I'll write code to "
        "detect the lines.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "from PIL import Image\n"
        "\n"
        "img = np.array(Image.open(image_path).convert('L'))\n"
        "dark = img < 128\n"
        "\n"
        "# Count horizontal lines: rows where >50% of pixels are dark\n"
        "h_mask = dark.mean(axis=1) > 0.5\n"
        "h_lines = np.diff(h_mask.astype(int))\n"
        "n_h = (h_lines == 1).sum()\n"
        "\n"
        "# Count vertical lines: columns where >50% of pixels are dark\n"
        "v_mask = dark.mean(axis=0) > 0.5\n"
        "v_lines = np.diff(v_mask.astype(int))\n"
        "n_v = (v_lines == 1).sum()\n"
        "\n"
        "print(f'Horizontal lines: {{n_h}}, Vertical lines: {{n_v}}')\n"
        "print(f'Rows: {{n_h - 1}}, Columns: {{n_v - 1}}')\n"
        "```\n"
        "\n"
        "```output\n"
        "Horizontal lines: {h_lines}, Vertical lines: {v_lines}\n"
        "Rows: {rows}, Columns: {cols}\n"
        "```\n"
        "\n"
        "The code found {h_lines} horizontal lines and {v_lines} vertical "
        "lines. Applying the subtraction rule: {rows} rows and {cols} columns.\n"
        "\n"
        "rows={rows} columns={cols}"
    ),
    # Variant 2 — thresholding approach
    (
        "The grid is dense — visual counting would be unreliable. Let me use "
        "a line-detection approach.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "from PIL import Image\n"
        "\n"
        "img = np.array(Image.open(image_path).convert('L'))\n"
        "threshold = 128\n"
        "binary = img < threshold\n"
        "\n"
        "# Project onto axes\n"
        "row_proj = binary.mean(axis=1)\n"
        "col_proj = binary.mean(axis=0)\n"
        "\n"
        "# Find line positions (peaks in projection)\n"
        "h_positions = np.where(row_proj > 0.3)[0]\n"
        "v_positions = np.where(col_proj > 0.3)[0]\n"
        "\n"
        "# Group adjacent pixels into lines\n"
        "def count_groups(positions):\n"
        "    if len(positions) == 0:\n"
        "        return 0\n"
        "    groups = 1\n"
        "    for i in range(1, len(positions)):\n"
        "        if positions[i] - positions[i-1] > 3:\n"
        "            groups += 1\n"
        "    return groups\n"
        "\n"
        "n_h = count_groups(h_positions)\n"
        "n_v = count_groups(v_positions)\n"
        "print(f'{{n_h}} horizontal lines, {{n_v}} vertical lines')\n"
        "print(f'Grid: {{n_h - 1}} rows x {{n_v - 1}} columns')\n"
        "```\n"
        "\n"
        "```output\n"
        "{h_lines} horizontal lines, {v_lines} vertical lines\n"
        "Grid: {rows} rows x {cols} columns\n"
        "```\n"
        "\n"
        "The line detection confirms: {rows} rows and {cols} columns.\n"
        "\n"
        "rows={rows} columns={cols}"
    ),
    # Variant 3 — gradients approach
    (
        "With this many cells, I should use code rather than try to count "
        "visually.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "from PIL import Image\n"
        "\n"
        "img = np.array(Image.open(image_path).convert('L')).astype(float)\n"
        "\n"
        "# Use gradient to find edges\n"
        "h_grad = np.abs(np.diff(img, axis=0)).mean(axis=1)\n"
        "v_grad = np.abs(np.diff(img, axis=1)).mean(axis=0)\n"
        "\n"
        "# Peaks in gradient correspond to line positions\n"
        "h_thresh = h_grad.max() * 0.3\n"
        "v_thresh = v_grad.max() * 0.3\n"
        "\n"
        "h_peaks = np.where(h_grad > h_thresh)[0]\n"
        "v_peaks = np.where(v_grad > v_thresh)[0]\n"
        "\n"
        "# Group nearby peaks\n"
        "def count_lines(peaks, min_gap=5):\n"
        "    if len(peaks) == 0:\n"
        "        return 0\n"
        "    count = 1\n"
        "    for i in range(1, len(peaks)):\n"
        "        if peaks[i] - peaks[i-1] > min_gap:\n"
        "            count += 1\n"
        "    return count\n"
        "\n"
        "n_h = count_lines(h_peaks)\n"
        "n_v = count_lines(v_peaks)\n"
        "print(f'Detected {{n_h}} horizontal lines, {{n_v}} vertical lines')\n"
        "print(f'Therefore: {{n_h - 1}} rows, {{n_v - 1}} columns')\n"
        "```\n"
        "\n"
        "```output\n"
        "Detected {h_lines} horizontal lines, {v_lines} vertical lines\n"
        "Therefore: {rows} rows, {cols} columns\n"
        "```\n"
        "\n"
        "The gradient analysis detected {h_lines} horizontal and {v_lines} "
        "vertical lines. Subtracting 1 from each gives {rows} rows and "
        "{cols} columns.\n"
        "\n"
        "rows={rows} columns={cols}"
    ),
]

# Templates for easy grids where the model explicitly skips tool use
TOOL_USE_SKIP_TEMPLATES = [
    # Variant 1
    (
        "This is a small grid — I can count the lines directly without code.\n"
        "\n"
        "Horizontal lines: {h_lines} (top + {h_interior} dividers + bottom).\n"
        "Vertical lines: {v_lines} (left + {v_interior} dividers + right).\n"
        "\n"
        "{h_lines} lines → {rows} rows, {v_lines} lines → {cols} columns.\n"
        "\n"
        "rows={rows} columns={cols}"
    ),
    # Variant 2
    (
        "The grid is small enough to count visually.\n"
        "\n"
        "I see {h_lines} horizontal lines and {v_lines} vertical lines. "
        "Applying the fence-post rule: rows = {h_lines} - 1 = {rows}, "
        "columns = {v_lines} - 1 = {cols}.\n"
        "\n"
        "rows={rows} columns={cols}"
    ),
    # Variant 3
    (
        "No need for code here — the grid is easy to count.\n"
        "\n"
        "Counting carefully: {h_lines} horizontal lines make {rows} rows, "
        "and {v_lines} vertical lines make {cols} columns.\n"
        "\n"
        "rows={rows} columns={cols}"
    ),
]


def build_ascii_sketch(rows: int, cols: int) -> str:
    """Build an ASCII sketch of a grid, capped at 8x8 for readability.

    If the grid exceeds 8 in either dimension, the sketch shows a
    truncated version with '...' markers.
    """
    show_rows = min(rows, 8)
    show_cols = min(cols, 8)
    truncate_rows = rows > 8
    truncate_cols = cols > 8

    lines = []
    # Top border
    if truncate_cols:
        lines.append("+" + "---+" * show_cols + " ...")
    else:
        lines.append("+" + "---+" * show_cols)

    for r in range(show_rows):
        if truncate_cols:
            lines.append("|" + "   |" * show_cols + " ...")
        else:
            lines.append("|" + "   |" * show_cols)
        if truncate_cols:
            lines.append("+" + "---+" * show_cols + " ...")
        else:
            lines.append("+" + "---+" * show_cols)

    if truncate_rows:
        lines.append("  ...")

    return "\n".join(lines)


def fill_template(
    template: str,
    rows: int,
    cols: int,
    rng: Random,
    include_self_correction: bool = False,
) -> str:
    """Fill a CoT template with derived grid values.

    Args:
        template: Template string with named placeholders.
        rows: Number of grid rows.
        cols: Number of grid columns.
        rng: Random instance for any stochastic choices.
        include_self_correction: If True, inject a self-correction pattern
            after the first mention of horizontal line count.

    Returns:
        Filled CoT string.
    """
    h_lines = rows + 1
    v_lines = cols + 1
    h_interior = h_lines - 2  # dividers between borders
    v_interior = v_lines - 2

    values = {
        "rows": rows,
        "cols": cols,
        "h_lines": h_lines,
        "v_lines": v_lines,
        "h_interior": h_interior,
        "v_interior": v_interior,
        "ascii_sketch": build_ascii_sketch(rows, cols),
    }

    filled = template.format(**values)

    if include_self_correction:
        correction = SELF_CORRECTION_INSERT.format(**values)
        # Insert after the first line that mentions horizontal lines
        result_lines = filled.split("\n")
        inserted = False
        for i, line in enumerate(result_lines):
            if "horizontal" in line.lower() and str(h_lines) in line:
                result_lines.insert(i + 1, correction.rstrip())
                inserted = True
                break
        if inserted:
            filled = "\n".join(result_lines)
        # Graceful fallback: if insertion point not found, skip

    return filled
