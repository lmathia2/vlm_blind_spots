"""Visual Sketchpad: pre-built vision primitives for image analysis.

Instead of asking the VLM to write code, we run tested vision primitives
programmatically and feed annotated images back to the model. The model
sees visual results (highlighted contours, numbered markers, measurement
lines) rather than text stdout.

Architecture:
  Step 0: decompose_question() → list of sub-questions
  Step 1: classify_query() per sub-question → list of (primitive, args) pairs
  Step 2: run primitives automatically (Pass 0, no model call)
  Step 3: multi-pass model-driven refinement (Passes 1-N)

Reference: Hu et al., 2024, "Visual Sketchpad" (arXiv:2410.08165)
"""

import re
from collections import defaultdict
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Annotation colors (semi-transparent overlays)
# ---------------------------------------------------------------------------

ANNOTATION_COLORS = [
    (255, 0, 0),      # red
    (0, 128, 255),     # blue
    (0, 200, 0),       # green
    (255, 165, 0),     # orange
    (128, 0, 255),     # purple
    (255, 0, 255),     # magenta
    (0, 200, 200),     # cyan
    (200, 200, 0),     # yellow
]

MAX_IMAGE_SIZE = 768  # Cap annotated images


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont:
    """Get a font, falling back to default if needed."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert image to RGB if needed."""
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _cap_size(img: Image.Image) -> Image.Image:
    """Resize image if it exceeds MAX_IMAGE_SIZE."""
    w, h = img.size
    if max(w, h) <= MAX_IMAGE_SIZE:
        return img
    scale = MAX_IMAGE_SIZE / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Vision Primitives
# ---------------------------------------------------------------------------
# Each primitive takes an image (PIL) + optional args and returns:
#   (annotated_image: Image, findings: str)

def detect_edges(image: Image.Image, **kwargs) -> tuple[Image.Image, str]:
    """Detect edges using a Laplacian-like filter and overlay on the image."""
    img = _ensure_rgb(image)
    gray = img.convert("L")

    # Edge detection via find_edges filter
    edges = gray.filter(ImageFilter.FIND_EDGES)

    # Threshold to binary
    arr = np.array(edges)
    threshold = max(30, np.percentile(arr, 85))
    binary = (arr > threshold).astype(np.uint8) * 255

    # Create red edge overlay
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    edge_pixels = np.argwhere(binary > 0)

    # Draw edges as red pixels on overlay
    edge_img = Image.fromarray(binary)
    r, g, b = img.split()
    r_arr = np.array(r)
    g_arr = np.array(g)
    b_arr = np.array(b)
    mask = binary > 0
    r_arr[mask] = 255
    g_arr[mask] = 0
    b_arr[mask] = 0
    overlay = Image.merge("RGB", (
        Image.fromarray(r_arr),
        Image.fromarray(g_arr),
        Image.fromarray(b_arr),
    ))

    n_edge_pixels = int(mask.sum())
    total_pixels = binary.size
    findings = (
        f"Edge detection: {n_edge_pixels} edge pixels found "
        f"({n_edge_pixels / total_pixels * 100:.1f}% of image). "
        f"Edges highlighted in red."
    )
    return overlay, findings


def count_line_transitions(
    image: Image.Image, axis: str = "horizontal", **kwargs,
) -> tuple[Image.Image, str]:
    """Count dark-light transitions along scan lines to detect grid lines.

    axis="horizontal": scan rows to count vertical lines
    axis="vertical": scan columns to count horizontal lines
    """
    img = _ensure_rgb(image)
    gray = np.array(img.convert("L"))
    h, w = gray.shape

    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    font = _get_font(12)

    # Sample multiple scan lines for robustness
    if axis == "horizontal":
        # Scan horizontal rows to count vertical line crossings
        scan_positions = [int(h * f) for f in [0.25, 0.4, 0.5, 0.6, 0.75]]
        counts = []
        for y in scan_positions:
            row = gray[y, :]
            # Find transitions: bright→dark or dark→bright
            dark = row < 128
            transitions = np.diff(dark.astype(int))
            # Count dark→bright transitions (entering a line)
            n_lines = int(np.sum(transitions == 1))
            counts.append(n_lines)

            # Draw the scan line
            draw.line([(0, y), (w - 1, y)], fill=(0, 128, 255), width=1)

            # Mark transition points
            transition_points = np.where(transitions == 1)[0]
            for x in transition_points:
                draw.ellipse(
                    [(int(x) - 3, y - 3), (int(x) + 3, y + 3)],
                    fill=(255, 0, 0),
                )

        # Use median count for robustness
        line_count = int(np.median(counts))
        # Draw result label
        draw.text(
            (10, 10), f"Vertical lines: {line_count}",
            fill=(255, 0, 0), font=font,
        )
        findings = (
            f"Horizontal scan: detected {line_count} vertical line transitions "
            f"(scanned {len(scan_positions)} rows, counts={counts}). "
            f"Scan lines shown in blue, transitions marked in red."
        )
    else:
        # Scan vertical columns to count horizontal line crossings
        scan_positions = [int(w * f) for f in [0.25, 0.4, 0.5, 0.6, 0.75]]
        counts = []
        for x in scan_positions:
            col = gray[:, x]
            dark = col < 128
            transitions = np.diff(dark.astype(int))
            n_lines = int(np.sum(transitions == 1))
            counts.append(n_lines)

            draw.line([(x, 0), (x, h - 1)], fill=(0, 128, 255), width=1)
            transition_points = np.where(transitions == 1)[0]
            for y in transition_points:
                draw.ellipse(
                    [(x - 3, int(y) - 3), (x + 3, int(y) + 3)],
                    fill=(255, 0, 0),
                )

        line_count = int(np.median(counts))
        draw.text(
            (10, 10), f"Horizontal lines: {line_count}",
            fill=(255, 0, 0), font=font,
        )
        findings = (
            f"Vertical scan: detected {line_count} horizontal line transitions "
            f"(scanned {len(scan_positions)} columns, counts={counts}). "
            f"Scan lines shown in blue, transitions marked in red."
        )

    return overlay, findings


def detect_contours(
    image: Image.Image, min_area: int = 50, **kwargs,
) -> tuple[Image.Image, str]:
    """Detect closed contours (shapes) via edge detection and flood fill.

    Returns annotated image with each contour outlined in a different color.
    """
    img = _ensure_rgb(image)
    gray = np.array(img.convert("L"))
    h, w = gray.shape

    # Edge detection
    edges = np.array(img.convert("L").filter(ImageFilter.FIND_EDGES))
    threshold = max(30, np.percentile(edges, 80))
    binary = (edges > threshold).astype(np.uint8)

    # Find connected components of non-edge (interior) regions
    # Use simple flood fill on the binary edge map
    visited = np.zeros_like(binary, dtype=bool)
    contours = []

    def flood_fill_area(start_y, start_x):
        """BFS flood fill, returns area and bounding box."""
        if visited[start_y, start_x] or binary[start_y, start_x] == 1:
            return 0, None
        stack = [(start_y, start_x)]
        area = 0
        min_y, max_y = start_y, start_y
        min_x, max_x = start_x, start_x
        while stack:
            cy, cx = stack.pop()
            if cy < 0 or cy >= h or cx < 0 or cx >= w:
                continue
            if visited[cy, cx] or binary[cy, cx] == 1:
                continue
            visited[cy, cx] = True
            area += 1
            min_y, max_y = min(min_y, cy), max(max_y, cy)
            min_x, max_x = min(min_x, cx), max(max_x, cx)
            stack.extend([
                (cy - 1, cx), (cy + 1, cx),
                (cy, cx - 1), (cy, cx + 1),
            ])
        bbox = (min_x, min_y, max_x, max_y)
        return area, bbox

    # Sample points in a grid pattern rather than every pixel
    step = max(1, min(w, h) // 50)
    for y in range(0, h, step):
        for x in range(0, w, step):
            if not visited[y, x] and binary[y, x] == 0:
                area, bbox = flood_fill_area(y, x)
                if area >= min_area and bbox is not None:
                    # Filter out the background (largest region)
                    bw = bbox[2] - bbox[0]
                    bh = bbox[3] - bbox[1]
                    if bw < w * 0.95 and bh < h * 0.95:
                        contours.append((area, bbox))

    # Sort by area (largest first) and deduplicate overlapping bboxes
    contours.sort(key=lambda x: x[0], reverse=True)
    unique_contours = []
    for area, bbox in contours:
        # Check if this bbox overlaps significantly with an existing one
        is_dup = False
        for _, existing_bbox in unique_contours:
            # Compute IoU-like overlap
            ix0 = max(bbox[0], existing_bbox[0])
            iy0 = max(bbox[1], existing_bbox[1])
            ix1 = min(bbox[2], existing_bbox[2])
            iy1 = min(bbox[3], existing_bbox[3])
            if ix0 < ix1 and iy0 < iy1:
                inter = (ix1 - ix0) * (iy1 - iy0)
                area_this = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if inter > area_this * 0.5:
                    is_dup = True
                    break
        if not is_dup:
            unique_contours.append((area, bbox))

    # Annotate
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    font = _get_font(14)

    for i, (area, bbox) in enumerate(unique_contours):
        color = ANNOTATION_COLORS[i % len(ANNOTATION_COLORS)]
        draw.rectangle(
            [(bbox[0], bbox[1]), (bbox[2], bbox[3])],
            outline=color, width=2,
        )
        draw.text(
            (bbox[0] + 2, bbox[1] - 16),
            f"#{i + 1} ({area}px²)",
            fill=color, font=font,
        )

    n = len(unique_contours)
    draw.text(
        (10, 10), f"Contours: {n}",
        fill=(255, 0, 0), font=font,
    )

    contour_desc = "; ".join(
        f"#{i+1}: bbox=({b[0]},{b[1]},{b[2]},{b[3]}), area={a}px²"
        for i, (a, b) in enumerate(unique_contours[:10])
    )
    findings = (
        f"Contour detection: found {n} distinct regions "
        f"(min_area={min_area}). {contour_desc}"
    )
    return overlay, findings


def segment_colors(
    image: Image.Image,
    exclude_white: bool = True,
    exclude_black: bool = False,
    **kwargs,
) -> tuple[Image.Image, str]:
    """Segment image into distinct color regions and compute area ratios.

    Works by clustering pixels in HSV space. Returns annotated image with
    percentage labels overlaid on each color region.
    """
    img = _ensure_rgb(image)
    arr = np.array(img)
    h, w, _ = arr.shape
    total_pixels = h * w

    # Convert to HSV for better color clustering
    # Manual RGB→HSV since we only need hue
    r, g, b = arr[:, :, 0].astype(float), arr[:, :, 1].astype(float), arr[:, :, 2].astype(float)
    cmax = np.maximum(r, np.maximum(g, b))
    cmin = np.minimum(r, np.minimum(g, b))
    delta = cmax - cmin

    # Compute saturation
    sat = np.where(cmax > 0, delta / cmax, 0)

    # Create masks for exclusion
    mask = np.ones((h, w), dtype=bool)
    if exclude_white:
        # Exclude near-white pixels (low saturation, high value)
        white_mask = (cmax > 220) & (sat < 0.15)
        mask &= ~white_mask
    if exclude_black:
        # Exclude near-black pixels
        black_mask = cmax < 35
        mask &= ~black_mask

    # Cluster by dominant color using simple hue binning
    hue = np.zeros((h, w))
    # Where delta > 0, compute hue
    nonzero = delta > 0
    # Red channel dominant
    r_dom = (cmax == r) & nonzero
    hue[r_dom] = (60 * ((g[r_dom] - b[r_dom]) / delta[r_dom]) + 360) % 360
    # Green channel dominant
    g_dom = (cmax == g) & nonzero
    hue[g_dom] = 60 * ((b[g_dom] - r[g_dom]) / delta[g_dom]) + 120
    # Blue channel dominant
    b_dom = (cmax == b) & nonzero
    hue[b_dom] = 60 * ((r[b_dom] - g[b_dom]) / delta[b_dom]) + 240

    # Also handle gray/achromatic pixels (sat < 0.1 but not white/black)
    gray_mask = (sat < 0.1) & mask
    colored_mask = (sat >= 0.1) & mask

    # Bin hues into color categories (30° bins)
    hue_bins = (hue[colored_mask] / 30).astype(int) % 12
    hue_names = [
        "red", "orange", "yellow", "lime",
        "green", "teal", "cyan", "azure",
        "blue", "violet", "magenta", "rose",
    ]

    # Count pixels per color
    color_counts = {}
    for bin_idx in range(12):
        count = int(np.sum(hue_bins == bin_idx))
        if count > total_pixels * 0.01:  # at least 1% of image
            color_counts[hue_names[bin_idx]] = count

    # Add gray if significant
    gray_count = int(gray_mask.sum())
    if gray_count > total_pixels * 0.01:
        color_counts["gray"] = gray_count

    # Merge similar colors (adjacent bins with small counts)
    # e.g., "red" and "rose" often belong to the same slice
    merged = {}
    for name, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True):
        merged[name] = count

    # Compute percentages relative to non-excluded pixels
    valid_total = max(1, int(mask.sum()))
    color_pcts = {
        name: round(count / valid_total * 100, 1)
        for name, count in merged.items()
    }

    # Sort by percentage descending
    color_pcts = dict(sorted(color_pcts.items(), key=lambda x: x[1], reverse=True))

    # Annotate: find centroid of each color region and label it
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    font = _get_font(16)

    for i, (color_name, pct) in enumerate(color_pcts.items()):
        if color_name == "gray":
            region_mask = gray_mask
        else:
            bin_idx = hue_names.index(color_name)
            region_mask = np.zeros((h, w), dtype=bool)
            region_mask[colored_mask] = hue_bins == bin_idx

        # Find centroid
        ys, xs = np.where(region_mask)
        if len(ys) == 0:
            continue
        cy, cx = int(np.mean(ys)), int(np.mean(xs))

        # Draw label with background
        label = f"{color_name}: {pct}%"
        draw.rectangle(
            [(cx - 2, cy - 10), (cx + len(label) * 8, cy + 10)],
            fill=(0, 0, 0, 180),
        )
        draw.text(
            (cx, cy - 8), label,
            fill=(255, 255, 255), font=font,
        )

    pct_desc = ", ".join(f"{name}={pct}%" for name, pct in color_pcts.items())
    findings = f"Color segmentation: {pct_desc}. Total non-excluded area: {valid_total}px."
    return overlay, findings


def measure_bar_fill(
    image: Image.Image, **kwargs,
) -> tuple[Image.Image, str]:
    """Measure the fill level of progress bars in the image.

    Detects horizontal bars by looking for green (filled) and gray (empty)
    horizontal bands.
    """
    img = _ensure_rgb(image)
    arr = np.array(img)
    h, w, _ = arr.shape

    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    font = _get_font(14)

    # Find green-ish pixels (fill color: ~#4CAF50)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    green_mask = (g > 120) & (g > r + 20) & (g > b + 20)

    # Find gray pixels (bar background: ~#E0E0E0)
    gray_mask = (
        (np.abs(r.astype(int) - g.astype(int)) < 30) &
        (np.abs(g.astype(int) - b.astype(int)) < 30) &
        (r > 150) & (r < 240)
    )

    # Find horizontal bands where green or gray dominate
    bar_regions = []
    in_bar = False
    bar_start = 0

    for y in range(h):
        row_green = green_mask[y, :].sum()
        row_gray = gray_mask[y, :].sum()
        is_bar_row = (row_green + row_gray) > w * 0.3

        if is_bar_row and not in_bar:
            bar_start = y
            in_bar = True
        elif not is_bar_row and in_bar:
            bar_end = y
            if bar_end - bar_start > 5:  # min height
                bar_regions.append((bar_start, bar_end))
            in_bar = False

    if in_bar:
        bar_regions.append((bar_start, h))

    # Measure fill for each bar
    measurements = []
    for i, (y_start, y_end) in enumerate(bar_regions):
        bar_green = green_mask[y_start:y_end, :]
        bar_gray = gray_mask[y_start:y_end, :]

        # Find leftmost and rightmost extent of bar (green or gray)
        bar_any = bar_green | bar_gray
        col_sums = bar_any.sum(axis=0)
        bar_cols = np.where(col_sums > 0)[0]
        if len(bar_cols) == 0:
            continue

        bar_left = int(bar_cols[0])
        bar_right = int(bar_cols[-1])
        total_width = bar_right - bar_left + 1

        # Find rightmost green pixel
        green_cols = np.where(bar_green.sum(axis=0) > 0)[0]
        if len(green_cols) == 0:
            fill_width = 0
        else:
            fill_width = int(green_cols[-1]) - bar_left + 1

        pct = round(fill_width / max(1, total_width) * 100, 1)
        measurements.append(pct)

        # Draw measurement annotation
        cy = (y_start + y_end) // 2
        color = ANNOTATION_COLORS[i % len(ANNOTATION_COLORS)]

        # Draw filled extent line
        draw.line(
            [(bar_left, cy), (bar_left + fill_width, cy)],
            fill=color, width=3,
        )
        # Draw total extent line (dashed effect)
        draw.line(
            [(bar_left + fill_width, cy), (bar_right, cy)],
            fill=(*color, 100), width=1,
        )
        # Label
        draw.text(
            (bar_right + 5, cy - 8),
            f"Bar {i + 1}: {pct}%",
            fill=color, font=font,
        )

    findings_parts = [
        f"Bar {i + 1}: {pct}% filled"
        for i, pct in enumerate(measurements)
    ]
    findings = f"Bar fill measurement: {'; '.join(findings_parts)}." if findings_parts else \
        "Bar fill measurement: no progress bars detected."
    return overlay, findings


def detect_boxes(
    image: Image.Image, **kwargs,
) -> tuple[Image.Image, str]:
    """Detect rectangular boxes (nodes in org charts, table cells).

    Uses edge detection + line scanning to find rectangular regions.
    """
    img = _ensure_rgb(image)
    gray = np.array(img.convert("L"))
    h, w = gray.shape

    # Edge detection
    edges = np.array(img.convert("L").filter(ImageFilter.FIND_EDGES))
    threshold = max(20, np.percentile(edges, 75))
    binary = (edges > threshold).astype(np.uint8)

    # Find horizontal and vertical line segments
    # Horizontal lines: rows with many edge pixels in sequence
    h_lines = []
    for y in range(h):
        row = binary[y, :]
        runs = []
        start = None
        for x in range(w):
            if row[x] > 0 and start is None:
                start = x
            elif row[x] == 0 and start is not None:
                if x - start > w * 0.05:  # min 5% of width
                    runs.append((start, x))
                start = None
        if start is not None and w - start > w * 0.05:
            runs.append((start, w))
        for x0, x1 in runs:
            h_lines.append((x0, y, x1, y))

    # Vertical lines: columns with many edge pixels in sequence
    v_lines = []
    for x in range(w):
        col = binary[:, x]
        start = None
        for y in range(h):
            if col[y] > 0 and start is None:
                start = y
            elif col[y] == 0 and start is not None:
                if y - start > h * 0.03:
                    v_lines.append((x, start, x, y))
                start = None
        if start is not None and h - start > h * 0.03:
            v_lines.append((x, start, x, h))

    # Find intersections of horizontal and vertical lines to form boxes
    # Group horizontal lines by y-coordinate (tolerance 5px)
    y_groups = defaultdict(list)
    for x0, y0, x1, y1 in h_lines:
        key = round(y0 / 5) * 5
        y_groups[key].append((x0, x1))

    # Group vertical lines by x-coordinate (tolerance 5px)
    x_groups = defaultdict(list)
    for x0, y0, x1, y1 in v_lines:
        key = round(x0 / 5) * 5
        x_groups[key].append((y0, y1))

    # Find box-like regions by looking for rectangles in gray level
    # Alternative: use connected components on the inverted binary
    # Simpler: look for regions of uniform brightness bounded by edges
    visited = np.zeros((h, w), dtype=bool)
    boxes = []

    # Scan for rectangular bright regions bounded by dark edges
    step = max(3, min(w, h) // 80)
    for sy in range(step, h - step, step):
        for sx in range(step, w - step, step):
            if visited[sy, sx]:
                continue
            if gray[sy, sx] < 180:  # skip dark regions
                continue
            if binary[sy, sx] > 0:  # skip edge pixels
                continue

            # Expand in all directions until hitting an edge
            # Right
            x_right = sx
            while x_right < w - 1 and binary[sy, x_right] == 0:
                x_right += 1
            # Left
            x_left = sx
            while x_left > 0 and binary[sy, x_left] == 0:
                x_left -= 1
            # Down
            y_down = sy
            while y_down < h - 1 and binary[y_down, sx] == 0:
                y_down += 1
            # Up
            y_up = sy
            while y_up > 0 and binary[y_up, sx] == 0:
                y_up -= 1

            box_w = x_right - x_left
            box_h = y_down - y_up
            if box_w > w * 0.04 and box_h > h * 0.02 and box_w < w * 0.9 and box_h < h * 0.9:
                # Mark as visited
                visited[y_up:y_down, x_left:x_right] = True
                boxes.append((x_left, y_up, x_right, y_down))

    # Deduplicate overlapping boxes
    unique_boxes = []
    for bbox in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
        is_dup = False
        for existing in unique_boxes:
            ix0 = max(bbox[0], existing[0])
            iy0 = max(bbox[1], existing[1])
            ix1 = min(bbox[2], existing[2])
            iy1 = min(bbox[3], existing[3])
            if ix0 < ix1 and iy0 < iy1:
                inter = (ix1 - ix0) * (iy1 - iy0)
                area_this = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if inter > area_this * 0.4:
                    is_dup = True
                    break
        if not is_dup:
            unique_boxes.append(bbox)

    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    font = _get_font(12)

    for i, bbox in enumerate(unique_boxes):
        color = ANNOTATION_COLORS[i % len(ANNOTATION_COLORS)]
        draw.rectangle(
            [(bbox[0], bbox[1]), (bbox[2], bbox[3])],
            outline=color, width=2,
        )
        draw.text(
            (bbox[0] + 2, bbox[1] + 2),
            f"Box {i + 1}",
            fill=color, font=font,
        )

    n = len(unique_boxes)
    draw.text(
        (10, 10), f"Boxes: {n}",
        fill=(255, 0, 0), font=font,
    )

    box_desc = "; ".join(
        f"Box {i+1}: ({b[0]},{b[1]})→({b[2]},{b[3]})"
        for i, b in enumerate(unique_boxes[:15])
    )
    findings = f"Box detection: found {n} rectangular regions. {box_desc}"

    return overlay, findings


def cluster_by_y(
    image: Image.Image, boxes: list[tuple] | None = None, **kwargs,
) -> tuple[Image.Image, str]:
    """Cluster detected boxes by y-coordinate to identify hierarchy levels.

    If boxes is None, runs detect_boxes first.
    """
    if boxes is None:
        _, box_findings = detect_boxes(image)
        # Re-run detect_boxes to get the actual boxes
        # (We need to extract box coordinates from the findings or re-detect)
        img = _ensure_rgb(image)
        # Use the detect_boxes logic inline to get boxes
        _, findings_text = detect_boxes(image)
        # Parse boxes from findings text
        box_matches = re.findall(
            r"Box \d+: \((\d+),(\d+)\)→\((\d+),(\d+)\)", findings_text
        )
        boxes = [
            (int(m[0]), int(m[1]), int(m[2]), int(m[3]))
            for m in box_matches
        ]

    if not boxes:
        return image.copy(), "Cluster by Y: no boxes to cluster."

    # Get y-center of each box
    y_centers = [(b[1] + b[3]) / 2 for b in boxes]

    # Cluster by y using a simple gap-based method
    sorted_indices = sorted(range(len(y_centers)), key=lambda i: y_centers[i])
    clusters = [[sorted_indices[0]]]
    for idx in sorted_indices[1:]:
        prev_y = y_centers[clusters[-1][-1]]
        curr_y = y_centers[idx]
        # Use adaptive threshold: gap must be > 30% of average box height
        avg_box_h = np.mean([b[3] - b[1] for b in boxes])
        threshold = max(avg_box_h * 0.5, 15)
        if curr_y - prev_y > threshold:
            clusters.append([idx])
        else:
            clusters[-1].append(idx)

    # Annotate: draw horizontal bands for each level
    overlay = _ensure_rgb(image).copy()
    draw = ImageDraw.Draw(overlay)
    font = _get_font(16)

    n_levels = len(clusters)
    for level_i, cluster in enumerate(clusters):
        color = ANNOTATION_COLORS[level_i % len(ANNOTATION_COLORS)]
        # Find y-range for this cluster
        y_min = min(boxes[i][1] for i in cluster)
        y_max = max(boxes[i][3] for i in cluster)

        # Draw semi-transparent band
        for y in range(y_min - 5, min(y_max + 5, overlay.size[1])):
            draw.line(
                [(0, y), (30, y)],
                fill=(*color, 80), width=1,
            )

        # Label
        draw.text(
            (5, y_min),
            f"L{level_i + 1}",
            fill=color, font=font,
        )

    draw.text(
        (10, 10), f"Levels: {n_levels}",
        fill=(255, 0, 0), font=font,
    )

    level_desc = "; ".join(
        f"Level {i+1}: {len(c)} boxes"
        for i, c in enumerate(clusters)
    )
    findings = f"Y-clustering: {n_levels} horizontal levels. {level_desc}"
    return overlay, findings


def crop_and_enhance(
    image: Image.Image, region: str = "center", **kwargs,
) -> tuple[Image.Image, str]:
    """Crop a region of the image and enhance it (sharpen, increase contrast).

    Regions: "center", "top-left", "top-right", "bottom-left", "bottom-right",
             "top", "bottom", "left", "right", or "full" (just enhance).
    """
    img = _ensure_rgb(image)
    w, h = img.size

    # Define crop box based on region
    region_map = {
        "center": (w // 4, h // 4, 3 * w // 4, 3 * h // 4),
        "top-left": (0, 0, w // 2, h // 2),
        "top-right": (w // 2, 0, w, h // 2),
        "bottom-left": (0, h // 2, w // 2, h),
        "bottom-right": (w // 2, h // 2, w, h),
        "top": (0, 0, w, h // 2),
        "bottom": (0, h // 2, w, h),
        "left": (0, 0, w // 2, h),
        "right": (w // 2, 0, w, h),
        "full": (0, 0, w, h),
    }

    bbox = region_map.get(region, region_map["center"])
    cropped = img.crop(bbox)

    # Enhance: sharpen + increase contrast
    from PIL import ImageEnhance
    enhanced = ImageEnhance.Sharpness(cropped).enhance(2.0)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.5)

    # Upscale to at least 512px on short side
    ew, eh = enhanced.size
    min_side = min(ew, eh)
    if min_side < 512:
        scale = 512 / min_side
        enhanced = enhanced.resize(
            (int(ew * scale), int(eh * scale)), Image.LANCZOS,
        )

    findings = (
        f"Crop & enhance: region='{region}', "
        f"crop box=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}), "
        f"enhanced with 2x sharpness + 1.5x contrast, "
        f"upscaled to {enhanced.size[0]}x{enhanced.size[1]}."
    )
    return enhanced, findings


def detect_points(
    image: Image.Image, target_color: str = "blue", **kwargs,
) -> tuple[Image.Image, str]:
    """Detect colored marker points in a scatter plot.

    Identifies colored dots/markers and estimates their data coordinates
    by mapping pixel positions relative to the plot area.
    """
    img = _ensure_rgb(image)
    arr = np.array(img)
    h, w, _ = arr.shape

    # Color ranges for common marker colors (RGB)
    color_ranges = {
        "blue": {"r": (0, 100), "g": (80, 180), "b": (140, 255)},
        "red": {"r": (180, 255), "g": (0, 100), "b": (0, 100)},
        "green": {"r": (0, 100), "g": (140, 255), "b": (0, 100)},
        "orange": {"r": (200, 255), "g": (100, 180), "b": (0, 80)},
    }

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    font = _get_font(12)

    points_found = []

    for color_name, ranges in color_ranges.items():
        mask = (
            (r >= ranges["r"][0]) & (r <= ranges["r"][1]) &
            (g >= ranges["g"][0]) & (g <= ranges["g"][1]) &
            (b >= ranges["b"][0]) & (b <= ranges["b"][1])
        )

        # Find clusters of colored pixels (markers)
        labeled = np.zeros_like(mask, dtype=int)
        cluster_id = 0
        cluster_centers = []

        # Simple connected components via scanning
        ys, xs = np.where(mask)
        if len(ys) < 3:
            continue

        # Use grid-based clustering
        from collections import defaultdict as dd
        cells = dd(list)
        cell_size = max(5, min(w, h) // 50)
        for y, x in zip(ys, xs):
            cells[(y // cell_size, x // cell_size)].append((y, x))

        # Merge adjacent cells
        visited_cells = set()
        for cell_key, pixels in cells.items():
            if cell_key in visited_cells or len(pixels) < 3:
                continue
            # BFS to find connected cells
            cluster_pixels = []
            stack = [cell_key]
            while stack:
                ck = stack.pop()
                if ck in visited_cells:
                    continue
                visited_cells.add(ck)
                if ck in cells:
                    cluster_pixels.extend(cells[ck])
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            neighbor = (ck[0] + dy, ck[1] + dx)
                            if neighbor not in visited_cells and neighbor in cells:
                                stack.append(neighbor)

            if len(cluster_pixels) >= 5:  # min pixels for a marker
                cy = int(np.mean([p[0] for p in cluster_pixels]))
                cx = int(np.mean([p[1] for p in cluster_pixels]))
                cluster_centers.append((cx, cy))

        for cx, cy in cluster_centers:
            points_found.append((color_name, cx, cy))
            ann_color = ANNOTATION_COLORS[
                list(color_ranges.keys()).index(color_name) % len(ANNOTATION_COLORS)
            ]
            # Draw crosshair
            draw.line([(cx - 10, cy), (cx + 10, cy)], fill=ann_color, width=2)
            draw.line([(cx, cy - 10), (cx, cy + 10)], fill=ann_color, width=2)
            draw.text(
                (cx + 12, cy - 8),
                f"({cx},{cy})",
                fill=ann_color, font=font,
            )

    n = len(points_found)
    draw.text(
        (10, 10), f"Points: {n}",
        fill=(255, 0, 0), font=font,
    )

    point_desc = "; ".join(
        f"{color} at pixel ({x},{y})"
        for color, x, y in points_found
    )
    findings = f"Point detection: found {n} marker points. {point_desc}"
    return overlay, findings


def trace_colored_paths(
    image: Image.Image, **kwargs,
) -> tuple[Image.Image, str]:
    """Trace colored paths (lines) in an image and identify their endpoints.

    Designed for colored_paths task where colored lines connect station nodes.
    """
    img = _ensure_rgb(image)
    arr = np.array(img)
    h, w, _ = arr.shape

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Detect distinct colored lines (exclude white, black, and near-gray)
    sat = np.max(arr, axis=2).astype(float) - np.min(arr, axis=2).astype(float)
    colored_mask = sat > 60  # reasonable saturation

    # Also exclude the plot background and station markers
    not_white = np.max(arr, axis=2) < 240
    not_black = np.max(arr, axis=2) > 30
    line_mask = colored_mask & not_white & not_black

    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    font = _get_font(14)

    # Cluster colored pixels by hue
    hue = np.zeros((h, w))
    cmax = np.max(arr.astype(float), axis=2)
    cmin = np.min(arr.astype(float), axis=2)
    delta = cmax - cmin
    nonzero = delta > 0

    r_f, g_f, b_f = r.astype(float), g.astype(float), b.astype(float)
    r_dom = (cmax == r_f) & nonzero
    hue[r_dom] = (60 * ((g_f[r_dom] - b_f[r_dom]) / delta[r_dom]) + 360) % 360
    g_dom = (cmax == g_f) & nonzero
    hue[g_dom] = 60 * ((b_f[g_dom] - r_f[g_dom]) / delta[g_dom]) + 120
    b_dom = (cmax == b_f) & nonzero
    hue[b_dom] = 60 * ((r_f[b_dom] - g_f[b_dom]) / delta[b_dom]) + 240

    # Bin by 60° segments (6 major colors)
    hue_bins = (hue / 60).astype(int) % 6
    color_names_6 = ["red", "yellow", "green", "cyan", "blue", "magenta"]

    path_info = []
    for bin_idx in range(6):
        bin_mask = (hue_bins == bin_idx) & line_mask
        n_pixels = int(bin_mask.sum())
        if n_pixels < 100:  # too few pixels
            continue

        ys, xs = np.where(bin_mask)
        # Find endpoints (extremes of the connected region)
        top_idx = np.argmin(ys)
        bot_idx = np.argmax(ys)
        left_idx = np.argmin(xs)
        right_idx = np.argmax(xs)

        endpoints = [
            (int(xs[top_idx]), int(ys[top_idx])),
            (int(xs[bot_idx]), int(ys[bot_idx])),
            (int(xs[left_idx]), int(ys[left_idx])),
            (int(xs[right_idx]), int(ys[right_idx])),
        ]
        # Deduplicate endpoints that are close together
        unique_endpoints = [endpoints[0]]
        for ep in endpoints[1:]:
            is_dup = any(
                abs(ep[0] - uep[0]) < 20 and abs(ep[1] - uep[1]) < 20
                for uep in unique_endpoints
            )
            if not is_dup:
                unique_endpoints.append(ep)

        # Pick the two most distant endpoints
        if len(unique_endpoints) >= 2:
            max_dist = 0
            best_pair = (unique_endpoints[0], unique_endpoints[1])
            for i, ep1 in enumerate(unique_endpoints):
                for ep2 in unique_endpoints[i + 1:]:
                    dist = ((ep1[0] - ep2[0]) ** 2 + (ep1[1] - ep2[1]) ** 2) ** 0.5
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (ep1, ep2)
            start, end = best_pair
        else:
            start = end = unique_endpoints[0]

        path_info.append({
            "color": color_names_6[bin_idx],
            "n_pixels": n_pixels,
            "start": start,
            "end": end,
        })

        # Annotate endpoints
        ann_color = ANNOTATION_COLORS[bin_idx % len(ANNOTATION_COLORS)]
        draw.ellipse(
            [(start[0] - 8, start[1] - 8), (start[0] + 8, start[1] + 8)],
            outline=ann_color, width=3,
        )
        draw.ellipse(
            [(end[0] - 8, end[1] - 8), (end[0] + 8, end[1] + 8)],
            outline=ann_color, width=3,
        )
        draw.text(
            (start[0] + 10, start[1] - 10),
            f"{color_names_6[bin_idx]}: start",
            fill=ann_color, font=font,
        )
        draw.text(
            (end[0] + 10, end[1] - 10),
            f"{color_names_6[bin_idx]}: end",
            fill=ann_color, font=font,
        )

    n_paths = len(path_info)
    path_desc = "; ".join(
        f"{p['color']} path: ({p['start'][0]},{p['start'][1]})→({p['end'][0]},{p['end'][1]})"
        for p in path_info
    )
    findings = f"Path tracing: found {n_paths} colored paths. {path_desc}"
    return overlay, findings


# ---------------------------------------------------------------------------
# Primitive Registry
# ---------------------------------------------------------------------------

PRIMITIVE_REGISTRY = {
    "detect_edges": detect_edges,
    "count_line_transitions": count_line_transitions,
    "detect_contours": detect_contours,
    "segment_colors": segment_colors,
    "measure_bar_fill": measure_bar_fill,
    "detect_boxes": detect_boxes,
    "cluster_by_y": cluster_by_y,
    "crop_and_enhance": crop_and_enhance,
    "trace_colored_paths": trace_colored_paths,
    "detect_points": detect_points,
}


# ---------------------------------------------------------------------------
# Question Decomposition
# ---------------------------------------------------------------------------

DECOMPOSITION_TEMPLATES = {
    "pie_chart": [
        "What distinct color regions exist in the image and what are their relative areas?",
        "What is the area of the target region as a percentage of the whole?",
    ],
    "counting_grid": [
        "How many horizontal lines are in the grid?",
        "How many vertical lines are in the grid?",
    ],
    "hierarchy_depth": [
        "Where are the boxes or nodes positioned in the image?",
        "How many distinct horizontal rows do the boxes form?",
    ],
    "progress_bar": [
        "Where are the progress bars in the image and what fraction is filled?",
    ],
    "colored_paths": [
        "What distinct colored paths exist in the image?",
        "What are the start and end points for each colored path?",
    ],
    "realistic_table": [
        "Where are the table cell boundaries?",
        "What is the content of the target cell?",
    ],
    "nested_squares": [
        "What rectangular contours exist in the image?",
        "How many concentric squares are there from outside to inside?",
    ],
    "scatter_plot": [
        "Where are the data points in the image?",
        "What is the y-value at the target x-coordinate?",
    ],
    "text_degradation": [
        "What text is visible in the image?",
    ],
}


def decompose_question(prompt: str, task_name: str = "") -> list[str]:
    """Decompose a question into sub-questions for structured analysis.

    Uses task-aware templates when available, falls back to query-based
    heuristics for unknown tasks.
    """
    # Use task-specific templates if available
    if task_name in DECOMPOSITION_TEMPLATES:
        return DECOMPOSITION_TEMPLATES[task_name]

    # Heuristic decomposition for unknown tasks
    # Split on "and" if the question has multiple clauses
    if " and " in prompt.lower() and "?" in prompt:
        parts = re.split(r"\band\b", prompt, flags=re.IGNORECASE)
        if len(parts) >= 2:
            return [p.strip().rstrip("?") + "?" for p in parts if len(p.strip()) > 10]

    # No decomposition needed — pass through
    return [prompt]


# ---------------------------------------------------------------------------
# Query Classification
# ---------------------------------------------------------------------------

QUERY_PATTERNS = [
    # Counting lines/grid
    (r"(grid|lines|rows.*columns|columns.*rows)", "grid_counting",
     [("count_line_transitions", {"axis": "horizontal"}),
      ("count_line_transitions", {"axis": "vertical"})]),

    # Counting shapes (squares, circles, etc.)
    (r"(count|how many).*(square|circle|shape|triangle|rectangle|contour)", "shape_counting",
     [("detect_contours", {})]),

    # Hierarchy / levels / depth
    (r"(level|depth|hierarch|layers?\b|how many.*deep|rows of boxes)", "hierarchy",
     [("detect_boxes", {}), ("cluster_by_y", {})]),

    # Percentage / proportion (pie chart, progress bar)
    (r"(percentage|percent|proportion|slice.*represent|area.*region)", "proportion",
     [("segment_colors", {})]),

    # Progress bar fill
    (r"(progress.*bar|bar.*fill|fraction.*filled)", "bar_fill",
     [("measure_bar_fill", {})]),

    # Path counting / connections
    (r"(path|route|connect|from.*to|go from)", "path_tracing",
     [("segment_colors", {"exclude_white": True, "exclude_black": True}),
      ("trace_colored_paths", {})]),

    # Table / cell lookup
    (r"(table|row|column|cell|what is the.*for)", "table_lookup",
     [("detect_boxes", {}), ("crop_and_enhance", {"region": "center"})]),

    # Scatter plot / coordinate reading
    (r"(y-value|x-value|scatter|point at|coordinate|data point)", "scatter",
     [("detect_points", {})]),

    # Text reading
    (r"(text|read|say|written|OCR|what does)", "text_reading",
     [("crop_and_enhance", {"region": "full"})]),

    # Counting (generic)
    (r"(count|how many)", "counting",
     [("detect_contours", {})]),
]

# Fallback: if no pattern matches
FALLBACK_PLAN = [("detect_edges", {}), ("detect_contours", {})]

# Task-name overrides for known tasks where query wording may be ambiguous
_TASK_PLAN_OVERRIDES = {
    "counting_grid": [
        ("count_line_transitions", {"axis": "horizontal"}),
        ("count_line_transitions", {"axis": "vertical"}),
    ],
    "pie_chart": [("segment_colors", {})],
    "colored_paths": [
        ("segment_colors", {"exclude_white": True, "exclude_black": True}),
        ("trace_colored_paths", {}),
    ],
    "nested_squares": [("detect_contours", {"min_area": 50})],
    "hierarchy_depth": [("detect_boxes", {}), ("cluster_by_y", {})],
    "realistic_table": [
        ("detect_boxes", {}),
        ("crop_and_enhance", {"region": "center"}),
    ],
    "progress_bar": [("measure_bar_fill", {})],
    "scatter_plot": [("detect_points", {})],
    "text_degradation": [("crop_and_enhance", {"region": "full"})],
}


def classify_query(
    prompt: str, task_name: str = "",
) -> list[tuple[str, dict]]:
    """Classify a query/sub-question and return a list of (primitive, kwargs) pairs.

    Priority: task_name overrides > query keyword patterns > fallback.
    """
    # Check task_name overrides first
    if task_name in _TASK_PLAN_OVERRIDES:
        return _TASK_PLAN_OVERRIDES[task_name]

    # Match query patterns
    prompt_lower = prompt.lower()
    for pattern, category, primitives in QUERY_PATTERNS:
        if re.search(pattern, prompt_lower):
            return primitives

    return FALLBACK_PLAN


# ---------------------------------------------------------------------------
# Sketchpad Orchestrator
# ---------------------------------------------------------------------------

def run_sketchpad_pass0(
    image: Image.Image,
    prompt: str,
    task_name: str = "",
) -> tuple[Image.Image, list[dict]]:
    """Run Pass 0: decompose question, classify sub-questions, execute primitives.

    Returns:
        (annotated_image, findings) where findings is a list of dicts:
        [{"sub_question": str, "primitives_run": list, "findings": str}, ...]
    """
    # Step 0: Decompose question
    sub_questions = decompose_question(prompt, task_name)

    # Current canvas starts as the original image
    canvas = _ensure_rgb(image).copy()
    all_findings = []

    for sq_i, sub_q in enumerate(sub_questions):
        # Step 1: Classify this sub-question
        plan = classify_query(sub_q, task_name)

        sq_findings_parts = []
        primitives_run = []

        for prim_name, prim_kwargs in plan:
            prim_fn = PRIMITIVE_REGISTRY.get(prim_name)
            if prim_fn is None:
                continue

            # Special handling: cluster_by_y needs boxes from detect_boxes
            if prim_name == "cluster_by_y":
                # Run on the current canvas (which may have box annotations)
                annotated, finding = prim_fn(canvas, **prim_kwargs)
            else:
                annotated, finding = prim_fn(canvas, **prim_kwargs)

            # For crop_and_enhance, we keep the enhanced crop separately
            # but also annotate the main canvas
            if prim_name == "crop_and_enhance":
                # The annotated image IS the crop — store it but keep canvas
                sq_findings_parts.append(finding)
                primitives_run.append(prim_name)
                # We'll include the enhanced crop in the final composite
                continue

            # Update canvas with annotations from this primitive
            canvas = annotated
            sq_findings_parts.append(finding)
            primitives_run.append(prim_name)

        all_findings.append({
            "sub_question": sub_q,
            "primitives_run": primitives_run,
            "findings": " | ".join(sq_findings_parts),
        })

    # Cap the final annotated image size
    canvas = _cap_size(canvas)

    return canvas, all_findings


def build_sketchpad_prompt(
    original_prompt: str,
    findings: list[dict],
    pass_num: int = 1,
) -> str:
    """Build the prompt for model-driven passes (Pass 1+).

    Includes sub-question findings summary and available tools.
    """
    # Build findings summary
    findings_text = ""
    for i, f in enumerate(findings):
        findings_text += f"\nSub-Q{i + 1}: {f['sub_question']}\n"
        findings_text += f"  Primitives used: {', '.join(f['primitives_run'])}\n"
        findings_text += f"  Findings: {f['findings']}\n"

    available_tools = (
        "- crop_and_enhance(region): Zoom into a region (\"center\", \"top-left\", \"top-right\", \"bottom-left\", \"bottom-right\")\n"
        "- detect_contours(): Find and highlight shape boundaries\n"
        "- segment_colors(): Isolate distinct color regions with area percentages\n"
        "- count_line_transitions(axis): Count lines along \"horizontal\" or \"vertical\"\n"
        "- measure_bar_fill(): Measure filled vs total width of bars\n"
        "- detect_boxes(): Find rectangular boxes and their positions\n"
        "- cluster_by_y(): Group boxes into horizontal levels\n"
        "- detect_edges(): Highlight edges in the image\n"
        "- detect_points(): Find colored data points\n"
        "- trace_colored_paths(): Trace colored line paths"
    )

    prompt = (
        f"You are analyzing an image with a visual sketchpad. The image has been "
        f"annotated with analysis results from vision tools.\n\n"
        f"The question was decomposed into sub-questions. Analysis so far (pass {pass_num}):\n"
        f"{findings_text}\n"
        f"Available tools (reply with TOOL(name, args) to use one):\n"
        f"{available_tools}\n\n"
        f"Synthesize the sub-question findings and reply with ANSWER followed by "
        f"your response to the original question:\n{original_prompt}"
    )
    return prompt


def parse_sketchpad_response(
    response: str,
) -> tuple[str, Optional[str], Optional[dict]]:
    """Parse a model response for TOOL() or ANSWER directives.

    Returns:
        (action_type, value, kwargs) where:
        - action_type is "tool", "answer", or "unknown"
        - value is the tool name or answer text
        - kwargs is the tool arguments (if action_type == "tool")
    """
    # Check for TOOL(name, args)
    tool_match = re.search(r"TOOL\((\w+)(?:,\s*(.+?))?\)", response)
    if tool_match:
        tool_name = tool_match.group(1)
        args_str = tool_match.group(2)
        kwargs = {}
        if args_str:
            # Parse simple key=value or positional args
            args_str = args_str.strip().strip('"').strip("'")
            if "=" in args_str:
                for part in args_str.split(","):
                    k, _, v = part.strip().partition("=")
                    kwargs[k.strip().strip('"').strip("'")] = v.strip().strip('"').strip("'")
            else:
                # Positional: assume it's the first arg
                # Map common primitives to their expected arg names
                arg_names = {
                    "crop_and_enhance": "region",
                    "count_line_transitions": "axis",
                    "detect_points": "target_color",
                }
                arg_name = arg_names.get(tool_name, "arg")
                kwargs[arg_name] = args_str
        return "tool", tool_name, kwargs

    # Check for ANSWER
    answer_match = re.search(r"ANSWER\s*[:]?\s*(.+)", response, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        return "answer", answer_text, None

    # If no explicit directive, treat the whole response as an answer
    return "unknown", response.strip(), None
