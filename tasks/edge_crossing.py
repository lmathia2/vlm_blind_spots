"""Task T3.3: Edge crossing disambiguation with bridge gaps."""

from random import Random

from PIL import Image, ImageDraw, ImageFont

TASK_CONFIG = {
    "task_name": "edge_crossing",
    "prompt_template": None,  # dynamic per sample
    "prompt_template_v2": None,
    "parser": "yes_no",
    "scorer": "exact_match",
    "default_params": {
        "n_nodes": 5,
        "n_edges": 6,
        "bridge_gap": 6,
        "resolution": 512,
    },
    "sweep_axes": {
        "n_nodes": [4, 5, 6],
        "n_edges": [4, 6, 8],
        "bridge_gap": [3, 6, 10],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_call_counter = 0


def _load_font(size: int):
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _segments_cross(p1, p2, p3, p4):
    """Check if segment p1-p2 crosses p3-p4. Returns intersection point or None."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if 0.05 < t < 0.95 and 0.05 < u < 0.95:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)
    return None


def render(
    n_nodes: int = 5,
    n_edges: int = 6,
    bridge_gap: int = 6,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    labels = [chr(65 + i) for i in range(n_nodes)]
    margin = resolution // 6
    # Fixed positions in a circle
    import math
    positions = []
    for i in range(n_nodes):
        angle = 2 * math.pi * i / n_nodes - math.pi / 2
        x = resolution // 2 + int((resolution // 2 - margin) * math.cos(angle))
        y = resolution // 2 + int((resolution // 2 - margin) * math.sin(angle))
        positions.append((x, y))

    # Generate random edges (no self-loops, no duplicates)
    all_possible = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    rng.shuffle(all_possible)
    edges = all_possible[:min(n_edges, len(all_possible))]

    # Build adjacency set
    adj = set()
    for i, j in edges:
        adj.add((i, j))
        adj.add((j, i))

    # Find crossings between edges to apply bridge gaps
    crossings = []
    for ei, (a, b) in enumerate(edges):
        for ej, (c, d) in enumerate(edges):
            if ej <= ei:
                continue
            pt = _segments_cross(positions[a], positions[b], positions[c], positions[d])
            if pt:
                crossings.append((ei, ej, pt))

    # Draw
    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)
    font = _load_font(max(10, resolution // 40))

    # Draw edges (bottom layer first, then top layer with bridge gaps)
    # For each crossing, the first edge (lower index) goes under
    bridge_set = set()  # edges that go under at crossings
    for ei, ej, pt in crossings:
        bridge_set.add(ei)  # lower index edge goes under

    for idx, (i, j) in enumerate(edges):
        p1, p2 = positions[i], positions[j]
        draw.line([p1, p2], fill="black", width=2)

    # Draw bridge gaps (white rectangles) over the "under" edges at crossing points
    for ei, ej, (cx, cy) in crossings:
        gap = bridge_gap
        # White rectangle at crossing for the "under" edge
        draw.rectangle([(cx - gap, cy - gap), (cx + gap, cy + gap)], fill="white")
        # Redraw the "over" edge segment through the gap area
        oi, oj = edges[ej]
        draw.line([positions[oi], positions[oj]], fill="black", width=2)

    # Draw nodes on top
    node_r = resolution // 25
    for i, (x, y) in enumerate(positions):
        draw.ellipse([(x - node_r, y - node_r), (x + node_r, y + node_r)],
                     fill="white", outline="black", width=2)
        bbox = draw.textbbox((0, 0), labels[i], font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x - tw // 2, y - th // 2), labels[i], fill="black", font=font)

    # Pick a query pair — mix connected and disconnected
    query_i = rng.randint(0, n_nodes - 1)
    query_j = rng.choice([j for j in range(n_nodes) if j != query_i])
    connected = (query_i, query_j) in adj
    ground_truth = "Yes" if connected else "No"

    prompt = (
        f"Is node {labels[query_i]} directly connected to node {labels[query_j]}? "
        f"Answer Yes or No."
    )

    metadata = {
        "prompt": prompt,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "bridge_gap": bridge_gap,
        "resolution": resolution,
        "edges": [(labels[i], labels[j]) for i, j in edges],
        "query": (labels[query_i], labels[query_j]),
        "n_crossings": len(crossings),
    }
    return img, ground_truth, metadata
