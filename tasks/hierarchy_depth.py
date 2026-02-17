"""Task T3.4: Hierarchy/org chart depth counting."""

from random import Random

from PIL import Image, ImageDraw, ImageFont

TASK_CONFIG = {
    "task_name": "hierarchy_depth",
    "prompt_template": None,  # dynamic per sample
    "prompt_template_v2": None,
    "parser": "integer",
    "scorer": "exact_match",
    "default_params": {
        "depth": 3,
        "branching": 2,
        "resolution": 768,
    },
    "sweep_axes": {
        "depth": [2, 3, 4, 5],
        "branching": [2, 3],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_NAMES = [
    "CEO", "CTO", "CFO", "COO", "VP Sales", "VP Eng", "VP Ops",
    "Dir A", "Dir B", "Dir C", "Dir D", "Dir E", "Dir F",
    "Mgr 1", "Mgr 2", "Mgr 3", "Mgr 4", "Mgr 5", "Mgr 6",
    "Mgr 7", "Mgr 8", "Mgr 9", "Mgr 10", "Mgr 11", "Mgr 12",
    "Team 1", "Team 2", "Team 3", "Team 4", "Team 5", "Team 6",
    "Team 7", "Team 8", "Team 9", "Team 10", "Team 11", "Team 12",
]

_call_counter = 0


def _load_font(size: int):
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _build_tree(depth: int, branching: int, rng: Random) -> list[dict]:
    """Build a tree structure. Returns flat list of nodes with level and parent info."""
    nodes = []
    name_pool = list(_NAMES)
    rng.shuffle(name_pool)
    idx = 0

    queue = [(0, -1)]  # (level, parent_idx)
    while queue:
        level, parent = queue.pop(0)
        if idx >= len(name_pool):
            break
        node = {"id": idx, "name": name_pool[idx], "level": level, "parent": parent}
        nodes.append(node)
        node_idx = idx
        idx += 1
        if level < depth - 1:
            n_children = branching if level < depth - 2 else rng.randint(1, branching)
            for _ in range(n_children):
                queue.append((level + 1, node_idx))

    return nodes


def render(
    depth: int = 3,
    branching: int = 2,
    resolution: int = 768,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    nodes = _build_tree(depth, branching, rng)

    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)
    font_size = max(8, resolution // 60)
    font = _load_font(font_size)

    # Layout: compute x positions per level
    levels: dict[int, list[int]] = {}
    for n in nodes:
        levels.setdefault(n["level"], []).append(n["id"])

    margin = 40
    bw = max(40, resolution // 10)
    bh = max(20, font_size + 10)
    level_gap = (resolution - margin * 2 - bh) / max(1, depth - 1)

    # Assign positions
    positions: dict[int, tuple[int, int]] = {}
    for level, node_ids in levels.items():
        y = margin + int(level * level_gap)
        n_count = len(node_ids)
        total_width = n_count * bw + (n_count - 1) * 10
        x_start = (resolution - total_width) // 2
        for i, nid in enumerate(node_ids):
            x = x_start + i * (bw + 10) + bw // 2
            positions[nid] = (x, y)

    # Draw connecting lines first
    for n in nodes:
        if n["parent"] >= 0:
            px, py = positions[n["parent"]]
            cx, cy = positions[n["id"]]
            draw.line([(px, py + bh), (cx, cy)], fill="black", width=2)

    # Draw boxes
    for n in nodes:
        cx, cy = positions[n["id"]]
        x0 = cx - bw // 2
        fill = "#D0E8FF" if n["level"] == 0 else "white"
        draw.rectangle([(x0, cy), (x0 + bw, cy + bh)], fill=fill, outline="black", width=2)
        bbox = draw.textbbox((0, 0), n["name"], font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((cx - tw // 2, cy + (bh - th) // 2), n["name"], fill="black", font=font)

    ground_truth = str(depth)
    prompt = (
        f"How many levels deep is this hierarchy? "
        f"Count from the top to the bottom. Put your answer in curly brackets, e.g., {{3}}."
    )

    metadata = {
        "prompt": prompt,
        "depth": depth,
        "branching": branching,
        "resolution": resolution,
        "n_nodes": len(nodes),
    }
    return img, ground_truth, metadata
