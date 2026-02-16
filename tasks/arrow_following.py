"""Task: Follow arrows through a DAG of labeled boxes to find the terminal box."""

import string
from io import BytesIO
from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

TASK_CONFIG = {
    "task_name": "arrow_following",
    "prompt_template": None,  # filled dynamically per sample
    "prompt_template_v2": None,  # dynamic v2 prompt set in render()
    "parser": "letter",
    "scorer": "set_member",
    "default_params": {
        "n_boxes": 5,
        "n_arrows": 5,
        "arrow_width": 2,
        "resolution": 512,
    },
    "sweep_axes": {
        "n_boxes": [4, 5, 6, 7],
        "n_arrows": [3, 4, 5, 6, 7],
    },
}

_call_counter = 0


def _grid_positions(n: int) -> list[tuple[float, float]]:
    """Compute box center positions on a grid layout in [0, 1] space.

    Arranges boxes in rows, roughly sqrt(n) columns wide, with even spacing.
    """
    import math
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    margin = 0.12
    x_span = 1.0 - 2 * margin
    y_span = 1.0 - 2 * margin

    positions = []
    for i in range(n):
        r = i // cols
        c = i % cols
        # Center boxes in each row (handle last row with fewer items)
        items_in_row = min(cols, n - r * cols)
        x = margin + (c + 0.5) * x_span / items_in_row if items_in_row > 1 else 0.5
        # Use even spacing; flip y so row 0 is at top
        y = margin + (rows - 1 - r + 0.5) * y_span / rows if rows > 1 else 0.5
        # Re-center horizontally for rows with fewer items
        if items_in_row < cols and items_in_row > 1:
            row_width = (items_in_row - 1) * (x_span / max(cols - 1, 1))
            x_start = 0.5 - row_width / 2
            x = x_start + c * (x_span / max(cols - 1, 1))
        positions.append((x, y))
    return positions


def _generate_dag_edges(n_boxes: int, n_arrows: int, rng: Random) -> list[tuple[int, int]]:
    """Generate n_arrows random directed edges forming a DAG (no cycles).

    Uses a topological ordering trick: only allow edges from lower to higher
    index in a random permutation, guaranteeing acyclicity.
    """
    # Random permutation defines topological order
    order = list(range(n_boxes))
    rng.shuffle(order)
    rank = [0] * n_boxes
    for pos, node in enumerate(order):
        rank[node] = pos

    # Collect all valid DAG edges (from lower rank to higher rank)
    all_possible = []
    for i in range(n_boxes):
        for j in range(n_boxes):
            if rank[i] < rank[j]:
                all_possible.append((i, j))

    rng.shuffle(all_possible)
    n_arrows = min(n_arrows, len(all_possible))
    return all_possible[:n_arrows]


def _follow_path(start: int, edges: list[tuple[int, int]], rng: Random) -> list[int]:
    """Follow a path from start through the DAG, choosing randomly at branches.

    Returns the full path from start to the terminal (dead-end) node.
    """
    # Build adjacency list
    adj: dict[int, list[int]] = {}
    for src, dst in edges:
        adj.setdefault(src, []).append(dst)

    path = [start]
    current = start
    while current in adj and adj[current]:
        next_node = rng.choice(adj[current])
        path.append(next_node)
        current = next_node
    return path


def _all_terminals(start: int, adj: dict[int, list[int]]) -> set[int]:
    """Find all terminal (dead-end) nodes reachable from start via DFS."""
    terminals = set()
    stack = [start]
    visited = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node not in adj or not adj[node]:
            terminals.add(node)
        else:
            for neighbor in adj[node]:
                stack.append(neighbor)
    return terminals


def render(
    n_boxes: int = 5,
    n_arrows: int = 5,
    arrow_width: float = 2,
    resolution: int = 512,
    seed: int | None = None,
    prompt_variant: int = 1,
) -> tuple[Image.Image, str, dict]:
    """Render labeled boxes connected by arrows forming a DAG.

    Picks a random start box, follows a path to the terminal box,
    and asks which box is reached last.

    Returns:
        (image, ground_truth_letter, metadata)
    """
    global _call_counter
    _call_counter += 1

    if seed is not None:
        rng = Random(seed)
    else:
        rng = Random(_call_counter)

    labels = list(string.ascii_uppercase[:n_boxes])

    # Regenerate until the start box has at least one outgoing path
    max_regen = 200
    for _ in range(max_regen):
        edges = _generate_dag_edges(n_boxes, n_arrows, rng)

        # Build adjacency for outgoing check
        out_nodes = {src for src, _ in edges}
        candidates = [i for i in range(n_boxes) if i in out_nodes]
        if not candidates:
            continue

        start = rng.choice(candidates)
        path = _follow_path(start, edges, rng)

        # Path must have length > 1 (start has at least one outgoing arrow)
        if len(path) > 1:
            break
    else:
        # Fallback: force a simple two-node path
        edges = [(0, 1)]
        start = 0
        path = [0, 1]

    terminal = path[-1]
    start_label = labels[start]
    terminal_label = labels[terminal]

    # Compute all valid terminal boxes reachable from start
    adj: dict[int, list[int]] = {}
    for src, dst in edges:
        adj.setdefault(src, []).append(dst)
    all_terminal_indices = _all_terminals(start, adj)
    all_terminal_labels = sorted(labels[i] for i in all_terminal_indices)

    prompt = (
        f"Starting at box {start_label}, follow the arrows. "
        f"What is the last box you reach? "
        f"Answer with just the letter in curly brackets, e.g., {{C}}."
    )

    if prompt_variant == 2:
        prompt = (
            f"Follow the directed arrows starting from box {start_label}. "
            f"Which box do you end up at? "
            f"Answer with just the letter in curly brackets, e.g., {{C}}."
        )

    # --- Render with matplotlib ---
    positions = _grid_positions(n_boxes)
    box_w = 0.10
    box_h = 0.08

    dpi = 100
    fig_size = resolution / dpi
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Draw boxes
    for i, (cx, cy) in enumerate(positions):
        rect = patches.FancyBboxPatch(
            (cx - box_w / 2, cy - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.01",
            linewidth=1.5,
            edgecolor="black",
            facecolor="#e0e0e0",
        )
        ax.add_patch(rect)
        ax.text(
            cx, cy, labels[i],
            ha="center", va="center",
            fontsize=14, fontweight="bold", color="black",
        )

    # Draw arrows between boxes
    for src, dst in edges:
        sx, sy = positions[src]
        dx, dy = positions[dst]

        # Shorten arrow so it starts/ends at box edge, not center
        import math
        angle = math.atan2(dy - sy, dx - sx)

        # Offset start point to box edge
        start_x = sx + math.cos(angle) * (box_w / 2 + 0.01)
        start_y = sy + math.sin(angle) * (box_h / 2 + 0.01)

        # Offset end point to box edge
        end_x = dx - math.cos(angle) * (box_w / 2 + 0.01)
        end_y = dy - math.sin(angle) * (box_h / 2 + 0.01)

        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                lw=arrow_width,
                mutation_scale=15,
                shrinkA=0,
                shrinkB=0,
            ),
        )

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    edge_labels = [(labels[s], labels[d]) for s, d in edges]
    path_labels = [labels[n] for n in path]

    ground_truth = ",".join(all_terminal_labels)
    metadata = {
        "prompt": prompt,
        "start_box": start_label,
        "terminal_box": terminal_label,
        "valid_terminals": all_terminal_labels,
        "edges": edge_labels,
        "path": path_labels,
        "n_boxes": n_boxes,
        "n_arrows": len(edges),
        "arrow_width": arrow_width,
        "resolution": resolution,
    }
    return img, ground_truth, metadata
