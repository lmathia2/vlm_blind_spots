"""Task: Count intersection points of two piecewise-linear paths."""

import random
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


TASK_CONFIG = {
    "task_name": "line_intersection",
    "prompt_template": (
        "Count the intersection points where the blue and red lines meet. "
        "Put your answer in curly brackets, e.g., {2}."
    ),
    "parser": "integer",
    "scorer": "integer_distance",
    "default_params": {"resolution": 512, "linewidth": 2, "grid_size": 6},
    "sweep_axes": {
        "linewidth": [1, 1.5, 2, 3, 4, 5, 8],
        "resolution": [384, 512, 768, 1024, 1152],
    },
}


def _cross(o: tuple[float, float], a: tuple[float, float],
           b: tuple[float, float]) -> float:
    """Cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(p: tuple[float, float], q: tuple[float, float],
                r: tuple[float, float]) -> bool:
    """Check if point q lies on segment pr (given p, q, r are collinear)."""
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))


def _segments_intersect(p1: tuple[float, float], q1: tuple[float, float],
                        p2: tuple[float, float], q2: tuple[float, float]) -> bool:
    """Check if segment p1-q1 intersects segment p2-q2 using cross product orientation."""
    d1 = _cross(p2, q2, p1)
    d2 = _cross(p2, q2, q1)
    d3 = _cross(p1, q1, p2)
    d4 = _cross(p1, q1, q2)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear cases
    if d1 == 0 and _on_segment(p2, p1, q2):
        return True
    if d2 == 0 and _on_segment(p2, q1, q2):
        return True
    if d3 == 0 and _on_segment(p1, p2, q1):
        return True
    if d4 == 0 and _on_segment(p1, q2, q1):
        return True

    return False


def _count_intersections(path_a: list[tuple[float, float]],
                         path_b: list[tuple[float, float]]) -> int:
    """Count the number of intersection points between two piecewise-linear paths."""
    count = 0
    for i in range(len(path_a) - 1):
        for j in range(len(path_b) - 1):
            if _segments_intersect(path_a[i], path_a[i + 1],
                                   path_b[j], path_b[j + 1]):
                count += 1
    return count


def _generate_path(rng: random.Random, grid_size: int) -> list[tuple[float, float]]:
    """Generate a random piecewise-linear path with 3 control points on a grid."""
    points = []
    for _ in range(3):
        x = rng.randint(0, grid_size)
        y = rng.randint(0, grid_size)
        points.append((float(x), float(y)))
    return points


def _paths_are_valid(path_a: list[tuple[float, float]],
                     path_b: list[tuple[float, float]]) -> bool:
    """Reject degenerate paths (duplicate consecutive points)."""
    for path in (path_a, path_b):
        for i in range(len(path) - 1):
            if path[i] == path[i + 1]:
                return False
    return True


def render(resolution: int = 512, linewidth: int = 2, grid_size: int = 6,
           target_intersections: int | None = None,
           seed: int | None = None) -> tuple[Image.Image, str, dict]:
    """Render two piecewise-linear paths and count their intersections.

    Args:
        resolution: Image size in pixels (square).
        linewidth: Line width for the paths.
        grid_size: Control points are placed on an integer grid [0, grid_size].
        target_intersections: If set, regenerate until this intersection count
            is achieved. If None, randomly pick 0, 1, or 2 with equal probability.
        seed: Optional RNG seed for reproducibility.

    Returns:
        (PIL.Image, ground_truth_str, metadata_dict)
    """
    rng = random.Random(seed)

    # Decide target intersection count if not specified
    if target_intersections is None:
        target_intersections = rng.choice([0, 1, 2])

    max_attempts = 5000
    attempt = 0
    path_a = []
    path_b = []
    n_intersections = -1

    while attempt < max_attempts:
        # Use a deterministic seed per attempt so results are reproducible
        attempt_rng = random.Random(rng.randint(0, 2**31))
        path_a = _generate_path(attempt_rng, grid_size)
        path_b = _generate_path(attempt_rng, grid_size)

        if not _paths_are_valid(path_a, path_b):
            attempt += 1
            continue

        n_intersections = _count_intersections(path_a, path_b)
        if n_intersections == target_intersections:
            break
        attempt += 1

    ground_truth = str(n_intersections)

    # Render with matplotlib — no axes, ticks, labels, or whitespace
    dpi = 100
    fig_size = resolution / dpi
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # fill the entire figure

    ax.plot([p[0] for p in path_a], [p[1] for p in path_a],
            color="blue", linewidth=linewidth, solid_capstyle="round",
            solid_joinstyle="round")
    ax.plot([p[0] for p in path_b], [p[1] for p in path_b],
            color="red", linewidth=linewidth, solid_capstyle="round",
            solid_joinstyle="round")

    ax.set_xlim(-0.5, grid_size + 0.5)
    ax.set_ylim(-0.5, grid_size + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    metadata = {
        "resolution": resolution,
        "linewidth": linewidth,
        "grid_size": grid_size,
        "path_blue": path_a,
        "path_red": path_b,
        "intersections": n_intersections,
        "target_intersections": target_intersections,
    }

    return img, ground_truth, metadata
