"""Task T7.4: Colored path counting between stations."""

from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

TASK_CONFIG = {
    "task_name": "colored_paths",
    "prompt_template": None,  # dynamic per sample
    "prompt_template_v2": None,
    "parser": "integer",
    "scorer": "exact_match",
    "default_params": {
        "n_stations": 5,
        "n_paths": 3,
        "thickness": 6,
        "resolution": 512,
    },
    "sweep_axes": {
        "n_paths": [1, 2, 3, 4],
        "thickness": [3, 6, 10, 20],
        "resolution": [384, 512, 768],
    },
}

_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#e377c2"]
_COLOR_NAMES = ["blue", "red", "green", "orange", "purple", "pink"]

_call_counter = 0


def render(
    n_stations: int = 5,
    n_paths: int = 3,
    thickness: int = 6,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    # Place stations on a grid
    station_labels = [chr(65 + i) for i in range(n_stations)]
    angles = np.linspace(0, 2 * np.pi, n_stations, endpoint=False)
    radius = 0.35
    positions = [(0.5 + radius * np.cos(a), 0.5 + radius * np.sin(a)) for a in angles]

    # Pick start and end stations
    start_idx = 0
    end_idx = n_stations // 2

    # Generate paths between various stations, some connecting start→end
    paths_connecting = rng.randint(1, min(n_paths, 3))
    path_data = []

    for i in range(n_paths):
        color = _COLORS[i % len(_COLORS)]
        if i < paths_connecting:
            # Path from start to end via waypoints
            waypoints = [positions[start_idx]]
            n_mid = rng.randint(1, 3)
            for _ in range(n_mid):
                wx = rng.uniform(0.15, 0.85)
                wy = rng.uniform(0.15, 0.85)
                waypoints.append((wx, wy))
            waypoints.append(positions[end_idx])
            path_data.append({"waypoints": waypoints, "color": color, "connects": True})
        else:
            # Random path between other stations
            si = rng.choice([j for j in range(n_stations) if j != start_idx])
            ei = rng.choice([j for j in range(n_stations) if j != si])
            waypoints = [positions[si]]
            n_mid = rng.randint(1, 2)
            for _ in range(n_mid):
                wx = rng.uniform(0.15, 0.85)
                wy = rng.uniform(0.15, 0.85)
                waypoints.append((wx, wy))
            waypoints.append(positions[ei])
            path_data.append({"waypoints": waypoints, "color": color, "connects": False})

    # Render
    dpi = 100
    fig_size = resolution / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw paths
    for pd_ in path_data:
        xs = [p[0] for p in pd_["waypoints"]]
        ys = [p[1] for p in pd_["waypoints"]]
        ax.plot(xs, ys, color=pd_["color"], linewidth=thickness, solid_capstyle="round")

    # Draw station markers
    for i, (x, y) in enumerate(positions):
        ax.plot(x, y, "ko", markersize=12, zorder=5)
        ax.plot(x, y, "wo", markersize=8, zorder=6)
        ax.text(x, y, station_labels[i], ha="center", va="center", fontsize=10, fontweight="bold", zorder=7)

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = Image.frombytes("RGBA", (w, h), buf).convert("RGB")
    plt.close(fig)

    ground_truth = str(paths_connecting)
    prompt = (
        f"How many paths go from station {station_labels[start_idx]} to station "
        f"{station_labels[end_idx]}? Put your answer in curly brackets, e.g., {{2}}."
    )

    metadata = {
        "prompt": prompt,
        "n_stations": n_stations,
        "n_paths": n_paths,
        "paths_connecting": paths_connecting,
        "thickness": thickness,
        "resolution": resolution,
    }
    return img, ground_truth, metadata
