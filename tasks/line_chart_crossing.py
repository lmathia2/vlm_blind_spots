"""Task: Count how many times two line series cross in a business chart."""

import random
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


TASK_CONFIG = {
    "task_name": "line_chart_crossing",
    "prompt_template": (
        "How many times do the blue and red lines cross each other? "
        "Answer with a number in curly brackets, e.g., {2}."
    ),
    "prompt_template_v2": (
        "In this chart, count the number of crossing points between "
        "the blue and red lines. Answer in curly brackets, e.g., {2}."
    ),
    "parser": "integer",
    "scorer": "integer_distance",
    "default_params": {"resolution": 768, "n_points": 100, "target_crossings": None},
    "sweep_axes": {
        "resolution": [384, 512, 768, 1024],
        "target_crossings": [0, 1, 2, 3],
    },
}


def _count_sign_changes(diff: np.ndarray) -> int:
    """Count zero-crossings via sign changes in the difference array."""
    signs = np.sign(diff)
    # Remove zeros — treat them as continuation of the previous sign
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return 0
    changes = np.diff(nonzero)
    return int(np.count_nonzero(changes))


def _build_curves(rng: random.Random, n_points: int,
                  target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build two y-value arrays over a shared x that cross exactly `target` times.

    Strategy per target count:
      0 — one curve always above the other (parallel with offset)
      1 — ascending vs descending (guaranteed single cross near the middle)
      2 — parabolic curve vs flat line positioned to cross twice
      3 — sinusoidal curve vs line with slight slope

    After construction, the actual crossing count is verified.
    """
    x = np.linspace(0, n_points - 1, n_points)
    base_scale = 50 + rng.random() * 100  # realistic dollar-scale values
    offset = base_scale * 0.5

    if target == 0:
        # Two smooth curves that never cross — one always above the other
        base = base_scale + np.cumsum(np.random.RandomState(rng.randint(0, 2**31))
                                       .normal(0, 0.5, n_points))
        gap = 5 + rng.random() * 15  # guaranteed separation
        y1 = base + gap
        y2 = base - gap
        # Add gentle noise well below the gap
        noise_scale = gap * 0.15
        rs = np.random.RandomState(rng.randint(0, 2**31))
        y1 += rs.normal(0, noise_scale, n_points)
        y2 += rs.normal(0, noise_scale, n_points)

    elif target == 1:
        # One ascending, one descending — single cross near the center
        mid = n_points // 2
        slope1 = (rng.random() * 0.3 + 0.2)
        slope2 = -(rng.random() * 0.3 + 0.2)
        center_val = base_scale
        y1 = center_val + slope1 * (x - mid)
        y2 = center_val + slope2 * (x - mid)
        # Small noise that won't create extra crossings
        rs = np.random.RandomState(rng.randint(0, 2**31))
        noise_amp = min(abs(slope1), abs(slope2)) * 0.1
        y1 += rs.normal(0, noise_amp, n_points)
        y2 += rs.normal(0, noise_amp, n_points)

    elif target == 2:
        # Parabola vs flat line — crosses exactly twice
        center_val = base_scale
        amplitude = 10 + rng.random() * 20
        # Parabola opening downward, vertex above the flat line
        vertex_height = amplitude * (0.6 + rng.random() * 0.3)
        y1 = center_val + vertex_height - amplitude * ((x - n_points / 2) / (n_points / 2)) ** 2 * vertex_height
        y2 = np.full(n_points, center_val) + (rng.random() - 0.5) * 2
        # Tiny noise
        rs = np.random.RandomState(rng.randint(0, 2**31))
        y1 += rs.normal(0, 0.1, n_points)
        y2 += rs.normal(0, 0.1, n_points)

    elif target == 3:
        # Sinusoidal vs slightly sloped line — crosses exactly 3 times
        # Use 1.5 full cycles so the sine wave crosses a gently sloped line 3 times
        center_val = base_scale
        amplitude = 8 + rng.random() * 15
        freq = 1.5  # 1.5 cycles over the domain
        y1 = center_val + amplitude * np.sin(2 * np.pi * freq * x / (n_points - 1))
        # Slight slope so the line intersects the sine 3 times
        slope = amplitude * 0.01 * (rng.choice([-1, 1]))
        y2 = center_val + slope * (x - n_points / 2)
        # Tiny noise
        rs = np.random.RandomState(rng.randint(0, 2**31))
        y1 += rs.normal(0, 0.05, n_points)
        y2 += rs.normal(0, 0.05, n_points)

    else:
        raise ValueError(f"Unsupported target_crossings={target}")

    return x, y1, y2


def render(resolution: int = 768, n_points: int = 100,
           target_crossings: int | None = None,
           seed: int | None = None) -> tuple[Image.Image, str, dict]:
    """Render a business-style line chart with two series and count crossings.

    Args:
        resolution: Image size in pixels (square).
        n_points: Number of data points per series.
        target_crossings: Desired number of crossings (0-3). If None, chosen
            randomly with equal probability.
        seed: Optional RNG seed for reproducibility.

    Returns:
        (PIL.Image, ground_truth_str, metadata_dict)
    """
    rng = random.Random(seed)

    if target_crossings is None:
        target_crossings = rng.choice([0, 1, 2, 3])

    # Try to generate curves that match the target crossing count exactly
    max_attempts = 2000
    best_x, best_y1, best_y2 = None, None, None
    actual_crossings = -1

    for _ in range(max_attempts):
        attempt_rng = random.Random(rng.randint(0, 2**31))
        x, y1, y2 = _build_curves(attempt_rng, n_points, target_crossings)
        diff = y1 - y2
        crossings = _count_sign_changes(diff)
        if crossings == target_crossings:
            best_x, best_y1, best_y2 = x, y1, y2
            actual_crossings = crossings
            break

    # Fallback: use last attempt if exact match wasn't found
    if best_x is None:
        best_x, best_y1, best_y2 = x, y1, y2
        actual_crossings = crossings

    ground_truth = str(actual_crossings)

    # Quarter labels for x-axis (business-style)
    n_ticks = min(8, n_points)
    tick_indices = np.linspace(0, n_points - 1, n_ticks, dtype=int)
    tick_labels = [f"Q{(i % 4) + 1} '{21 + i // 4}" for i in range(n_ticks)]

    # Render the chart
    dpi = 100
    fig_size = resolution / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)

    ax.plot(best_x, best_y1, color="blue", linewidth=2, label="Revenue")
    ax.plot(best_x, best_y2, color="red", linewidth=2, label="Cost")

    ax.set_xlabel("Quarter", fontsize=11)
    ax.set_ylabel("Amount ($)", fontsize=11)
    ax.set_title("Revenue vs Cost", fontsize=13, pad=10)
    ax.set_xticks(best_x[tick_indices])
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=10)

    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    metadata = {
        "resolution": resolution,
        "n_points": n_points,
        "target_crossings": target_crossings,
        "actual_crossings": actual_crossings,
        "series_a": best_y1.tolist(),
        "series_b": best_y2.tolist(),
    }

    return img, ground_truth, metadata
