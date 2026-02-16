"""Task: Line style discrimination (exact_string)."""

from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

TASK_CONFIG = {
    "task_name": "line_style",
    "prompt_template": None,  # dynamic per sample
    "prompt_template_v2": None,
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "n_lines": 3,
        "line_width": 2,
        "resolution": 512,
    },
    "sweep_axes": {
        "n_lines": [2, 3, 4],
        "line_width": [1, 2, 3],
    },
}

_LINE_COLORS = ["Blue", "Red", "Green", "Orange"]
_COLOR_HEX = {"Blue": "#1f77b4", "Red": "#d62728", "Green": "#2ca02c", "Orange": "#ff7f0e"}
_STYLE_POOL = ["solid", "dashed", "dotted", "dashdot"]

_call_counter = 0


def render(
    n_lines: int = 3,
    line_width: int = 2,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    n_lines = min(n_lines, len(_LINE_COLORS))
    colors = _LINE_COLORS[:n_lines]

    # Assign distinct styles to each line
    styles = rng.sample(_STYLE_POOL[:max(n_lines, 3)], n_lines)

    # Pick target line to ask about
    target_idx = rng.randint(0, n_lines - 1)
    target_color = colors[target_idx]
    target_style = styles[target_idx]

    # Simplify answer: map dashdot → dashdot for ground truth
    # The question lists all possible answers
    style_display = target_style

    # Build prompt with the actual style options present
    style_options = " or ".join(sorted(set(styles)))
    prompt = (
        f"What line style is the {target_color} line — {style_options}? "
        f"Put your answer in curly brackets, e.g., {{dashed}}."
    )

    # Generate line data
    x = np.linspace(0, 10, 50)
    dpi = 100
    fig_size = resolution / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)

    for i in range(n_lines):
        # Generate distinct curves
        offset = rng.uniform(-2, 2)
        amplitude = rng.uniform(1, 4)
        freq = rng.uniform(0.3, 1.5)
        phase = rng.uniform(0, 2 * np.pi)
        y = offset + amplitude * np.sin(freq * x + phase)

        ax.plot(
            x, y,
            color=_COLOR_HEX[colors[i]],
            linestyle=styles[i],
            linewidth=line_width,
            label=f"{colors[i]} ({styles[i]})" if False else colors[i],
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Multi-Series Trend")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = Image.frombytes("RGBA", (w, h), buf).convert("RGB")
    plt.close(fig)

    ground_truth = style_display
    metadata = {
        "prompt": prompt,
        "n_lines": n_lines,
        "line_width": line_width,
        "resolution": resolution,
        "colors": colors,
        "styles": styles,
        "target_color": target_color,
        "target_style": target_style,
    }
    return img, ground_truth, metadata
