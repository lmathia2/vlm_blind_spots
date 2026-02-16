"""Task: Two filled circles with parameterized distance."""

import math
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

TASK_CONFIG = {
    "task_name": "touching_circles",
    "prompt_template": "Are the two circles touching each other? Answer Yes/No.",
    "prompt_template_v2": "Do the two circles overlap or touch? Answer Yes or No.",
    "parser": "yes_no",
    "scorer": "exact_match",
    "default_params": {
        "distance": 0.0,
        "resolution": 768,
        "rotation": "horizontal",
        "diameter": 0.2,
    },
    "sweep_axes": {
        "distance": [-0.25, -0.15, -0.05, -0.02, -0.01, 0.0,
                      0.01, 0.02, 0.05, 0.15, 0.25],
        "diameter": [0.08, 0.12, 0.15, 0.2],
        "resolution": [384, 768, 1152],
    },
}


def render(
    distance: float = 0.0,
    resolution: int = 768,
    rotation: str = "horizontal",
    diameter: float = 0.2,
) -> tuple[Image.Image, str, dict]:
    """Render two filled circles and judge whether they touch.

    Args:
        distance: Gap between circles as a fraction of 2*radius.
                  Negative = overlap, 0 = tangent, positive = gap.
        resolution: Image size in pixels (square).
        rotation: Layout angle — "horizontal", "vertical", or "diagonal".
        diameter: Circle diameter as a fraction of the image width.

    Returns:
        (image, ground_truth, metadata)
    """
    radius = diameter / 2.0
    # Center-to-center separation: 2*radius + distance*(2*radius)
    gap = 2 * radius * (1 + distance)

    # Direction vector based on rotation
    angle_map = {"horizontal": 0.0, "vertical": math.pi / 2, "diagonal": math.pi / 4}
    angle = angle_map[rotation]
    dx = math.cos(angle) * gap / 2
    dy = math.sin(angle) * gap / 2

    center = 0.5  # normalized image center
    c1 = (center - dx, center - dy)
    c2 = (center + dx, center + dy)

    dpi = 100
    figsize = resolution / dpi

    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    circle1 = plt.Circle(c1, radius, color="steelblue")
    circle2 = plt.Circle(c2, radius, color="tomato")
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Remove all whitespace around the plot
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white",
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")

    ground_truth = "Yes" if distance <= 0 else "No"
    metadata = {
        "distance": distance,
        "resolution": resolution,
        "rotation": rotation,
        "diameter": diameter,
    }
    return img, ground_truth, metadata
