"""Task: Venn diagram region identification.

Renders 2–4 overlapping labeled circles with colored regions.
Questions ask which items belong to a specific region (intersection, union, exclusive).
"""

from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from io import BytesIO

TASK_CONFIG = {
    "task_name": "venn_diagram",
    "prompt_template": None,  # dynamic per sample
    "prompt_template_v2": None,
    "parser": "csv_words",
    "scorer": "set_match",
    "default_params": {
        "n_circles": 2,
        "question_type": "intersection",
        "resolution": 512,
    },
    "sweep_axes": {
        "n_circles": [2, 3, 4],
        "question_type": ["intersection", "exclusive", "union_minus"],
    },
}

_SET_NAMES = ["A", "B", "C", "D"]
_SET_COLORS = ["#3B82F6", "#EF4444", "#22C55E", "#F59E0B"]  # blue, red, green, amber
_SET_COLORS_ALPHA = [
    (0.23, 0.51, 0.96, 0.25),
    (0.94, 0.27, 0.27, 0.25),
    (0.13, 0.77, 0.33, 0.25),
    (0.96, 0.62, 0.04, 0.25),
]

_ITEM_POOLS = {
    "fruits": ["Apple", "Banana", "Cherry", "Date", "Fig", "Grape", "Kiwi", "Lemon",
               "Mango", "Orange", "Peach", "Plum"],
    "animals": ["Cat", "Dog", "Eagle", "Fox", "Goat", "Horse", "Ibis", "Jaguar",
                "Koala", "Lion", "Mouse", "Newt"],
    "colors": ["Aqua", "Bronze", "Coral", "Denim", "Ebony", "Fawn", "Gold", "Hazel",
               "Ivory", "Jade", "Khaki", "Lilac"],
}

# Circle center positions for 2, 3, 4 circles (in 0-1 normalized coords)
_LAYOUTS = {
    2: [(0.38, 0.5), (0.62, 0.5)],
    3: [(0.38, 0.58), (0.62, 0.58), (0.50, 0.38)],
    4: [(0.35, 0.58), (0.65, 0.58), (0.42, 0.38), (0.58, 0.38)],
}

_RADII = {2: 0.22, 3: 0.20, 4: 0.18}

_call_counter = 0


def _assign_items_to_regions(n_circles: int, rng: Random) -> dict[frozenset[int], list[str]]:
    """Assign items to Venn diagram regions.

    Returns a dict mapping frozenset of circle indices → list of items in that region.
    Each item belongs to exactly one region (the most specific one).
    """
    pool_name = rng.choice(list(_ITEM_POOLS.keys()))
    items = list(_ITEM_POOLS[pool_name])
    rng.shuffle(items)

    # Generate non-empty subsets of circle indices as candidate regions
    circle_indices = list(range(n_circles))
    regions: list[frozenset[int]] = []
    for mask in range(1, 1 << n_circles):
        region = frozenset(i for i in circle_indices if mask & (1 << i))
        regions.append(region)

    # For 4 circles, skip 3+ way intersections to avoid crowding the center.
    # Keep: exclusives (size 1) and pairwise intersections (size 2).
    if n_circles >= 4:
        regions = [r for r in regions if len(r) <= 2]

    # Allocate 1 item per intersection region, 1-2 per exclusive region
    assignment: dict[frozenset[int], list[str]] = {}
    item_idx = 0

    # First pass: pairwise intersections get 1 item each
    intersections = [r for r in regions if len(r) > 1]
    rng.shuffle(intersections)
    for region in intersections:
        if item_idx >= len(items):
            break
        assignment[region] = items[item_idx:item_idx + 1]
        item_idx += 1

    # Second pass: exclusive regions get 1-2 items each
    exclusives = [r for r in regions if len(r) == 1]
    rng.shuffle(exclusives)
    for region in exclusives:
        if item_idx >= len(items):
            break
        n_items = rng.randint(1, 2)
        n_items = min(n_items, len(items) - item_idx)
        assignment[region] = items[item_idx:item_idx + n_items]
        item_idx += n_items

    return assignment


def _items_in_set(assignment: dict[frozenset[int], list[str]], set_idx: int) -> list[str]:
    """Get all items that belong to a given set (including intersections)."""
    result = []
    for region, items in assignment.items():
        if set_idx in region:
            result.extend(items)
    return sorted(result)


def _items_exclusive_to(assignment: dict[frozenset[int], list[str]], set_idx: int) -> list[str]:
    """Get items that belong ONLY to the given set (not shared with any other)."""
    key = frozenset([set_idx])
    return sorted(assignment.get(key, []))


def _items_in_intersection(assignment: dict[frozenset[int], list[str]], *set_indices: int) -> list[str]:
    """Get items in the intersection of the given sets (exactly those sets, not more)."""
    key = frozenset(set_indices)
    # Items in all specified sets (could be in broader intersections too)
    result = []
    for region, items in assignment.items():
        if key.issubset(region):
            result.extend(items)
    return sorted(result)


def _generate_question(
    n_circles: int,
    assignment: dict[frozenset[int], list[str]],
    question_type: str,
    rng: Random,
) -> tuple[str, str]:
    """Generate a question and ground truth answer.

    Returns (prompt, ground_truth).
    """
    names = _SET_NAMES[:n_circles]

    if question_type == "intersection":
        if n_circles >= 2:
            # Pick 2 sets to ask about intersection
            pair = rng.sample(range(n_circles), 2)
            pair.sort()
            items = _items_in_intersection(assignment, *pair)
            if items:
                ground_truth = ", ".join(items)
                question = (
                    f"Which items are in BOTH {names[pair[0]]} AND {names[pair[1]]}? "
                    f"List them separated by commas in curly brackets, e.g., {{Apple, Banana}}. "
                    f"If none, answer {{None}}."
                )
            else:
                ground_truth = "None"
                question = (
                    f"Which items are in BOTH {names[pair[0]]} AND {names[pair[1]]}? "
                    f"List them separated by commas in curly brackets, e.g., {{Apple, Banana}}. "
                    f"If none, answer {{None}}."
                )
        else:
            # fallback
            ground_truth = "None"
            question = "Which items are shared between sets? Answer {None} if none."

    elif question_type == "exclusive":
        target = rng.randint(0, n_circles - 1)
        items = _items_exclusive_to(assignment, target)
        if items:
            ground_truth = ", ".join(items)
        else:
            ground_truth = "None"
        question = (
            f"Which items are ONLY in {names[target]} and not in any other set? "
            f"List them separated by commas in curly brackets, e.g., {{Apple, Banana}}. "
            f"If none, answer {{None}}."
        )

    elif question_type == "union_minus":
        if n_circles >= 2:
            # "Items in A but NOT in B"
            a, b = rng.sample(range(n_circles), 2)
            items_a = set(_items_in_set(assignment, a))
            items_b = set(_items_in_set(assignment, b))
            diff = sorted(items_a - items_b)
            if diff:
                ground_truth = ", ".join(diff)
            else:
                ground_truth = "None"
            question = (
                f"Which items are in {names[a]} but NOT in {names[b]}? "
                f"List them separated by commas in curly brackets, e.g., {{Apple, Banana}}. "
                f"If none, answer {{None}}."
            )
        else:
            ground_truth = "None"
            question = "Which items are in A but not in B? Answer {None} if none."

    else:
        raise ValueError(f"Unknown question_type: {question_type}")

    return question, ground_truth


def render(
    n_circles: int = 2,
    question_type: str = "intersection",
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    n_circles = min(n_circles, 4)
    centers = _LAYOUTS[n_circles]
    radius = _RADII[n_circles]
    names = _SET_NAMES[:n_circles]

    assignment = _assign_items_to_regions(n_circles, rng)
    prompt, ground_truth = _generate_question(n_circles, assignment, question_type, rng)

    # Render the Venn diagram
    dpi = 100
    fig_size = resolution / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Draw circles with alpha fill
    for i in range(n_circles):
        cx, cy = centers[i]
        circle = plt.Circle(
            (cx, cy), radius,
            facecolor=_SET_COLORS_ALPHA[i],
            edgecolor=_SET_COLORS[i],
            linewidth=2,
        )
        ax.add_patch(circle)
        # Label the set outside the circle
        label_offset_y = radius + 0.04
        ax.text(
            cx, cy + label_offset_y, names[i],
            ha="center", va="bottom", fontsize=14, fontweight="bold",
            color=_SET_COLORS[i],
        )

    # Place item labels in their regions
    font_size = {2: 9, 3: 8, 4: 7}[n_circles]
    line_spacing = {2: 0.028, 3: 0.025, 4: 0.022}[n_circles]

    for region, items in assignment.items():
        # Compute centroid of the region: average of circle centers in the region
        region_centers = [centers[i] for i in region]
        cx = sum(c[0] for c in region_centers) / len(region_centers)
        cy = sum(c[1] for c in region_centers) / len(region_centers)

        # For exclusive regions, push text toward the far edge of the circle
        if len(region) == 1:
            my_idx = list(region)[0]
            my_cx, my_cy = centers[my_idx]
            all_cx = sum(c[0] for c in centers) / len(centers)
            all_cy = sum(c[1] for c in centers) / len(centers)
            dx = my_cx - all_cx
            dy = my_cy - all_cy
            dist = (dx**2 + dy**2)**0.5 or 0.01
            push = radius * 0.5
            cx = my_cx + dx / dist * push
            cy = my_cy + dy / dist * push
        elif len(region) >= 2 and len(region) < n_circles:
            # For partial intersections, push slightly away from the global center
            all_cx = sum(c[0] for c in centers) / len(centers)
            all_cy = sum(c[1] for c in centers) / len(centers)
            dx = cx - all_cx
            dy = cy - all_cy
            dist = (dx**2 + dy**2)**0.5 or 0.01
            push = radius * 0.15
            cx = cx + dx / dist * push
            cy = cy + dy / dist * push

        # Render items as stacked text
        for j, item in enumerate(items):
            y_offset = -line_spacing * (len(items) - 1) / 2 + line_spacing * j
            ax.text(
                cx, cy - y_offset, item,
                ha="center", va="center", fontsize=font_size,
                color="#1a1a1a",
            )

    ax.set_title("Venn Diagram", fontsize=12, pad=10)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    # Build flat item-to-region mapping for metadata
    item_regions = {}
    for region, items in assignment.items():
        region_label = " ∩ ".join(names[i] for i in sorted(region))
        for item in items:
            item_regions[item] = region_label

    metadata = {
        "prompt": prompt,
        "n_circles": n_circles,
        "question_type": question_type,
        "resolution": resolution,
        "set_names": names,
        "item_regions": item_regions,
    }
    return img, ground_truth, metadata
