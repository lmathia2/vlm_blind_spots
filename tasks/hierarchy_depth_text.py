"""Task: Text-only hierarchy depth — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

_ROLE_NAMES = [
    "CEO", "CTO", "CFO", "VP Engineering", "VP Sales", "VP Marketing",
    "Director A", "Director B", "Director C", "Director D",
    "Manager 1", "Manager 2", "Manager 3", "Manager 4", "Manager 5",
    "Lead 1", "Lead 2", "Lead 3", "Lead 4",
    "Staff 1", "Staff 2", "Staff 3", "Staff 4", "Staff 5",
]

TASK_CONFIG = {
    "task_name": "hierarchy_depth_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "integer",
    "scorer": "exact_match",
    "default_params": {
        "depth": 3,
        "branching": 2,
    },
    "sweep_axes": {
        "depth": [2, 3, 4, 5],
        "branching": [2, 3],
    },
}

_call_counter = 0


def _build_tree(depth, branching, rng, counter):
    """Recursively build a tree structure as nested text."""
    counter[0] += 1
    name = f"Node {counter[0]}"
    if depth <= 1:
        return {"name": name, "children": []}
    children = []
    for _ in range(branching):
        children.append(_build_tree(depth - 1, branching, rng, counter))
    return {"name": name, "children": children}


def _tree_to_text(tree, indent=0):
    """Convert tree dict to indented text representation."""
    lines = ["  " * indent + f"- {tree['name']}"]
    for child in tree["children"]:
        lines.extend(_tree_to_text(child, indent + 1))
    return lines


def render(
    depth: int = 3,
    branching: int = 2,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    counter = [0]

    tree = _build_tree(depth, branching, rng, counter)
    tree_text = "\n".join(_tree_to_text(tree))

    prompt = (
        f"Organization chart:\n{tree_text}\n\n"
        f"How many levels deep is this hierarchy? "
        f"Count the root as level 1. "
        f"Answer with a number in curly brackets, e.g., {{3}}."
    )

    ground_truth = str(depth)
    metadata = {
        "prompt": prompt,
        "depth": depth,
        "branching": branching,
        "mode": "text_only",
    }
    return placeholder_image(), ground_truth, metadata
