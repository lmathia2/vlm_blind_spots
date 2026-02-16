"""Task: Text-only arrow following — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image
from tasks.arrow_following import _generate_dag_edges, _all_terminals

TASK_CONFIG = {
    "task_name": "arrow_following_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "letter",
    "scorer": "set_member",
    "default_params": {
        "n_boxes": 5,
        "n_arrows": 5,
    },
    "sweep_axes": {
        "n_boxes": [4, 5, 6, 7, 8, 10],
        "n_arrows": [4, 6, 8, 10, 14],
    },
}

_call_counter = 0


def render(
    n_boxes: int = 5,
    n_arrows: int = 5,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    labels = [chr(65 + i) for i in range(n_boxes)]
    edges = _generate_dag_edges(n_boxes, n_arrows, rng)

    # Build adjacency list
    adj = [[] for _ in range(n_boxes)]
    for u, v in edges:
        adj[u].append(v)

    # Find valid start nodes (those with outgoing edges)
    candidates = [i for i in range(n_boxes) if adj[i]]
    if not candidates:
        candidates = list(range(n_boxes))
    start = rng.choice(candidates)
    start_label = labels[start]

    all_terminal_labels = sorted(labels[i] for i in _all_terminals(start, adj))
    ground_truth = ",".join(all_terminal_labels)

    edge_desc = ", ".join(f"{labels[u]}→{labels[v]}" for u, v in edges)

    prompt = (
        f"Boxes: {', '.join(labels)}\n"
        f"Arrows: {edge_desc}\n\n"
        f"Starting at box {start_label}, follow the arrows. "
        f"What box do you end at? If multiple paths exist, name all possible "
        f"final boxes. Answer with just the letter(s)."
    )

    metadata = {
        "prompt": prompt,
        "start_box": start_label,
        "valid_terminals": all_terminal_labels,
        "edges": [(labels[u], labels[v]) for u, v in edges],
        "n_boxes": n_boxes,
        "n_arrows": len(edges),
        "mode": "text_only",
    }
    return placeholder_image(), ground_truth, metadata
