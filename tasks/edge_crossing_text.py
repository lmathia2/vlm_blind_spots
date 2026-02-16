"""Task: Text-only edge crossing — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

TASK_CONFIG = {
    "task_name": "edge_crossing_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "yes_no",
    "scorer": "exact_match",
    "default_params": {
        "n_nodes": 5,
        "n_edges": 6,
    },
    "sweep_axes": {
        "n_nodes": [4, 5, 6],
        "n_edges": [4, 6, 8],
    },
}

_call_counter = 0


def render(
    n_nodes: int = 5,
    n_edges: int = 6,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    labels = [chr(65 + i) for i in range(n_nodes)]

    # Generate random edges
    possible_edges = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)]
    n_actual = min(n_edges, len(possible_edges))
    edge_indices = rng.sample(possible_edges, n_actual)
    edges = [(labels[i], labels[j]) for i, j in edge_indices]

    # Build adjacency for connectivity check
    adj = {l: set() for l in labels}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    # Pick query pair
    query_a, query_b = rng.sample(labels, 2)

    # BFS to check connectivity
    visited = set()
    queue = [query_a]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                queue.append(neighbor)

    connected = query_b in visited
    ground_truth = "Yes" if connected else "No"

    edge_desc = ", ".join(f"{a}-{b}" for a, b in edges)

    prompt = (
        f"Nodes: {', '.join(labels)}\n"
        f"Edges: {edge_desc}\n\n"
        f"Is node {query_a} connected to node {query_b} (directly or through other nodes)? "
        f"Answer Yes or No."
    )

    metadata = {
        "prompt": prompt,
        "n_nodes": n_nodes,
        "n_edges": n_actual,
        "edges": edges,
        "query": (query_a, query_b),
        "node_labels": labels,
        "mode": "text_only",
    }
    return placeholder_image(), ground_truth, metadata
