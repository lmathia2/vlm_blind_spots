"""Task: Text-only colored paths — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

_STATION_NAMES = ["Central", "North", "South", "East", "West", "Harbor", "Airport", "Park"]
_PATH_COLORS = ["Red", "Blue", "Green", "Orange", "Purple", "Yellow"]

TASK_CONFIG = {
    "task_name": "colored_paths_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "integer",
    "scorer": "integer_distance",
    "default_params": {
        "n_stations": 5,
        "n_paths": 3,
    },
    "sweep_axes": {
        "n_paths": [1, 2, 3, 4],
    },
}

_call_counter = 0


def render(
    n_stations: int = 5,
    n_paths: int = 3,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    stations = rng.sample(_STATION_NAMES, min(n_stations, len(_STATION_NAMES)))
    colors = rng.sample(_PATH_COLORS, min(n_paths, len(_PATH_COLORS)))

    # Pick query stations
    query_a, query_b = rng.sample(stations, 2)

    # Generate paths — some connect query stations, some don't
    paths_connecting = 0
    path_details = []
    for i in range(n_paths):
        color = colors[i]
        if i == 0:
            # Guarantee at least one connects query stations
            path_details.append({"color": color, "from": query_a, "to": query_b})
            paths_connecting += 1
        else:
            if rng.random() < 0.4:
                path_details.append({"color": color, "from": query_a, "to": query_b})
                paths_connecting += 1
            else:
                s1, s2 = rng.sample(stations, 2)
                path_details.append({"color": color, "from": s1, "to": s2})

    path_desc = "\n".join(
        f"  {p['color']} line: {p['from']} → {p['to']}" for p in path_details
    )

    prompt = (
        f"Stations: {', '.join(stations)}\n"
        f"Paths:\n{path_desc}\n\n"
        f"How many direct paths connect {query_a} to {query_b}? "
        f"Answer with a number in curly brackets, e.g., {{2}}."
    )

    ground_truth = str(paths_connecting)
    metadata = {
        "prompt": prompt,
        "n_stations": n_stations,
        "n_paths": n_paths,
        "paths_connecting": paths_connecting,
        "station_names": stations,
        "path_details": path_details,
        "mode": "text_only",
    }
    return placeholder_image(), ground_truth, metadata
