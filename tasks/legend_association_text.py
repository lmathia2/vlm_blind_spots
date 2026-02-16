"""Task: Text-only legend association — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

TASK_CONFIG = {
    "task_name": "legend_association_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "n_series": 3,
    },
    "sweep_axes": {
        "n_series": [2, 3, 4],
    },
}

_SERIES_NAMES = ["Revenue", "Costs", "Profit", "Growth", "Expenses"]
_COLORS = ["blue", "red", "green", "orange"]

_call_counter = 0


def render(
    n_series: int = 3,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    names = rng.sample(_SERIES_NAMES, min(n_series, len(_SERIES_NAMES)))
    colors = _COLORS[:n_series]
    n_points = 10
    series_data = {}
    for name in names:
        series_data[name] = [rng.randint(10, 90) for _ in range(n_points)]

    # Find winner (highest peak)
    winner = max(names, key=lambda n: max(series_data[n]))

    data_desc = []
    for name, color in zip(names, colors):
        pts = ", ".join(str(v) for v in series_data[name])
        data_desc.append(f"  {name} ({color}): [{pts}]")
    data_text = "\n".join(data_desc)

    prompt = (
        f"Line chart series data:\n{data_text}\n\n"
        f"Which series has the highest peak value? "
        f"Put your answer in curly brackets, e.g., {{Revenue}}."
    )

    metadata = {
        "prompt": prompt,
        "n_series": n_series,
        "series_names": names,
        "winner": winner,
        "series_data": series_data,
        "mode": "text_only",
    }
    return placeholder_image(), winner, metadata
