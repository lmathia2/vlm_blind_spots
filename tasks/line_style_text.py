"""Task: Text-only line style — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

_COLORS = ["red", "blue", "green", "orange", "purple"]
_STYLES = ["solid", "dashed", "dotted", "dash-dot"]

TASK_CONFIG = {
    "task_name": "line_style_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "n_lines": 3,
    },
    "sweep_axes": {
        "n_lines": [2, 3, 4],
    },
}

_call_counter = 0


def render(
    n_lines: int = 3,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    colors = _COLORS[:n_lines]
    styles = rng.sample(_STYLES, min(n_lines, len(_STYLES)))
    target_idx = rng.randint(0, n_lines - 1)
    target_color = colors[target_idx]
    target_style = styles[target_idx]

    line_desc = ", ".join(f"{colors[i]}={styles[i]}" for i in range(n_lines))

    prompt = (
        f"Lines plotted with styles: {line_desc}\n\n"
        f"What line style is the {target_color} line? "
        f"Put your answer in curly brackets, e.g., {{solid}}."
    )

    metadata = {
        "prompt": prompt,
        "n_lines": n_lines,
        "colors": colors,
        "styles": styles,
        "target_color": target_color,
        "target_style": target_style,
        "mode": "text_only",
    }
    return placeholder_image(), target_style, metadata
