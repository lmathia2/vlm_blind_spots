"""Task: Text-only dense text — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image
from tasks.dense_text import _TEXT_LINES

TASK_CONFIG = {
    "task_name": "dense_text_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "n_lines": 7,
    },
    "sweep_axes": {
        "n_lines": [5, 7, 10],
    },
}

_call_counter = 0


def render(
    n_lines: int = 7,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    lines = rng.sample(_TEXT_LINES, min(n_lines, len(_TEXT_LINES)))
    target_line_num = rng.randint(1, len(lines))
    target_text = lines[target_line_num - 1]

    ordinal = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
               6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}
    line_ref = ordinal.get(target_line_num, f"{target_line_num}th")

    numbered_lines = "\n".join(f"  {i+1}: {line}" for i, line in enumerate(lines))

    prompt = (
        f"The document contains these lines:\n{numbered_lines}\n\n"
        f"What does the {line_ref} line say? "
        f"Copy it exactly. Put your answer in curly brackets."
    )

    metadata = {
        "prompt": prompt,
        "n_lines": len(lines),
        "target_line_num": target_line_num,
        "lines": lines,
        "mode": "text_only",
    }
    return placeholder_image(), target_text, metadata
