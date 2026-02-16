"""Task: Text-only stacked bar — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

TASK_CONFIG = {
    "task_name": "stacked_bar_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "mc4",
    "scorer": "exact_match",
    "default_params": {
        "n_bars": 4,
        "n_segments": 3,
    },
    "sweep_axes": {
        "n_bars": [3, 4, 5, 6],
        "n_segments": [2, 3],
    },
}

_call_counter = 0


def render(
    n_bars: int = 4,
    n_segments: int = 3,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    bar_names = [f"Bar {chr(65+i)}" for i in range(n_bars)]
    segment_names = ["Bottom", "Middle", "Top"][:n_segments]
    data = {}
    for bname in bar_names:
        data[bname] = {s: rng.randint(10, 50) for s in segment_names}

    target_bar = rng.choice(bar_names)
    target_segment = rng.choice(segment_names)
    target_value = data[target_bar][target_segment]

    distractors = set()
    while len(distractors) < 3:
        d = target_value + rng.choice([-12, -8, -4, 4, 8, 12])
        if d != target_value and 0 < d <= 60:
            distractors.add(d)

    options = [target_value] + sorted(distractors)
    rng.shuffle(options)
    correct_letter = chr(65 + options.index(target_value))
    options_text = "\n".join(f"  {chr(65+i)}) {v}" for i, v in enumerate(options))

    data_lines = []
    for bname in bar_names:
        segs = ", ".join(f"{s}={data[bname][s]}" for s in segment_names)
        data_lines.append(f"  {bname} segments: {segs}")
    data_text = "\n".join(data_lines)

    prompt = (
        f"Stacked bar chart data:\n{data_text}\n\n"
        f"What is the {target_segment} segment value for {target_bar}?\n"
        f"{options_text}\n\n"
        f"Answer with just the letter (A, B, C, or D)."
    )

    metadata = {
        "prompt": prompt,
        "n_bars": n_bars,
        "n_segments": n_segments,
        "data": data,
        "target_bar": target_bar,
        "target_segment": target_segment,
        "target_value": target_value,
        "correct_letter": correct_letter,
        "mode": "text_only",
    }
    return placeholder_image(), correct_letter, metadata
