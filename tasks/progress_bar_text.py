"""Task: Text-only progress bar — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

TASK_CONFIG = {
    "task_name": "progress_bar_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "mc4",
    "scorer": "exact_match",
    "default_params": {
        "n_bars": 3,
    },
    "sweep_axes": {
        "n_bars": [2, 3, 4],
    },
}

_BAR_NAMES = ["CPU", "Memory", "Disk", "Network", "GPU", "Battery"]

_call_counter = 0


def render(
    n_bars: int = 3,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    names = rng.sample(_BAR_NAMES, min(n_bars, len(_BAR_NAMES)))
    levels = [rng.randint(5, 95) for _ in range(n_bars)]
    target_idx = rng.randint(0, n_bars - 1)
    target_name = names[target_idx]
    target_pct = levels[target_idx]

    distractors = set()
    while len(distractors) < 3:
        d = target_pct + rng.choice([-15, -10, -5, 5, 10, 15])
        if d != target_pct and 1 <= d <= 100:
            distractors.add(d)

    options = [target_pct] + sorted(distractors)
    rng.shuffle(options)
    correct_letter = chr(65 + options.index(target_pct))
    options_text = "\n".join(f"  {chr(65+i)}) {v}%" for i, v in enumerate(options))

    bars_str = ", ".join(f"{names[i]}={levels[i]}%" for i in range(n_bars))

    prompt = (
        f"Progress bars: {bars_str}\n\n"
        f"What is the {target_name} level?\n"
        f"{options_text}\n\n"
        f"Answer with just the letter (A, B, C, or D)."
    )

    metadata = {
        "prompt": prompt,
        "n_bars": n_bars,
        "names": names,
        "levels": levels,
        "target_name": target_name,
        "target_pct": target_pct,
        "correct_letter": correct_letter,
        "mode": "text_only",
    }
    return placeholder_image(), correct_letter, metadata
