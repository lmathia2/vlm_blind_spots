"""Task T4.4: Progress bar reading (MC4)."""

from random import Random

from PIL import Image, ImageDraw, ImageFont

from mc4_utils import format_mc4_prompt

TASK_CONFIG = {
    "task_name": "progress_bar",
    "prompt_template": None,  # dynamic MC4 prompt
    "prompt_template_v2": None,
    "parser": "mc4",
    "scorer": "exact_match",
    "default_params": {
        "n_bars": 3,
        "bar_height": 25,
        "resolution": 512,
    },
    "sweep_axes": {
        "n_bars": [2, 3, 4],
        "bar_height": [15, 25, 30],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

_BAR_NAMES = ["Upload", "Download", "Processing", "Analysis", "Sync",
              "Export", "Import", "Backup"]

_call_counter = 0


def _load_font(size: int):
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render(
    n_bars: int = 3,
    bar_height: int = 25,
    resolution: int = 512,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    names = rng.sample(_BAR_NAMES, n_bars)
    # Generate fill levels spaced ≥15% apart
    levels = []
    for _ in range(n_bars):
        for _attempt in range(100):
            v = rng.randint(10, 90)
            if all(abs(v - existing) >= 15 for existing in levels):
                levels.append(v)
                break
        else:
            levels.append(rng.randint(10, 90))

    target_idx = rng.randint(0, n_bars - 1)
    correct_pct = levels[target_idx]

    # Distractors from other bars' levels
    other_levels = [levels[i] for i in range(n_bars) if i != target_idx]
    distractors = []
    for v in other_levels:
        if abs(v - correct_pct) >= 15 and all(abs(v - d) >= 15 for d in distractors):
            distractors.append(v)

    for offset in [20, -20, 15, -15, 25, -25]:
        if len(distractors) >= 3:
            break
        v = correct_pct + offset
        if 5 <= v <= 95 and abs(v - correct_pct) >= 15 and all(abs(v - d) >= 15 for d in distractors):
            distractors.append(v)

    question = f"What percentage is the '{names[target_idx]}' progress bar at?"
    prompt, correct_letter = format_mc4_prompt(
        question, f"{correct_pct}%", [f"{d}%" for d in distractors[:3]],
        rng=Random(rng.randint(0, 2**31)),
    )

    # Render
    font = _load_font(max(10, resolution // 40))
    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)

    margin = resolution // 8
    bar_width = resolution - margin * 2 - 100
    spacing = min((resolution - margin * 2) // (n_bars + 1), bar_height * 4)
    y_start = (resolution - spacing * n_bars) // 2

    for i in range(n_bars):
        y = y_start + spacing * i
        # Label
        draw.text((margin, y - bar_height // 2 - 2), names[i], fill="black", font=font)
        # Bar background
        bar_x = margin + 100
        draw.rectangle([(bar_x, y), (bar_x + bar_width, y + bar_height)],
                       fill="#E0E0E0", outline="#999999")
        # Fill
        fill_w = int(bar_width * levels[i] / 100)
        draw.rectangle([(bar_x, y), (bar_x + fill_w, y + bar_height)],
                       fill="#4CAF50", outline="#999999")

    ground_truth = correct_letter
    metadata = {
        "prompt": prompt,
        "n_bars": n_bars,
        "bar_height": bar_height,
        "resolution": resolution,
        "names": names,
        "levels": levels,
        "target_name": names[target_idx],
        "target_pct": correct_pct,
        "correct_letter": correct_letter,
    }
    return img, ground_truth, metadata
