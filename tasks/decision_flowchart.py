"""Task T3.2: Decision flowchart traversal with fixed templates."""

from random import Random

from PIL import Image, ImageDraw, ImageFont

TASK_CONFIG = {
    "task_name": "decision_flowchart",
    "prompt_template": None,  # dynamic per sample
    "prompt_template_v2": None,
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "template": "two_decision",
        "resolution": 768,
    },
    "sweep_axes": {
        "template": ["linear", "two_decision", "diamond_chain", "loop_with_exit"],
        "resolution": [512, 768],
    },
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

# Condition pools
_CONDITIONS = [
    ("Amount > $500", "amount_high"),
    ("Manager Approved", "mgr_approved"),
    ("Budget Available", "budget_ok"),
    ("Priority = High", "priority_high"),
    ("Customer VIP", "customer_vip"),
    ("In Stock", "in_stock"),
    ("Credit Check Passed", "credit_ok"),
    ("Signature Required", "sig_required"),
]

_OUTCOMES = [
    "Approved", "Rejected", "Pending Review", "Escalated",
    "Auto-Processed", "Manual Review", "Completed", "On Hold",
]

_PROCESS_LABELS = [
    "Validate Input", "Check Records", "Process Request",
    "Send Notification", "Update Database", "Generate Report",
]

_call_counter = 0


def _load_font(size: int):
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _draw_box(draw, cx, cy, w, h, text, font, fill="white"):
    x0, y0 = cx - w // 2, cy - h // 2
    draw.rectangle([(x0, y0), (x0 + w, y0 + h)], fill=fill, outline="black", width=2)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th_ = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - tw // 2, cy - th_ // 2), text, fill="black", font=font)


def _draw_diamond(draw, cx, cy, size, text, font):
    pts = [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)]
    draw.polygon(pts, fill="#FFFFCC", outline="black", width=2)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th_ = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - tw // 2, cy - th_ // 2), text, fill="black", font=font)


def _draw_arrow(draw, x1, y1, x2, y2, label=None, font=None):
    draw.line([(x1, y1), (x2, y2)], fill="black", width=2)
    # Arrowhead
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    arrow_len = 10
    for da in [2.7, -2.7]:
        ax = x2 - arrow_len * math.cos(angle + da)
        ay = y2 - arrow_len * math.sin(angle + da)
        draw.line([(x2, y2), (int(ax), int(ay))], fill="black", width=2)
    if label and font:
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        draw.text((mx + 4, my - 12), label, fill="blue", font=font)


def render(
    template: str = "two_decision",
    resolution: int = 768,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    img = Image.new("RGB", (resolution, resolution), "white")
    draw = ImageDraw.Draw(img)
    font = _load_font(max(9, resolution // 60))
    small_font = _load_font(max(8, resolution // 75))

    conds = rng.sample(_CONDITIONS, 2)
    outcomes = rng.sample(_OUTCOMES, 4)
    process = rng.choice(_PROCESS_LABELS)

    # Randomize Yes/No answers for the question
    c1_answer = rng.choice(["Yes", "No"])
    c2_answer = rng.choice(["Yes", "No"])

    bw, bh = resolution // 5, resolution // 14  # box width, height
    ds = resolution // 12  # diamond half-size
    cx = resolution // 2

    if template == "linear":
        # Start → Process → Decision → End1/End2
        _draw_box(draw, cx, 60, bw, bh, "Start", font, fill="#D0F0D0")
        _draw_arrow(draw, cx, 60 + bh // 2, cx, 170 - ds)
        _draw_box(draw, cx, 170, bw, bh, process, font)
        _draw_arrow(draw, cx, 170 + bh // 2, cx, 320 - ds)
        _draw_diamond(draw, cx, 320, ds, conds[0][0], small_font)
        _draw_arrow(draw, cx - ds, 320, cx - 180, 470, "Yes", small_font)
        _draw_arrow(draw, cx + ds, 320, cx + 180, 470, "No", small_font)
        _draw_box(draw, cx - 180, 470, bw, bh, outcomes[0], font, fill="#FFD0D0")
        _draw_box(draw, cx + 180, 470, bw, bh, outcomes[1], font, fill="#FFD0D0")

        if c1_answer == "Yes":
            ground_truth = outcomes[0]
        else:
            ground_truth = outcomes[1]
        prompt = (
            f"In this flowchart, if '{conds[0][0]}' is {c1_answer}, what is the outcome? "
            f"Put your answer in curly brackets."
        )

    elif template == "two_decision":
        # Start → D1 → (Yes→D2, No→End1). D2 → (Yes→End2, No→End3)
        _draw_box(draw, cx, 50, bw, bh, "Start", font, fill="#D0F0D0")
        _draw_arrow(draw, cx, 50 + bh // 2, cx, 180 - ds)
        _draw_diamond(draw, cx, 180, ds, conds[0][0], small_font)
        _draw_arrow(draw, cx + ds, 180, cx + 200, 180, "No", small_font)
        _draw_box(draw, cx + 200, 180 - bh // 2, bw, bh, outcomes[0], font, fill="#FFD0D0")
        _draw_arrow(draw, cx, 180 + ds, cx, 350 - ds, "Yes", small_font)
        _draw_diamond(draw, cx, 350, ds, conds[1][0], small_font)
        _draw_arrow(draw, cx - ds, 350, cx - 200, 500, "Yes", small_font)
        _draw_arrow(draw, cx + ds, 350, cx + 200, 500, "No", small_font)
        _draw_box(draw, cx - 200, 500, bw, bh, outcomes[1], font, fill="#FFD0D0")
        _draw_box(draw, cx + 200, 500, bw, bh, outcomes[2], font, fill="#FFD0D0")

        if c1_answer == "No":
            ground_truth = outcomes[0]
        elif c2_answer == "Yes":
            ground_truth = outcomes[1]
        else:
            ground_truth = outcomes[2]
        prompt = (
            f"In this flowchart, if '{conds[0][0]}' is {c1_answer} and "
            f"'{conds[1][0]}' is {c2_answer}, what is the outcome? "
            f"Put your answer in curly brackets."
        )

    elif template == "diamond_chain":
        # Start → D1 → Process → D2 → End1/End2
        _draw_box(draw, cx, 40, bw, bh, "Start", font, fill="#D0F0D0")
        _draw_arrow(draw, cx, 40 + bh // 2, cx, 140 - ds)
        _draw_diamond(draw, cx, 140, ds, conds[0][0], small_font)
        _draw_arrow(draw, cx + ds, 140, cx + 200, 140, "No", small_font)
        _draw_box(draw, cx + 200, 140 - bh // 2, bw, bh, outcomes[0], font, fill="#FFD0D0")
        _draw_arrow(draw, cx, 140 + ds, cx, 280, "Yes", small_font)
        _draw_box(draw, cx, 280, bw, bh, process, font)
        _draw_arrow(draw, cx, 280 + bh // 2, cx, 420 - ds)
        _draw_diamond(draw, cx, 420, ds, conds[1][0], small_font)
        _draw_arrow(draw, cx - ds, 420, cx - 200, 550, "Yes", small_font)
        _draw_arrow(draw, cx + ds, 420, cx + 200, 550, "No", small_font)
        _draw_box(draw, cx - 200, 550, bw, bh, outcomes[1], font, fill="#FFD0D0")
        _draw_box(draw, cx + 200, 550, bw, bh, outcomes[2], font, fill="#FFD0D0")

        if c1_answer == "No":
            ground_truth = outcomes[0]
        elif c2_answer == "Yes":
            ground_truth = outcomes[1]
        else:
            ground_truth = outcomes[2]
        prompt = (
            f"In this flowchart, if '{conds[0][0]}' is {c1_answer} and "
            f"'{conds[1][0]}' is {c2_answer}, what is the outcome? "
            f"Put your answer in curly brackets."
        )

    elif template == "loop_with_exit":
        # Start → Process → Decision → (Yes→Process loop, No→End)
        _draw_box(draw, cx, 60, bw, bh, "Start", font, fill="#D0F0D0")
        _draw_arrow(draw, cx, 60 + bh // 2, cx, 200)
        _draw_box(draw, cx, 200, bw, bh, process, font)
        _draw_arrow(draw, cx, 200 + bh // 2, cx, 360 - ds)
        _draw_diamond(draw, cx, 360, ds, conds[0][0], small_font)
        # No → End
        _draw_arrow(draw, cx + ds, 360, cx + 220, 360, "No", small_font)
        _draw_box(draw, cx + 220, 360 - bh // 2, bw, bh, outcomes[0], font, fill="#FFD0D0")
        # Yes → loop back to Process
        _draw_arrow(draw, cx - ds, 360, cx - 150, 360, "Yes", small_font)
        draw.line([(cx - 150, 360), (cx - 150, 200)], fill="black", width=2)
        _draw_arrow(draw, cx - 150, 200, cx - bw // 2, 200)

        # The loop eventually exits on No
        ground_truth = outcomes[0]
        prompt = (
            f"In this flowchart, when '{conds[0][0]}' becomes {c1_answer}, "
            f"what is the final outcome? Put your answer in curly brackets."
        )
        # For loop: No→End, Yes→loops back. Final outcome is always the End box.
        # Adjust ground truth: the question says "when X becomes Y, what happens?"
        if c1_answer == "No":
            ground_truth = outcomes[0]
        else:
            # "Yes" means loop continues; outcome is still the same end box eventually
            ground_truth = outcomes[0]

    else:
        raise ValueError(f"Unknown template: {template}")

    metadata = {
        "prompt": prompt,
        "template": template,
        "resolution": resolution,
        "conditions": [(c[0], ans) for c, ans in zip(conds, [c1_answer, c2_answer])],
        "ground_truth_outcome": ground_truth,
    }
    return img, ground_truth, metadata
