"""Task: Text-only radio button — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

_GROUP_NAMES = ["Shipping", "Payment", "Priority", "Category", "Status"]
_OPTION_POOLS = {
    "Shipping": ["Standard", "Express", "Overnight", "Economy"],
    "Payment": ["Credit Card", "PayPal", "Wire Transfer", "Check"],
    "Priority": ["Low", "Medium", "High", "Critical"],
    "Category": ["Personal", "Business", "Government", "Education"],
    "Status": ["Active", "Pending", "Closed", "On Hold"],
}

TASK_CONFIG = {
    "task_name": "radio_button_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "n_groups": 2,
        "options_per_group": 3,
    },
    "sweep_axes": {
        "n_groups": [1, 2, 3],
        "options_per_group": [3, 4],
    },
}

_call_counter = 0


def render(
    n_groups: int = 2,
    options_per_group: int = 3,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    group_names = rng.sample(_GROUP_NAMES, min(n_groups, len(_GROUP_NAMES)))
    target_group_idx = rng.randint(0, len(group_names) - 1)
    target_group = group_names[target_group_idx]

    group_descriptions = []
    selected_option = None
    for gi, gname in enumerate(group_names):
        pool = _OPTION_POOLS[gname]
        options = rng.sample(pool, min(options_per_group, len(pool)))
        sel_idx = rng.randint(0, len(options) - 1)
        if gi == target_group_idx:
            selected_option = options[sel_idx]
        option_lines = []
        for oi, opt in enumerate(options):
            marker = "(selected)" if oi == sel_idx else "(not selected)"
            option_lines.append(f"    {marker} {opt}")
        group_descriptions.append(f"  {gname}:\n" + "\n".join(option_lines))

    groups_text = "\n".join(group_descriptions)

    prompt = (
        f"Radio button groups:\n{groups_text}\n\n"
        f"What option is selected in the \"{target_group}\" group? "
        f"Put your answer in curly brackets, e.g., {{Express}}."
    )

    metadata = {
        "prompt": prompt,
        "n_groups": n_groups,
        "options_per_group": options_per_group,
        "target_group": target_group,
        "selected_option": selected_option,
        "mode": "text_only",
    }
    return placeholder_image(), selected_option, metadata
