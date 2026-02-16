"""Task: Text-only decision flowchart — perception vs reasoning diagnostic.

Provides the flowchart structure as a text adjacency list with conditions.
If text accuracy >> image accuracy, the failure is perceptual.
"""

from random import Random

from PIL import Image

from tasks.decision_flowchart import _CONDITIONS, _OUTCOMES, _PROCESS_LABELS

TASK_CONFIG = {
    "task_name": "decision_flowchart_text",
    "prompt_template": "",  # filled dynamically
    "prompt_template_v2": "",
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "template": "two_decision",
    },
    "sweep_axes": {
        "template": ["linear", "two_decision", "diamond_chain", "loop_with_exit"],
    },
}

_call_counter = 0


def render(
    template: str = "two_decision",
    seed: int | None = None,
    prompt_variant: int = 1,
) -> tuple[Image.Image, str, dict]:
    """Return a tiny placeholder image with flowchart structure as text."""
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    conds = rng.sample(_CONDITIONS, 2)
    outcomes = rng.sample(_OUTCOMES, 4)
    process = rng.choice(_PROCESS_LABELS)

    c1_answer = rng.choice(["Yes", "No"])
    c2_answer = rng.choice(["Yes", "No"])

    if template == "linear":
        flowchart_text = (
            f"Flowchart structure:\n"
            f"  1. Start\n"
            f"  2. {process}\n"
            f"  3. Decision: \"{conds[0][0]}\"?\n"
            f"     - If Yes → {outcomes[0]}\n"
            f"     - If No → {outcomes[1]}\n"
        )
        if c1_answer == "Yes":
            ground_truth = outcomes[0]
        else:
            ground_truth = outcomes[1]
        question = (
            f"If '{conds[0][0]}' is {c1_answer}, what is the outcome?"
        )

    elif template == "two_decision":
        flowchart_text = (
            f"Flowchart structure:\n"
            f"  1. Start\n"
            f"  2. Decision: \"{conds[0][0]}\"?\n"
            f"     - If No → {outcomes[0]} (end)\n"
            f"     - If Yes → go to step 3\n"
            f"  3. Decision: \"{conds[1][0]}\"?\n"
            f"     - If Yes → {outcomes[1]} (end)\n"
            f"     - If No → {outcomes[2]} (end)\n"
        )
        if c1_answer == "No":
            ground_truth = outcomes[0]
        elif c2_answer == "Yes":
            ground_truth = outcomes[1]
        else:
            ground_truth = outcomes[2]
        question = (
            f"If '{conds[0][0]}' is {c1_answer} and "
            f"'{conds[1][0]}' is {c2_answer}, what is the outcome?"
        )

    elif template == "diamond_chain":
        flowchart_text = (
            f"Flowchart structure:\n"
            f"  1. Start\n"
            f"  2. Decision: \"{conds[0][0]}\"?\n"
            f"     - If No → {outcomes[0]} (end)\n"
            f"     - If Yes → go to step 3\n"
            f"  3. {process}\n"
            f"  4. Decision: \"{conds[1][0]}\"?\n"
            f"     - If Yes → {outcomes[1]} (end)\n"
            f"     - If No → {outcomes[2]} (end)\n"
        )
        if c1_answer == "No":
            ground_truth = outcomes[0]
        elif c2_answer == "Yes":
            ground_truth = outcomes[1]
        else:
            ground_truth = outcomes[2]
        question = (
            f"If '{conds[0][0]}' is {c1_answer} and "
            f"'{conds[1][0]}' is {c2_answer}, what is the outcome?"
        )

    elif template == "loop_with_exit":
        flowchart_text = (
            f"Flowchart structure:\n"
            f"  1. Start\n"
            f"  2. {process}\n"
            f"  3. Decision: \"{conds[0][0]}\"?\n"
            f"     - If Yes → go back to step 2 (loop)\n"
            f"     - If No → {outcomes[0]} (end)\n"
        )
        ground_truth = outcomes[0]
        question = (
            f"When '{conds[0][0]}' becomes {c1_answer}, "
            f"what is the final outcome?"
        )

    else:
        raise ValueError(f"Unknown template: {template}")

    prompt = (
        f"{flowchart_text}\n"
        f"{question} Put your answer in curly brackets."
    )

    # Tiny placeholder image
    img = Image.new("RGB", (64, 64), "white")

    metadata = {
        "prompt": prompt,
        "template": template,
        "conditions": [(c[0], ans) for c, ans in zip(conds, [c1_answer, c2_answer])],
        "ground_truth_outcome": ground_truth,
        "mode": "text_only",
    }
    return img, ground_truth, metadata
