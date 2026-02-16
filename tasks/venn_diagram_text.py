"""Task: Text-only venn diagram — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

_ITEMS = [
    "Apple", "Banana", "Cherry", "Date", "Fig", "Grape", "Kiwi",
    "Lemon", "Mango", "Nectarine", "Orange", "Peach", "Plum",
]

TASK_CONFIG = {
    "task_name": "venn_diagram_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "csv_words",
    "scorer": "set_match",
    "default_params": {
        "n_circles": 2,
        "question_type": "intersection",
    },
    "sweep_axes": {
        "n_circles": [2, 3, 4],
        "question_type": ["intersection", "exclusive", "union_minus"],
    },
}

_call_counter = 0


def render(
    n_circles: int = 2,
    question_type: str = "intersection",
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    set_names = [chr(65 + i) for i in range(n_circles)]  # A, B, C, ...
    items = rng.sample(_ITEMS, min(10, len(_ITEMS)))

    # Assign items to sets with some overlap
    sets = {name: set() for name in set_names}
    for item in items:
        # Each item belongs to 1-2 sets
        n_memberships = rng.randint(1, min(2, n_circles))
        member_sets = rng.sample(set_names, n_memberships)
        for s in member_sets:
            sets[s].add(item)

    # Compute answer based on question type
    if question_type == "intersection":
        result = sets[set_names[0]]
        for name in set_names[1:]:
            result = result & sets[name]
        question = f"Which items are in ALL of sets {', '.join(set_names)}?"
    elif question_type == "exclusive":
        target_set = set_names[0]
        result = sets[target_set].copy()
        for name in set_names[1:]:
            result -= sets[name]
        question = f"Which items are ONLY in set {target_set} (not in any other set)?"
    elif question_type == "union_minus":
        target_set = set_names[0]
        union_rest = set()
        for name in set_names[1:]:
            union_rest |= sets[name]
        result = union_rest - sets[target_set]
        question = f"Which items are NOT in set {target_set} but ARE in at least one other set?"
    else:
        raise ValueError(f"Unknown question_type: {question_type}")

    set_desc = []
    for name in set_names:
        members = sorted(sets[name])
        set_desc.append(f"  Set {name}: {{{', '.join(members)}}}")
    sets_text = "\n".join(set_desc)

    ground_truth = ",".join(sorted(result, key=str.lower)) if result else ""

    prompt = (
        f"Sets:\n{sets_text}\n\n"
        f"{question}\n"
        f"List the items separated by commas in curly brackets, e.g., {{Apple, Banana}}. "
        f"If none, answer {{}}."
    )

    metadata = {
        "prompt": prompt,
        "n_circles": n_circles,
        "question_type": question_type,
        "set_names": set_names,
        "sets": {name: sorted(s) for name, s in sets.items()},
        "mode": "text_only",
    }
    return placeholder_image(), ground_truth, metadata
