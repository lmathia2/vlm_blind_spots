"""Task: Text-only form field — perception vs reasoning diagnostic."""

from random import Random

from PIL import Image

from tasks._text_control import placeholder_image

_FIELD_DATA = [
    ("Full Name", ["John Smith", "Maria Garcia", "Wei Chen", "Aisha Khan"]),
    ("Email", ["john@example.com", "maria@corp.net", "wchen@mail.org", "ak@firm.io"]),
    ("Phone", ["555-0123", "555-0456", "555-0789", "555-0321"]),
    ("Address", ["123 Main St", "456 Oak Ave", "789 Pine Rd", "321 Elm Dr"]),
    ("City", ["Springfield", "Portland", "Austin", "Denver"]),
    ("Company", ["Acme Corp", "GlobalTech", "DataFlow Inc", "NetBridge"]),
    ("Position", ["Manager", "Analyst", "Director", "Engineer"]),
    ("Date", ["03/15/2024", "07/22/2024", "11/01/2024", "01/30/2025"]),
    ("Account #", ["AC-78432", "AC-91205", "AC-33891", "AC-55672"]),
    ("Reference", ["REF-2024-A", "REF-2024-B", "REF-2024-C", "REF-2024-D"]),
    ("Amount", ["$1,250.00", "$3,478.50", "$892.75", "$15,600.00"]),
    ("Status", ["Active", "Pending", "Approved", "Under Review"]),
]

TASK_CONFIG = {
    "task_name": "form_field_text",
    "prompt_template": "",
    "prompt_template_v2": "",
    "parser": "exact_string",
    "scorer": "exact_match",
    "default_params": {
        "n_fields": 6,
    },
    "sweep_axes": {
        "n_fields": [5, 8, 12],
    },
}

_call_counter = 0


def render(
    n_fields: int = 6,
    seed: int | None = None,
) -> tuple[Image.Image, str, dict]:
    global _call_counter
    _call_counter += 1
    rng = Random(seed if seed is not None else _call_counter)

    selected_fields = rng.sample(_FIELD_DATA, min(n_fields, len(_FIELD_DATA)))
    fields = [(label, rng.choice(values)) for label, values in selected_fields]
    target_idx = rng.randint(0, len(fields) - 1)
    target_label, target_value = fields[target_idx]

    field_lines = "\n".join(f"  {label}: {value}" for label, value in fields)

    prompt = (
        f"Form fields:\n{field_lines}\n\n"
        f"What is the value in the \"{target_label}\" field? "
        f"Put your answer in curly brackets, e.g., {{John Smith}}."
    )

    metadata = {
        "prompt": prompt,
        "n_fields": n_fields,
        "target_label": target_label,
        "fields": fields,
        "mode": "text_only",
    }
    return placeholder_image(), target_value, metadata
