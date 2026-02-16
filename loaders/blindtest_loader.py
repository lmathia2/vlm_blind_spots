"""Loaders for existing BlindTest images from the reference repository.

Scans image files, parses ground truth from filenames, produces manifest JSONL.
"""

import json
import re
import uuid
from collections import defaultdict
from pathlib import Path
from random import Random

from config import REFERENCE_DIR

# Fixed seed for reproducible sampling
RNG = Random(42)

# Reference prompts from the BlindTest paper
PROMPTS = {
    "line_intersection": (
        "Count the intersection points where the blue and red lines meet. "
        "Put your answer in curly brackets, e.g., {2}."
    ),
    "touching_circle": (
        "Are the two circles touching each other? Answer with Yes/No."
    ),
    "overlapping_circle": (
        "Are the two circles overlapping? Answer with Yes/No."
    ),
    "nested_squares": (
        "Count total number of squares in the image. "
        "Answer with only the number in numerical format in curly brackets e.g. {3}."
    ),
    "counting_grid": (
        "Count the number of rows and columns and answer with numbers in curly brackets. "
        "For example, rows={5} columns={6}"
    ),
    "counting_circles": (
        "Count the circles in the image. "
        "Answer with a number in curly brackets e.g. {3}."
    ),
}

# Preferred model directory to source images from
PREFERRED_MODELS = ["sonnet-3.5", "Sonnet-3.5", "sonnet3.5"]


def _find_model_dir(base_dir: Path) -> Path | None:
    """Find the best model directory to source images from."""
    for model in PREFERRED_MODELS:
        candidate = base_dir / model
        if candidate.exists():
            return candidate
    # Fallback: first available
    for d in sorted(base_dir.iterdir()):
        if d.is_dir():
            return d
    return None


def _collect_images(model_dir: Path) -> list[Path]:
    """Collect all PNG images from correct/ and incorrect/ subdirectories."""
    images = []
    for subdir in ["correct", "incorrect"]:
        d = model_dir / subdir
        if d.exists():
            images.extend(sorted(d.glob("*.png")))
    return images


def _balanced_sample(items: list, key_fn, max_per_class: int = 20) -> list:
    """Sample a balanced subset with at most max_per_class items per ground truth class."""
    by_class = defaultdict(list)
    for item in items:
        by_class[key_fn(item)].append(item)
    sampled = []
    for cls in sorted(by_class.keys()):
        pool = by_class[cls]
        RNG.shuffle(pool)
        sampled.extend(pool[:max_per_class])
    return sampled


def load_line_intersection(max_per_class: int = 20) -> list[dict]:
    """Load LineIntersection images. Ground truth from gt_{N} in filename."""
    base = REFERENCE_DIR / "src" / "LineIntersection" / "images" / "Count-prompt"
    model_dir = _find_model_dir(base)
    if not model_dir:
        print("  WARNING: No LineIntersection model dir found")
        return []

    images = _collect_images(model_dir)
    pattern = re.compile(r"gt_(\d+)_image_(\d+)_thickness_(\d+)_resolution_(\d+)")

    parsed = []
    for img in images:
        m = pattern.search(img.name)
        if m:
            gt = m.group(1)
            parsed.append({
                "image_path": str(img),
                "ground_truth": gt,
                "params": {
                    "gt_intersections": int(gt),
                    "image_id": int(m.group(2)),
                    "thickness": int(m.group(3)),
                    "resolution": int(m.group(4)),
                },
            })

    sampled = _balanced_sample(parsed, lambda x: x["ground_truth"], max_per_class)
    records = []
    for item in sampled:
        records.append({
            "sample_id": uuid.uuid4().hex[:8],
            "task_name": "line_intersection",
            "image_path": item["image_path"],
            "prompt": PROMPTS["line_intersection"],
            "ground_truth": item["ground_truth"],
            "parser": "integer",
            "scorer": "integer_distance",
            "params": item["params"],
            "source": "blindtest",
        })

    print(f"  LineIntersection: {len(records)} samples "
          f"(from {len(images)} images, {len(parsed)} parseable)")
    return records


def load_touching_circle(max_per_class: int = 20) -> list[dict]:
    """Load TouchingCircle images. Ground truth from distance in filename."""
    base = REFERENCE_DIR / "src" / "TouchingCircle" / "images" / "touching-prompt"
    model_dir = _find_model_dir(base)
    if not model_dir:
        print("  WARNING: No TouchingCircle model dir found")
        return []

    images = _collect_images(model_dir)
    pattern = re.compile(
        r"pixels_(\d+)_rotation_(\w+)_diameter_([\d.]+)_distance_(-?[\d.]+)\.png"
    )

    parsed = []
    for img in images:
        m = pattern.search(img.name)
        if m:
            distance = float(m.group(4))
            # For "touching" prompt: touching means distance == 0.0
            gt = "Yes" if distance == 0.0 else "No"
            parsed.append({
                "image_path": str(img),
                "ground_truth": gt,
                "params": {
                    "resolution": int(m.group(1)),
                    "rotation": m.group(2),
                    "diameter": float(m.group(3)),
                    "distance": distance,
                },
            })

    sampled = _balanced_sample(parsed, lambda x: x["ground_truth"], max_per_class)
    records = []
    for item in sampled:
        records.append({
            "sample_id": uuid.uuid4().hex[:8],
            "task_name": "touching_circle",
            "image_path": item["image_path"],
            "prompt": PROMPTS["touching_circle"],
            "ground_truth": item["ground_truth"],
            "parser": "yes_no",
            "scorer": "exact_match",
            "params": item["params"],
            "source": "blindtest",
        })

    print(f"  TouchingCircle: {len(records)} samples "
          f"(from {len(images)} images, {len(parsed)} parseable)")
    return records


def load_nested_squares(max_per_class: int = 20) -> list[dict]:
    """Load NestedSquares images. Ground truth from depth_{N} in filename."""
    base = REFERENCE_DIR / "src" / "NestedSquares" / "images" / "count-prompt"
    model_dir = _find_model_dir(base)
    if not model_dir:
        print("  WARNING: No NestedSquares model dir found")
        return []

    images = _collect_images(model_dir)
    pattern = re.compile(r"nested_squares_depth_(\d+)_image_(\d+)_thickness_(\d+)")

    parsed = []
    for img in images:
        m = pattern.search(img.name)
        if m:
            gt = m.group(1)
            parsed.append({
                "image_path": str(img),
                "ground_truth": gt,
                "params": {
                    "depth": int(gt),
                    "image_id": int(m.group(2)),
                    "thickness": int(m.group(3)),
                },
            })

    sampled = _balanced_sample(parsed, lambda x: x["ground_truth"], max_per_class)
    records = []
    for item in sampled:
        records.append({
            "sample_id": uuid.uuid4().hex[:8],
            "task_name": "nested_squares",
            "image_path": item["image_path"],
            "prompt": PROMPTS["nested_squares"],
            "ground_truth": item["ground_truth"],
            "parser": "integer",
            "scorer": "integer_distance",
            "params": item["params"],
            "source": "blindtest",
        })

    print(f"  NestedSquares: {len(records)} samples "
          f"(from {len(images)} images, {len(parsed)} parseable)")
    return records


def load_counting_grid(max_per_class: int = 20) -> list[dict]:
    """Load CountingRowsAndColumns images. Ground truth from grid_{R}x{C}."""
    base = REFERENCE_DIR / "src" / "CountingRowsAndColumns" / "images" / "CountRC-prompt"
    model_dir = _find_model_dir(base)
    if not model_dir:
        print("  WARNING: No CountingRowsAndColumns model dir found")
        return []

    images = _collect_images(model_dir)
    pattern = re.compile(r"grid_(\d+)x(\d+)_(\d+)_(\d+)")

    parsed = []
    for img in images:
        m = pattern.search(img.name)
        if m:
            rows, cols = m.group(1), m.group(2)
            gt = f"{rows},{cols}"
            parsed.append({
                "image_path": str(img),
                "ground_truth": gt,
                "params": {
                    "rows": int(rows),
                    "cols": int(cols),
                    "resolution": int(m.group(3)),
                    "line_width": int(m.group(4)),
                },
            })

    sampled = _balanced_sample(parsed, lambda x: x["ground_truth"], max_per_class)
    records = []
    for item in sampled:
        records.append({
            "sample_id": uuid.uuid4().hex[:8],
            "task_name": "counting_grid_blindtest",
            "image_path": item["image_path"],
            "prompt": PROMPTS["counting_grid"],
            "ground_truth": item["ground_truth"],
            "parser": "row_col",
            "scorer": "row_col",
            "params": item["params"],
            "source": "blindtest",
        })

    print(f"  CountingGrid: {len(records)} samples "
          f"(from {len(images)} images, {len(parsed)} parseable)")
    return records


def load_counting_circles(max_per_class: int = 20) -> list[dict]:
    """Load CountingCircles images. Ground truth from numCircles_{N}."""
    base = REFERENCE_DIR / "src" / "CountingCircles" / "images" / "circles" / "Count-prompt"
    model_dir = _find_model_dir(base)
    if not model_dir:
        print("  WARNING: No CountingCircles model dir found")
        return []

    images = _collect_images(model_dir)
    pattern = re.compile(r"numCircles_(\d+)")

    parsed = []
    for img in images:
        m = pattern.search(img.name)
        if m:
            gt = m.group(1)
            parsed.append({
                "image_path": str(img),
                "ground_truth": gt,
                "params": {"num_circles": int(gt)},
            })

    sampled = _balanced_sample(parsed, lambda x: x["ground_truth"], max_per_class)
    records = []
    for item in sampled:
        records.append({
            "sample_id": uuid.uuid4().hex[:8],
            "task_name": "counting_circles",
            "image_path": item["image_path"],
            "prompt": PROMPTS["counting_circles"],
            "ground_truth": item["ground_truth"],
            "parser": "integer",
            "scorer": "integer_distance",
            "params": item["params"],
            "source": "blindtest",
        })

    print(f"  CountingCircles: {len(records)} samples "
          f"(from {len(images)} images, {len(parsed)} parseable)")
    return records


# SubwayMap and CircledWord skipped: ground truth not extractable from filenames.
# SubwayMap model-result filenames have station pairs + UUIDs but no path count.
# CircledWord filenames are UUIDs with no letter info.
# Both are Priority 3 per project plan.


ALL_LOADERS = [
    load_line_intersection,
    load_touching_circle,
    load_nested_squares,
    load_counting_grid,
    load_counting_circles,
]


def load_all_blindtest(output_path: Path, max_per_class: int = 20) -> Path:
    """Load all BlindTest tasks and write a combined manifest JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    print(f"Loading BlindTest images (max {max_per_class} per class)...")
    for loader in ALL_LOADERS:
        records = loader(max_per_class=max_per_class)
        all_records.extend(records)

    with open(output_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    print(f"\nTotal: {len(all_records)} samples → {output_path}")
    return output_path
