"""SFT training data generation for grid counting.

Generates (image, chain-of-thought, answer) triples using the existing
grid renderer and CoT templates. Outputs JSONL + images.

Seed ranges (non-overlapping):
  SFT:  [0, 100K) — direct [0, 20K), intermediate [20K, 40K), tool_use [40K, 50K)
  RL:   [100K, 500K)
  Eval: [500K, 510K)

Anti-shortcut randomization
----------------------------
After rendering, images are post-processed to prevent the model from
inferring grid size from image dimensions or pixel spacing:
- Random padding (0-30px per side, variable bg color)
- Slight aspect-ratio stretch (up to 15%)
- Background color variation (white, light gray, cream)
"""

import base64
import json
from io import BytesIO
from pathlib import Path
from random import Random

from PIL import ImageOps

from tasks.counting_grid import render as render_grid
from training.cot_templates import (
    DIRECT_COT_TEMPLATES,
    INTERMEDIATE_COT_TEMPLATES,
    TOOL_USE_COT_TEMPLATES,
    TOOL_USE_SKIP_TEMPLATES,
    fill_template,
)

# Seed offsets per strategy
_SEED_OFFSETS = {
    "direct": 0,
    "intermediate_repr": 20_000,
    "tool_use": 40_000,
}

# Grid size ranges per strategy
_GRID_RANGES = {
    "direct": (3, 12),
    "intermediate_repr": (3, 15),
    "tool_use": (12, 25),
    "tool_use_skip": (3, 8),
}

# Default sample counts
_DEFAULT_COUNTS = {
    "direct": 2000,
    "intermediate_repr": 2000,
    "tool_use": 1000,
}

# Background color palette for anti-shortcut randomization
_BG_COLORS = [
    (255, 255, 255),  # white
    (240, 240, 240),  # light gray
    (245, 245, 230),  # cream
    (230, 240, 250),  # light blue-gray
]

# Pre-computed valid (rows, cols) pairs per strategy for uniform sampling
_GRID_PAIRS_CACHE: dict[str, list[tuple[int, int]]] = {}


def _get_grid_pairs(strategy: str, is_skip: bool = False) -> list[tuple[int, int]]:
    """Return all valid (rows, cols) pairs for a strategy, cached."""
    key = f"{strategy}{'_skip' if is_skip else ''}"
    if key not in _GRID_PAIRS_CACHE:
        if is_skip:
            lo, hi = _GRID_RANGES["tool_use_skip"]
        else:
            lo, hi = _GRID_RANGES[strategy]
        _GRID_PAIRS_CACHE[key] = [
            (r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)
        ]
    return _GRID_PAIRS_CACHE[key]

# Prompt (same for all SFT samples — always grid_size question)
_PROMPT = (
    "Count the number of rows and columns in this grid. "
    "Reply in the format: rows=N columns=M"
)


def _sample_grid_params(rng: Random, strategy: str, is_skip: bool = False) -> dict:
    """Sample grid rendering parameters with uniform (rows, cols) distribution.

    Uses uniform sampling over all valid (rows, cols) pairs to ensure
    every grid size appears with equal probability. Also samples
    anti-shortcut randomization parameters (padding, bg_color, aspect stretch).
    """
    # Uniform sampling over all (rows, cols) pairs
    pairs = _get_grid_pairs(strategy, is_skip)
    rows, cols = rng.choice(pairs)

    # Resolution: larger for denser grids
    if max(rows, cols) > 15:
        resolution = rng.choice([512, 768, 1024])
    else:
        resolution = rng.choice([384, 512, 768])

    line_width = rng.choice([1, 2, 3])

    # Anti-shortcut randomization
    padding = tuple(rng.randint(0, 30) for _ in range(4))  # top, right, bottom, left
    bg_color = rng.choice(_BG_COLORS)
    # Aspect stretch: up to 15% in width or height
    aspect_stretch = (
        1.0 + rng.uniform(-0.15, 0.15),  # width multiplier
        1.0 + rng.uniform(-0.15, 0.15),  # height multiplier
    )

    return {
        "rows": rows,
        "cols": cols,
        "resolution": resolution,
        "line_width": line_width,
        "padding": padding,
        "bg_color": bg_color,
        "aspect_stretch": aspect_stretch,
    }


def _select_template(rng: Random, strategy: str, is_skip: bool = False) -> str:
    """Select a random template for the given strategy."""
    if strategy == "direct":
        return rng.choice(DIRECT_COT_TEMPLATES)
    elif strategy == "intermediate_repr":
        return rng.choice(INTERMEDIATE_COT_TEMPLATES)
    elif strategy == "tool_use":
        if is_skip:
            return rng.choice(TOOL_USE_SKIP_TEMPLATES)
        return rng.choice(TOOL_USE_COT_TEMPLATES)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _apply_anti_shortcut(img, params: dict):
    """Apply post-render randomization to prevent shortcut strategies.

    Adds padding with a random background color and applies a slight
    aspect-ratio stretch.  Keeps the renderer itself untouched.
    """
    padding = params["padding"]  # (top, right, bottom, left)
    bg_color = params["bg_color"]

    # Add padding
    if any(p > 0 for p in padding):
        # ImageOps.expand takes (left, top, right, bottom)
        border = (padding[3], padding[0], padding[1], padding[2])
        img = ImageOps.expand(img, border=border, fill=bg_color)

    # Apply aspect stretch
    w_mult, h_mult = params["aspect_stretch"]
    new_w = max(1, round(img.width * w_mult))
    new_h = max(1, round(img.height * h_mult))
    if (new_w, new_h) != (img.width, img.height):
        img = img.resize((new_w, new_h))

    return img


def _image_to_base64(img) -> str:
    """Convert PIL Image to base64 PNG string."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_one_sample(seed: int, strategy: str, rng: Random) -> dict:
    """Generate a single SFT training sample.

    Args:
        seed: Deterministic seed for the grid renderer.
        strategy: One of "direct", "intermediate_repr", "tool_use".
        rng: Random instance for template/param selection.

    Returns:
        Dict with keys: image, image_base64, prompt, chain_of_thought,
        answer, strategy, metadata, seed.
    """
    # For tool_use, first 200 seeds use small grids with skip templates
    is_skip = (
        strategy == "tool_use"
        and (seed - _SEED_OFFSETS["tool_use"]) < 200
    )

    params = _sample_grid_params(rng, strategy, is_skip=is_skip)
    rows, cols = params["rows"], params["cols"]

    # Render grid image (no merged cells for SFT)
    img, ground_truth, metadata = render_grid(
        rows=rows,
        cols=cols,
        resolution=params["resolution"],
        line_width=params["line_width"],
        n_merged=0,
        question_type="grid_size",
        seed=seed,
    )

    # Apply anti-shortcut post-processing
    img = _apply_anti_shortcut(img, params)

    # Select and fill template
    template = _select_template(rng, strategy, is_skip=is_skip)

    # 20% chance of self-correction for direct strategy
    include_self_correction = (
        strategy == "direct" and rng.random() < 0.2
    )

    cot = fill_template(
        template, rows, cols, rng,
        include_self_correction=include_self_correction,
    )

    return {
        "image": img,
        "image_base64": _image_to_base64(img),
        "prompt": _PROMPT,
        "chain_of_thought": cot,
        "answer": f"rows={rows} columns={cols}",
        "ground_truth": ground_truth,  # "R,C" format
        "strategy": strategy,
        "is_skip": is_skip,
        "seed": seed,
        "metadata": {
            **metadata,
            "strategy": strategy,
            "is_skip": is_skip,
            "include_self_correction": include_self_correction,
            "padding": params["padding"],
            "bg_color": params["bg_color"],
            "aspect_stretch": params["aspect_stretch"],
        },
    }


def generate_sft_dataset(
    output_dir: str | Path,
    strategy: str,
    n_samples: int | None = None,
    seed_offset: int | None = None,
) -> Path:
    """Generate N SFT samples for a strategy, saving images + JSONL.

    Args:
        output_dir: Root output directory.
        strategy: One of "direct", "intermediate_repr", "tool_use".
        n_samples: Number of samples (default per strategy).
        seed_offset: Starting seed (default per strategy).

    Returns:
        Path to the generated JSONL file.
    """
    if n_samples is None:
        n_samples = _DEFAULT_COUNTS[strategy]
    if seed_offset is None:
        seed_offset = _SEED_OFFSETS[strategy]

    output_dir = Path(output_dir)
    strategy_dir = output_dir / strategy
    img_dir = strategy_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = strategy_dir / "samples.jsonl"

    with open(jsonl_path, "w") as f:
        for i in range(n_samples):
            seed = seed_offset + i
            rng = Random(seed)
            sample = generate_one_sample(seed, strategy, rng)

            # Save image
            img_filename = f"{seed:06d}.png"
            img_path = img_dir / img_filename
            sample["image"].save(img_path)

            # Write JSONL record (without PIL Image or base64)
            record = {
                "seed": sample["seed"],
                "strategy": sample["strategy"],
                "is_skip": sample["is_skip"],
                "image_path": str(img_path),
                "prompt": sample["prompt"],
                "chain_of_thought": sample["chain_of_thought"],
                "answer": sample["answer"],
                "ground_truth": sample["ground_truth"],
                "metadata": sample["metadata"],
            }
            f.write(json.dumps(record) + "\n")

            if (i + 1) % 500 == 0:
                print(f"  [{strategy}] {i + 1}/{n_samples} samples generated")

    print(f"  [{strategy}] Done: {n_samples} samples → {jsonl_path}")
    return jsonl_path


def generate_all(output_dir: str | Path) -> list[Path]:
    """Generate all SFT datasets (direct + intermediate_repr + tool_use).

    Returns list of JSONL paths.
    """
    output_dir = Path(output_dir)
    paths = []
    for strategy in ["direct", "intermediate_repr", "tool_use"]:
        path = generate_sft_dataset(output_dir, strategy)
        paths.append(path)
    print(f"\nAll done. {len(paths)} datasets in {output_dir}")
    return paths
