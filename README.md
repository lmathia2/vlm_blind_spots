# VLM Blind Spots Evaluation

Framework for evaluating Claude Haiku 4.5 on low-level vision tasks, characterizing perceptual blind spots, and connecting them to document processing failures.

## Setup

```bash
conda activate agentic
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Commands

```bash
PYTHON=/opt/homebrew/Caskroom/miniconda/base/envs/agentic/bin/python3

# Generate images + manifest for a task
$PYTHON cli.py generate --task counting_grid --n 20
$PYTHON cli.py generate --task line_intersection --sweep --n-per-config 3

# Evaluate a manifest against the model
$PYTHON cli.py evaluate --manifest data/counting_grid/manifest.jsonl

# Analyze results
$PYTHON cli.py analyze --results results/manifest/results.jsonl
$PYTHON cli.py analyze --results results/manifest/results.jsonl --plot

# Run baseline on BlindTest reference images
$PYTHON cli.py baseline
```

## Available Tasks

| Task | Parser | Scorer | Description |
|------|--------|--------|-------------|
| `counting_grid` | row_col | row_col | Count rows and columns in a grid |
| `line_intersection` | integer | integer_distance | Count intersection points of two lines |
| `line_intersection_text` | integer | integer_distance | Text-only control for line intersection |
| `touching_circles` | yes_no | exact_match | Are two circles touching? |
| `nested_squares` | integer | integer_distance | Count nested squares |
| `table_cell_read` | integer | exact_match | Read a number from a specific table cell |
| `line_chart_crossing` | integer | integer_distance | Count line crossings in a chart |
| `arrow_following` | letter | exact_match | Follow arrows in a DAG to find terminal box |
| `form_checkboxes` | csv_letters | set_match | Identify checked checkboxes |

## Adding a New Task

Create `tasks/your_task.py` with two exports:

```python
from PIL import Image

TASK_CONFIG = {
    "task_name": "your_task",
    "prompt_template": "Your prompt here. Answer in curly brackets, e.g., {3}.",
    "parser": "integer",           # one of: integer, yes_no, letter, row_col, csv_letters
    "scorer": "integer_distance",  # one of: exact_match, integer_distance, row_col, set_match
    "default_params": {"resolution": 512, "difficulty": 3},
    "sweep_axes": {
        "resolution": [384, 512, 768],
        "difficulty": [1, 2, 3, 4, 5],
    },
}

def render(resolution: int = 512, difficulty: int = 3) -> tuple[Image.Image, str, dict]:
    """Generate one sample. Returns (image, ground_truth_string, metadata_dict)."""
    img = Image.new("RGB", (resolution, resolution), "white")
    # ... draw your stimulus ...
    ground_truth = "42"
    metadata = {"resolution": resolution, "difficulty": difficulty}
    return img, ground_truth, metadata
```

The task is auto-discovered — no registration needed. For dynamic prompts, include a `"prompt"` key in the metadata dict and it will override `prompt_template`.

## Project Structure

```
├── cli.py              # CLI: generate, evaluate, analyze, baseline
├── config.py           # Model, paths, parallelism settings
├── harness.py          # VisionClient + parallel eval with resume
├── parsers.py          # Response parsers (integer, yes_no, letter, row_col, csv_letters)
├── scorers.py          # Scoring functions (exact_match, integer_distance, row_col, set_match)
├── analysis.py         # Summary tables, plots, confusion matrices
├── tasks/              # One file per task, auto-discovered
├── loaders/            # BlindTest image loaders
├── data/               # Generated images + manifests (JSONL)
├── results/            # Evaluation output (JSONL)
├── report_assets/      # Plots and failure examples
└── reference/          # Clone of vision-llms-are-blind repo
```

## Pipeline

```
generate (or load) → manifest.jsonl → evaluate → results.jsonl → analyze
```

- **Manifests** and **results** are JSONL files. Re-evaluate without regenerating. Resume crashed runs automatically (skips completed sample IDs).
- Model defaults to `claude-haiku-4-5-20251001` (override with `VLM_MODEL` env var).
- Temperature is fixed at 0.0 for reproducibility.
- Evaluation runs 10 parallel workers by default (override with `VLM_MAX_WORKERS`).
