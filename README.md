# VLM Blind Spots Evaluation

Framework for evaluating Claude Haiku 4.5 on low-level vision tasks, characterizing perceptual blind spots, and connecting them to document processing failures.

## Setup

```bash
conda activate agentic
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quick Start

```bash
# Generate 50 samples for a task
python3 cli.py generate --task arrow_following --n 50

# Evaluate against Claude Haiku 4.5
python3 cli.py evaluate --manifest data/arrow_following/manifest.jsonl

# View results
python3 cli.py analyze --results results/manifest/results.jsonl
```

## Commands

### Generate

Create images and a manifest JSONL for a task.

```bash
# Fixed number of samples at default parameters
python3 cli.py generate --task counting_grid --n 50

# Sweep all parameter combinations (e.g., resolution × depth × thickness)
python3 cli.py generate --task nested_squares --sweep --n-per-config 5

# Use rephrased prompt variant
python3 cli.py generate --task arrow_following --n 50 --prompt-variant 2
```

### Evaluate

Send images to the model and record responses.

```bash
# Standard evaluation (temperature=0)
python3 cli.py evaluate --manifest data/counting_grid/manifest.jsonl

# With extended thinking (reasoning mode)
python3 cli.py evaluate --manifest data/counting_grid/manifest.jsonl --reasoning

# Custom output path and parallelism
python3 cli.py evaluate --manifest data/combined/manifest.jsonl \
    --output results/my_run/results.jsonl --workers 10

# Override model
python3 cli.py evaluate --manifest data/combined/manifest.jsonl --model claude-sonnet-4-20250514
```

Evaluation supports automatic resume — if interrupted, re-running the same command skips already-completed samples.

### Analyze

Print summary tables and generate plots.

```bash
# Summary table with Wilson confidence intervals
python3 cli.py analyze --results results/combined/results.jsonl

# With plots (saved to report_assets/)
python3 cli.py analyze --results results/combined/results.jsonl --plot
```

If the results file contains both reasoning and non-reasoning results, a side-by-side comparison table is printed automatically.

### Baseline

Evaluate pre-existing BlindTest reference images.

```bash
python3 cli.py baseline
```

## Running the Full Evaluation

To reproduce the complete evaluation with both reasoning modes:

```bash
# 1. Generate samples for all tasks
for task in arrow_following counting_grid form_checkboxes line_chart_crossing \
           line_intersection line_intersection_text nested_squares \
           table_cell_read touching_circles; do
    python3 cli.py generate --task $task --sweep --n-per-config 5
done

# 2. Combine all manifests
cat data/*/manifest.jsonl > data/combined/manifest.jsonl

# 3. Evaluate without reasoning
python3 cli.py evaluate --manifest data/combined/manifest.jsonl \
    --output results/combined/results.jsonl --workers 10

# 4. Evaluate with reasoning (extended thinking)
python3 cli.py evaluate --manifest data/combined/manifest.jsonl --reasoning \
    --output results/combined_reasoning/results.jsonl --workers 10

# 5. Merge and analyze side-by-side
cat results/combined/results.jsonl results/combined_reasoning/results.jsonl \
    > results/combined_both/results.jsonl
python3 cli.py analyze --results results/combined_both/results.jsonl --plot
```

## Reasoning Mode

The `--reasoning` flag enables Claude's extended thinking (chain-of-thought) during evaluation. This uses the `thinking` API parameter with a 4096-token budget.

Key differences when reasoning is enabled:
- Temperature cannot be set to 0 (API constraint); uses default temp=1
- Responses include a `thinking_text` field with the model's internal reasoning
- `max_tokens` is increased to accommodate thinking + response tokens
- Results are tagged with `reasoning_mode: true`

Traces show a summary of the thinking process:
```
── [1/50] 9684203a | arrow_following | CORRECT
   Thinking: Let me trace the arrows starting from box D...
   Response: {E}
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

## Results

### Accuracy by Task (1,226 samples, Claude Haiku 4.5)

| Task | N | No-Reasoning | Reasoning | Delta |
|------|---|-------------|-----------|-------|
| line_intersection_text | 60 | 20.0% | 90.0% | +70.0% |
| arrow_following | 50 | 34.0% | 46.0% | +12.0% |
| nested_squares | 315 | 51.7% | 55.9% | +4.1% |
| line_intersection | 175 | 55.4% | 53.7% | -1.7% |
| touching_circles | 396 | 74.5% | 78.5% | +4.0% |
| line_chart_crossing | 80 | 81.2% | 96.2% | +15.0% |
| counting_grid | 50 | 100.0% | 90.0% | -10.0% |
| form_checkboxes | 50 | 100.0% | 100.0% | +0.0% |
| table_cell_read | 50 | 100.0% | 100.0% | +0.0% |
| **TOTAL** | **1226** | **65.2%** | **71.8%** | **+6.6%** |

Key findings:
- Reasoning gives **+6.6% overall improvement** (65.2% → 71.8%)
- Biggest gain: **line_intersection_text +70%** — reasoning dramatically helps with computing intersections from coordinates
- Reasoning hurts on **counting_grid (-10%)** — likely overthinking a simple task
- **line_intersection** is unchanged — visual perception is the bottleneck, not reasoning

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
- Temperature is fixed at 0.0 for reproducibility (except in reasoning mode, where the API requires temp=1).
- Evaluation runs 10 parallel workers by default (override with `VLM_MAX_WORKERS`).
