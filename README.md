# VLM Blind Spots

Diagnostic framework for evaluating vision-language model failures. Tests whether errors come from **perception** (can't see) or **reasoning** (can't think) by comparing image tasks against matched text-only controls.

## How It Works

For each visual task, a paired **text-only control** presents the same data as plain text. If accuracy jumps when images are replaced with text, the failure is perceptual. If accuracy stays low in both, it's a reasoning limitation.

```
generate → manifest.jsonl → evaluate → results.jsonl → analyze
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# Generate samples for all tasks with parameter sweeps
python cli.py generate --task all --sweep --min-samples 50 --max-total 75

# Evaluate (reasoning enabled by default)
python cli.py evaluate --manifest data/combined/manifest.jsonl --workers 10

# Analyze with perception vs reasoning diagnostic
python cli.py analyze --results results/combined/results.jsonl --diagnostic --plot
```

Evaluation supports **automatic resume** — re-running skips completed samples.

### CLI Flags

| Command | Flag | Description |
|---------|------|-------------|
| `evaluate` | `--no-reasoning` | Disable extended thinking (on by default) |
| `evaluate` | `--model MODEL` | Override model (default: `claude-haiku-4-5-20251001`) |
| `evaluate` | `--output PATH` | Custom results path |
| `evaluate` | `--workers N` | Parallel workers (default: 10) |
| `analyze` | `--diagnostic` | Print perception vs reasoning diagnostic table |
| `analyze` | `--plot` | Generate accuracy plots |

## Project Structure

```
├── cli.py                  # CLI entry point: generate, evaluate, analyze
├── config.py               # Model, paths, parallelism settings
├── harness.py              # VisionClient + parallel eval with resume
├── parsers.py              # Response parsers (integer, mc4, csv_words, etc.)
├── scorers.py              # Scoring functions (exact_match, set_match, etc.)
├── analysis.py             # Diagnostics, classification taxonomy, plots
├── tasks/                  # 34 image tasks + 34 text controls (auto-discovered)
│   ├── _text_control.py    # Shared utilities for text-only controls
│   ├── counting_grid.py    # Example image task
│   ├── counting_grid_text.py  # Matched text-only control
│   └── ...
├── report/
│   └── generate_figures.py # Figure generation for reports
├── report_haiku45/         # Haiku 4.5 evaluation report + figures
├── data/                   # Generated images + manifests (gitignored)
└── results/                # Evaluation results JSONL (gitignored)
```

## Tasks

34 synthetic image tasks across 7 categories, each with a matched text-only control (`*_text.py`):

| Category | Tasks |
|----------|-------|
| Text reading | `dense_text`, `rotated_text`, `text_degradation` |
| Annotation detection | `arrow_annotation`, `circled_text`, `highlighted_text`, `strikethrough` |
| Form/UI elements | `form_checkboxes`, `form_field`, `radio_button` |
| Table lookup | `color_coded_cells`, `merged_cell_read`, `realistic_table`, `table_cell_read` |
| Chart reading | `bar_chart_value`, `grouped_bar`, `heatmap`, `line_chart_point`, `pie_chart`, `progress_bar`, `scatter_plot`, `stacked_bar` |
| Chart association | `legend_association`, `line_chart_crossing`, `line_style` |
| Spatial/graph | `arrow_following`, `colored_paths`, `counting_grid`, `decision_flowchart`, `edge_crossing`, `hierarchy_depth`, `nested_squares`, `touching_circles`, `venn_diagram` |

Tasks are auto-discovered from `tasks/`. Each exports a `TASK_CONFIG` dict and a `render()` function.

## Adding a Task

Create `tasks/your_task.py`:

```python
TASK_CONFIG = {
    "task_name": "your_task",
    "prompt_template": "What value is in cell B2? Answer in {brackets}.",
    "parser": "integer",
    "scorer": "exact_match",
}

def render(**params) -> tuple[Image.Image, str, dict]:
    """Returns (image, ground_truth, metadata)."""
    ...
```

For a text-only control, create `tasks/your_task_text.py` that calls the parent's `render()`, discards the image, and returns a text description with a placeholder image.

## Key Results (Haiku 4.5)

- **84% mean image accuracy** vs **96% on text controls** (12-point perceptual gap)
- **9/34 tasks** have perceptual blind spots (counting, degraded text, proportions)
- **1/34 tasks** has a reasoning bottleneck (arrow following)
- **24/34 tasks** work well at 95–100%

See `report_haiku45/blind_spots_report.md` for the full analysis.
