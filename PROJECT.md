# VLM Blind Spots Evaluation — Implementation Plan

## Project Goal

Build a framework to evaluate Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) on low-level vision tasks, characterize perceptual blind spots, and connect them to document processing failures.

## Key Design Principles

1. **E2E first.** Get one task generating → evaluating → printing results before building anything else.
2. **Config-driven tasks.** Each task is a single .py file with a `TASK_CONFIG` dict and a `render()` function. No class hierarchies. Auto-discovered by registry.
3. **Decouple generate / evaluate / analyze.** Three separate CLI commands connected by JSONL manifests. Re-evaluate without regenerating. Resume crashed runs.
4. **Use existing BlindTest images for baseline.** The repo at `./reference/vision-llms-are-blind/` has 841MB of pre-generated images with ground truth in filenames. Don't regenerate for baseline — write loaders.
5. **Parallel API calls from the start.** Use ThreadPoolExecutor (10 workers) in the eval harness. ~500 calls serial = 17 min. Parallel = 2 min.

---

## Step 0: Sprint Zero (do this FIRST)

Before building any framework, write a single throwaway script `sprint_zero.py` that:
1. Generates a 5×6 grid image using PIL
2. Sends it to Haiku 4.5 with the prompt "Count the rows and columns"
3. Prints the response

This validates the API works, the image encoding is correct, and gives you first signal in minutes. Delete this file after extracting into the framework.

---

## Step 1: Repository Structure

```
vlm-blindspots/
├── README.md
├── requirements.txt           # anthropic, pillow, matplotlib, numpy, seaborn, pandas
├── config.py                  # MODEL, TEMPERATURE=0.0, MAX_WORKERS=10, paths
├── cli.py                     # Subcommands: generate, evaluate, analyze, baseline
├── harness.py                 # VisionClient + parallel eval + resume support
├── parsers.py                 # All parsers: integer, yes_no, letter, row_col, csv_letters
├── scorers.py                 # All scorers: exact_match, integer_distance, row_col, set_match
├── analysis.py                # Summary tables, accuracy curves, confusion matrices, failure examples
├── tasks/                     # One file per task, auto-discovered
│   ├── __init__.py            # Scans modules for TASK_CONFIG + render(), builds TASK_REGISTRY
│   └── *.py                   # Task files (see below)
├── loaders/
│   └── blindtest_loader.py    # Parse existing BlindTest image filenames → manifest JSONL
├── data/                      # Generated images + manifests
├── results/                   # Evaluation JSONL output
├── report_assets/             # Plots and failure images for the report
└── reference/
    └── vision-llms-are-blind/ # Clone of https://github.com/anguyen8/vision-llms-are-blind
```

---

## Step 2: Framework Core

### Task file contract

Every file in `tasks/` must export:

- `TASK_CONFIG`: dict with keys `task_name`, `prompt_template`, `parser` (name from PARSER_REGISTRY), `scorer` (name from SCORER_REGISTRY), `default_params`, and optionally `sweep_axes`
- `render(**params)`: function that returns `(PIL.Image, ground_truth_str, metadata_dict)`

The `tasks/__init__.py` auto-discovers all modules with both exports and builds `TASK_REGISTRY`.

### Parsers (parsers.py)

Implement and register these parsers:
- `integer`: extract int from `{N}` format or plain number
- `yes_no`: extract Yes/No from response
- `letter`: extract single letter
- `row_col`: extract "rows=N columns=M" or "(N,M)" into "N,M" string
- `csv_letters`: extract sorted comma-separated letters

### Scorers (scorers.py)

Implement and register these scorers:
- `exact_match`: case-insensitive string match → `{correct, score}`
- `integer_distance`: exact match + signed error (positive=overcount) → `{correct, score, error, abs_error}`
- `row_col`: score rows and columns independently → `{correct, row_correct, col_correct}`
- `set_match`: unordered set comparison with precision/recall → `{correct, precision, recall}`

### Harness (harness.py)

- `VisionClient`: wraps `anthropic.Anthropic().messages.create()` with base64 image encoding, temperature=0.0, auto-detects media type from file extension
- `evaluate_manifest(manifest_path, results_path, model, max_workers)`: loads manifest JSONL, skips sample_ids already in results file (resume support), evaluates remaining in parallel with ThreadPoolExecutor, writes results in append mode with `flush()` after each result

### CLI (cli.py)

Three subcommands:
- `generate --task <name> --n <count>` — runs render() N times at default_params, saves images + manifest JSONL to `data/<task>/`
- `generate --task <name> --sweep --n-per-config <count>` — runs render() across all sweep_axes combinations
- `evaluate --manifest <path> [--model <model>]` — evaluates manifest, writes to `results/`
- `analyze --results <path> [--plot]` — prints summary table, optionally generates plots
- `baseline` — convenience command that loads BlindTest images + generates new tasks, evaluates everything

### Manifest JSONL format (output of generate, input to evaluate)

```json
{"sample_id": "a1b2c3d4", "task_name": "counting_grid", "image_path": "data/counting_grid/img_001.png", "prompt": "Count the rows and columns...", "ground_truth": "5,6", "parser": "row_col", "scorer": "row_col", "params": {"rows": 5, "cols": 6, "resolution": 512}}
```

### Results JSONL format (output of evaluate, input to analyze)

Same as manifest plus: `raw_response`, `parsed_answer`, `correct`, `score`, `latency_s`, `model`, `input_tokens`, `output_tokens`, and any scorer-specific fields like `error`.

---

## Step 3: BlindTest Image Loaders

Write `loaders/blindtest_loader.py` to scan existing images and produce manifest JSONL files. The images are at `reference/vision-llms-are-blind/src/<TaskDir>/images/`.

### Filename patterns to parse

- **LineIntersection:** `gt_{0,1,2}_image_{N}_thickness_{2,4}_resolution_{384,768,1152}.png` — ground truth is the first number after `gt_`
- **TouchingCircle:** Images in subdirectories `touching-prompt/` and `overlapping-prompt/`. Filenames encode `pixels_{N}_rotation_{type}_diameter_{D}_distance_{D}`. Ground truth: touching=Yes for distance≤0, No for distance>0 (check the metadata.json in the directory too)
- **NestedSquares:** `nested_squares_depth_{N}_image_{I}_thickness_{T}.png` — ground truth is the depth
- **CountingRowsAndColumns:** `grid_{R}x{C}_{size}_{linewidth}.png` — ground truth is R,C
- **CountingCircles:** Check `metadata.json` or filename patterns for circle count
- **SubwayMap:** Has `metadata.json` with connection info

For each task, the loader should:
1. Glob for image files
2. Parse ground truth from filename (or load metadata.json)
3. Assign the correct prompt from the BlindTest prompts
4. Sample a balanced subset (e.g., 15–20 per ground-truth class) to keep API costs down
5. Write manifest JSONL

The loaders also need to set the correct `parser` and `scorer` fields for each task.

---

## Step 4: Task Implementations

### Priority 1 — Build these generators (needed for sweeps + baseline augmentation)

**counting_grid** — PIL ImageDraw. Draw black grid lines on white background. Params: `rows`, `cols`, `resolution`, `line_width`. Prompt asks for row and col count. Parser: `row_col`. Scorer: `row_col`. Sweep axes: rows [3,5,7,10,15], cols [3,5,7,10,15], line_width [1,2,3,5,10], resolution [256,512,768,1024].

**touching_circles** — Matplotlib. Two filled circles with parameterized distance. Params: `distance` (fraction of 2×radius; negative=overlap, 0=tangent, positive=gap), `resolution`, `rotation` (horizontal/vertical/diagonal), `diameter_ratio`. Prompt: "Are the two circles touching each other? Answer Yes/No." Parser: `yes_no`. Scorer: `exact_match`. Sweep axes: distance [-0.25 to 0.25 in steps of 0.05], resolution [384,768,1152].

**nested_squares** — Matplotlib. Recursively nested squares with random center offsets. Params: `depth` (2–5), `resolution`, `line_thickness`, `reduction_factor`. Prompt asks for total square count. Parser: `integer`. Scorer: `integer_distance`. Sweep: depth [2,3,4,5], line_thickness [1,2,3,5].

### Priority 2 — New document processing probes

**table_cell_read** — PIL. Draw a grid with 2-digit numbers in cells. Ask "What number is in row R, column C?" Params: `rows`, `cols`, `font_size`, `line_width`, `resolution`, `target_row`, `target_col`. Test the hard case: small font (10–12px), thin lines (1–2px), many cells (8×6+). Use DejaVuSans font (`/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`). Parser: `integer`. Scorer: `exact_match`. This directly tests spreadsheet/table reading.

**line_chart_crossing** — Matplotlib. Two line series ("Revenue" blue, "Cost" red) plotted on axes with labels, gridlines, and legend. Construct curves that cross exactly N times (0–3) by design (e.g., one ascending + one descending = 1 crossing; one sinusoidal + one flat = 2). Verify actual crossings via sign-change detection on the computed y-values. Prompt: "How many times do the blue and red lines cross?" Parser: `integer`. Scorer: `integer_distance`. This tests line intersection detection in a realistic chart context.

**arrow_following** — Matplotlib or PIL. Draw labeled boxes (A, B, C, D, E) at fixed positions, connected by arrows (lines with arrowheads). Generate a simple DAG. Ask: "Starting at box A, follow the arrows. What box do you reach?" or "Which boxes can be reached from A?" Params: `n_boxes`, `n_arrows`, `arrow_width`, `resolution`. Parser: `letter`. Scorer: `exact_match`. This is a simpler, more controllable version of the subway map that directly tests flowchart/diagram comprehension.

**form_checkboxes** — PIL. Draw a vertical list of labeled checkboxes (e.g., "☐ Option A", "☑ Option B", ...). Some checked, some unchecked. Draw checkmarks as simple lines inside small squares. Ask: "Which options are checked?" Params: `n_options`, `n_checked`, `box_size`, `font_size`, `resolution`. Parser: `csv_letters`. Scorer: `set_match`. Tests small shape discrimination + spatial association.

### Priority 3 — Use existing BlindTest images only (no generator needed)

**circled_letter** — Load from `reference/.../CircledWord/images/`. The freetype dependency makes regeneration annoying. Use existing images + metadata.

**counting_circles** — Load from `reference/.../CountingCircles/images/`. Olympic ring layout is fiddly to reimplement.

**subway_map** — Load from `reference/.../SubwayMap/images/`. The random walk graph generator is 150+ lines. Use existing images; the arrow_following probe covers the same primitive more cleanly.

---

## Step 5: Analysis (analysis.py)

Implement these functions:

- `print_summary(results_path)` — Load JSONL, group by task_name, print accuracy table with N, % correct, parse fail rate, and mean error where applicable
- `plot_accuracy_vs_param(results_path, param_name, group_by=None)` — For sweep results, plot accuracy as a function of one parameter, optionally with separate lines per group_by value. Save to report_assets/.
- `plot_confusion(results_path, task_name)` — For counting tasks, plot predicted vs ground truth heatmap. Reveals over/undercounting bias.
- `plot_accuracy_heatmap(results_path, x_param, y_param)` — 2D heatmap of accuracy across two sweep parameters (e.g., linewidth × resolution)
- `compute_bias(results_path, task_name)` — For counting tasks, compute mean signed error, overcount rate, undercount rate
- `save_failure_examples(results_path, output_dir, n=20)` — Copy the N worst failures (lowest score, most interesting params) to report_assets/ for the report

---

## Step 6: Perception vs. Reasoning Diagnostic

For the 1–2 worst blind spots, run a text-only control to confirm the failure is perceptual. 

For line intersection: provide the same line coordinates as text (e.g., "Blue line from (0.1, 0.8) through (2.5, 0.3) to (5.0, 0.7). Red line from ...") and ask the same counting question. If text accuracy >> image accuracy, the failure is perceptual. If text accuracy ≈ image accuracy, it's reasoning.

Implement this as a special task where `render()` returns a tiny placeholder image but the prompt contains all the geometric information as text.

---

## Execution Order

```
1. Sprint Zero: one-file script, validate API works           (~20 min)
2. Extract into framework: config, harness, parsers,          (~60 min)
   scorers, cli, tasks/__init__.py, tasks/counting_grid.py
   Verify: generate → evaluate → analyze pipeline works
3. BlindTest loaders: load existing images for                 (~40 min)
   touching_circles, nested_squares,
   counting_grid, circled_letter, counting_circles, subway_map
4. Run baseline evaluation on loaded BlindTest images          (~30 min)
   ~150 images across all tasks. Identify worst 2–3 tasks.
5. Build generators for worst 2 tasks (probably                (~60 min)
   touching_circles). Run parameter sweeps.
6. Build 2–3 doc processing probes: table_cell_read,           (~90 min)
   line_chart_crossing, arrow_following. Run evaluations.
7. Perception vs reasoning diagnostic on worst blind spot       (~20 min)
8. Analysis: produce all plots, save failure examples           (~40 min)
9. Buffer for iteration / deeper dives                          (~60 min)
```

Total: ~8 hours of focused implementation. First results by step 4 (~2.5 hours in).

---

## Critical Implementation Notes

- **Temperature must be 0.0** for all API calls. The BlindTest paper warns default temp=1 introduces high variance.
- **Model string:** `claude-haiku-4-5-20251001`. Make it configurable via env var `VLM_MODEL`.
- **Resume support is essential.** The evaluate command must skip sample_ids already in the results file and append new results. Use `flush()` after each write to survive crashes.
- **Balance ground truth classes.** For counting tasks, ensure equal samples per GT value (e.g., 15 images each for 0, 1, 2 intersections). Unbalanced data makes accuracy misleading.
- **Image resolution matters.** The Anthropic API rescales images. Test at the resolutions the BlindTest paper used (384, 768, 1152) to enable comparison.
- **Keep prompts simple and constrained.** Use the exact BlindTest prompts where applicable. For new tasks, ask for specific format (curly brackets, Yes/No, single letter) to make parsing reliable.