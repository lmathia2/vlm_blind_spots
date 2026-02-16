# VLM Blind Spots — Execution Plan

## Phase 0: Sprint Zero
- [x] Write `sprint_zero.py` — generate a 5x6 grid, send to Haiku 4.5, print response
- [x] Validate API key, image encoding, and model access work
  - Verified by user: `rows=5 columns=6`, 392 tokens in, 10 out.

## Phase 1: Framework Core
- [x] Create `requirements.txt`
- [x] Create `config.py` — model, temperature, max_workers, paths
- [x] Create `parsers.py` — integer, yes_no, letter, row_col, csv_letters parsers + registry
- [x] Create `scorers.py` — exact_match, integer_distance, row_col, set_match scorers + registry
- [x] Create `harness.py` — VisionClient + parallel evaluate_manifest with resume
- [x] Create `tasks/__init__.py` — auto-discovery task registry
- [x] Create `tasks/counting_grid.py` — first task implementation
- [x] Create `cli.py` — generate, evaluate, analyze subcommands
- [x] Create `analysis.py` — summary tables, plots, confusion matrices
- [x] Verify E2E: `generate → evaluate → analyze` pipeline works for counting_grid
  - 5/5 correct on default 5x6 grid. Resume support verified (skips completed samples).

## Phase 2: BlindTest Loaders
- [x] Create `loaders/__init__.py`
- [x] Create `loaders/blindtest_loader.py` — scan reference images, parse filenames, produce manifests
- [x] Implement loaders for 5 tasks: LineIntersection (60), TouchingCircle (40), NestedSquares (80), CountingGrid (264), CountingCircles (100)
- [x] Verify: loaders produce valid manifests — 544 records, all images exist, classes balanced
- SKIPPED: SubwayMap, CircledWord (Priority 3, ground truth not extractable from filenames)

## Phase 3: Baseline Evaluation
- [x] Run baseline evaluation on 544 BlindTest images
- [x] Identify worst tasks:
  - line_intersection: **61.7%** (mean_error=+0.58, overcounting bias)
  - counting_grid_blindtest: **65.5%**
  - nested_squares: **67.5%** (mean_error=+0.28, overcounting bias)
  - counting_circles: **73.0%** (mean_error=-0.25, undercounting bias)
  - touching_circle: **75.0%**

## Phase 4: Priority 1 Task Generators + Sweeps
- [x] Implement `tasks/line_intersection.py` generator
- [x] Implement `tasks/touching_circles.py` generator
- [x] Implement `tasks/nested_squares.py` generator
- [x] Run parameter sweeps:
  - line_intersection sweep (105 samples): **46.7%** accuracy (mean_error=+0.30)
  - touching_circles sweep (33 samples): **90.9%** accuracy
  - nested_squares sweep (48 samples): **97.9%** accuracy (mean_error=+0.02)

## Phase 5: Document Processing Probes (Priority 2)
- [x] Implement `tasks/table_cell_read.py`
- [x] Implement `tasks/line_chart_crossing.py`
- [x] Implement `tasks/arrow_following.py`
- [x] Implement `tasks/form_checkboxes.py`
- [x] Run evaluations (20 samples each):
  - table_cell_read: **0%** — model reads out full grid but identifies wrong cell
  - arrow_following: **20%** — model struggles to follow directed arrows in DAGs
  - line_chart_crossing: **80%** (mean_error=+0.55)
  - form_checkboxes: **100%** — perfect checkbox identification

## Phase 6: Perception vs Reasoning Diagnostic
- [x] Create `tasks/line_intersection_text.py` — text-only control
- [x] Run and compare image vs text accuracy for line intersection:
  - Image-based (sweep): **46.7%**
  - Text-only (coordinates): **20.0%** (mean_error=+1.90)
  - **Finding**: Text-only is WORSE than image-based. The failure is primarily reasoning
    (computing segment intersections from coordinates), not purely perceptual.

## Phase 7: Analysis & Reporting
- [x] Implement `analysis.py` — summary tables, plots, confusion matrices
- [x] Generate accuracy_by_task plot
- [x] Generate combined analysis with all results (845 samples, 64.4% overall accuracy)
- [x] Save 20 failure examples to report_assets/failures/
- [x] Compute bias metrics:
  - line_intersection: overcount=38%, undercount=10%, exact=52%
  - line_intersection_text: overcount=60%, undercount=20%, exact=20%
  - nested_squares: overcount=18%, undercount=3%, exact=79%
  - counting_circles: overcount=7%, undercount=20%, exact=73%
  - line_chart_crossing: overcount=15%, undercount=5%, exact=80%
- [x] Generate 34 analysis plots in report_assets/

---

## Summary of All Results

| Task | Source | N | Accuracy | Key Finding |
|------|--------|---|----------|-------------|
| line_intersection | BlindTest | 60 | 61.7% | Overcounting bias (+0.58) |
| line_intersection | Generated sweep | 105 | 46.7% | Worst at generated images |
| line_intersection_text | Text-only control | 30 | 20.0% | Reasoning failure, not perceptual |
| counting_grid_blindtest | BlindTest | 264 | 65.5% | |
| nested_squares | BlindTest | 80 | 67.5% | Overcounting bias (+0.28) |
| nested_squares | Generated sweep | 48 | 97.9% | Much better on clean images |
| counting_circles | BlindTest | 100 | 73.0% | Undercounting bias (-0.25) |
| touching_circle | BlindTest | 40 | 75.0% | |
| touching_circles | Generated sweep | 33 | 90.9% | Better on clean images |
| table_cell_read | Generated | 20 | 0% | Complete failure on cell lookup |
| arrow_following | Generated | 20 | 20% | Cannot follow directed arrows |
| line_chart_crossing | Generated | 20 | 80% | Reasonable |
| form_checkboxes | Generated | 20 | 100% | Perfect |

## Files Created
- `config.py`, `parsers.py`, `scorers.py`, `harness.py`, `cli.py`, `analysis.py`
- `tasks/__init__.py`, `tasks/counting_grid.py`, `tasks/line_intersection.py`
- `tasks/touching_circles.py`, `tasks/nested_squares.py`
- `tasks/table_cell_read.py`, `tasks/line_chart_crossing.py`
- `tasks/arrow_following.py`, `tasks/form_checkboxes.py`
- `tasks/line_intersection_text.py`
- `loaders/__init__.py`, `loaders/blindtest_loader.py`
- `requirements.txt`, `sprint_zero.py`
