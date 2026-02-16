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

## Phase 8: Rigor Improvements
- [x] Fix `letter` parser — search from end, skip "I"/"A", match "answer is X" patterns
- [x] Fix `csv_letters` parser — try comma-separated list and `{A,C,E}` before broad regex
- [x] Fix prompt format constraints — add curly bracket format hints to table_cell_read, arrow_following, form_checkboxes
- [x] Add Wilson binomial confidence intervals to `analysis.py` (summary table + bar plot error bars)
- [x] Add `--prompt-variant` CLI flag and `prompt_template_v2` to all 9 task configs
- [x] Audit difficulty gap: nested_squares sweep now includes depth 2-8, reduction_factor 0.4-0.8; touching_circles adds finer boundary distances and smaller diameters
- [x] Create `verify_determinism.py` script
- [x] Re-generate all 9 tasks with N≥50 per task (1,226 total samples)
- [x] Re-evaluate all 1,226 samples (0% parse failures!)
- [ ] Run determinism verification (requires API calls)

## Phase 9: Reasoning Mode (Extended Thinking)
- [x] Add `THINKING_BUDGET = 4096` to config.py
- [x] Add reasoning support to VisionClient + evaluate_manifest in harness.py
- [x] Add `--reasoning` flag to CLI evaluate subcommand
- [x] Add comparison table and grouped bar chart to analysis.py
- [x] Normalize missing `reasoning_mode` field for backward compatibility
- [x] Run full reasoning evaluation on all 1,226 samples
- [x] Generate comparison analysis and plots

---

## Summary of All Results (Rigorous, Phase 8)

| Task | N | Accuracy | 95% CI | Key Finding |
|------|---|----------|--------|-------------|
| line_intersection_text | 60 | 20.0% | [11.8%, 31.8%] | Reasoning failure (mean_error=+15.45) |
| arrow_following | 50 | 34.0% | [22.4%, 47.8%] | Real blind spot (was 20% with null prompts) |
| nested_squares | 315 | 51.7% | [46.2%, 57.2%] | Overcounting (+1.48), now harder params |
| line_intersection | 175 | 55.4% | [48.0%, 62.6%] | Overcounting (+0.14) |
| touching_circles | 396 | 74.5% | [70.0%, 78.5%] | Boundary confusion at small gaps |
| line_chart_crossing | 80 | 81.2% | [71.3%, 88.3%] | Overcounting (+0.80) |
| counting_grid | 50 | 100.0% | [92.9%, 100.0%] | Perfect on default 5×6 |
| form_checkboxes | 50 | 100.0% | [92.9%, 100.0%] | Perfect |
| table_cell_read | 50 | 100.0% | [92.9%, 100.0%] | Was 0% due to null-prompt bug |
| **TOTAL** | **1226** | **65.2%** | **[62.5%, 67.8%]** | |

## Reasoning Mode Comparison (Phase 9)

| Task | N | No-Reasoning | Reasoning | Delta |
|------|---|-------------|-----------|-------|
| arrow_following | 50 | 34.0% | 46.0% | +12.0% |
| counting_grid | 50 | 100.0% | 90.0% | -10.0% |
| form_checkboxes | 50 | 100.0% | 100.0% | +0.0% |
| line_chart_crossing | 80 | 81.2% | 96.2% | +15.0% |
| line_intersection | 175 | 55.4% | 53.7% | -1.7% |
| line_intersection_text | 60 | 20.0% | 90.0% | +70.0% |
| nested_squares | 315 | 51.7% | 55.9% | +4.1% |
| table_cell_read | 50 | 100.0% | 100.0% | +0.0% |
| touching_circles | 396 | 74.5% | 78.5% | +4.0% |
| **TOTAL** | **1226** | **65.2%** | **71.8%** | **+6.6%** |

### Key Findings — Reasoning Mode
1. **Overall +6.6% improvement** (65.2% → 71.8%) with extended thinking
2. **line_intersection_text: +70.0%** — biggest gain. Reasoning dramatically helps compute segment intersections from coordinates (20% → 90%)
3. **line_chart_crossing: +15.0%** — reasoning helps count crossings more accurately
4. **arrow_following: +12.0%** — reasoning improves arrow path tracing
5. **counting_grid: -10.0%** — reasoning *hurts* on this already-easy task (100% → 90%), likely overthinking
6. **line_intersection: -1.7%** — essentially unchanged, visual perception is the bottleneck
7. Tasks already at 100% (form_checkboxes, table_cell_read) remain perfect

### Key Corrections from Phase 8
1. **table_cell_read: 0% → 100%** — entirely a measurement artifact (null prompts in manifest)
2. **arrow_following: 20% → 34%** — partially artifact, but still a genuine blind spot
3. **nested_squares: 97.9% → 51.7%** — adding harder depth (6-8) and reduction factors exposed true difficulty
4. **touching_circles: 90.9% → 74.5%** — finer boundary distances revealed confusion
5. **0% parse failures** across all tasks — parser fixes eliminated all parsing errors

## Files Created
- `config.py`, `parsers.py`, `scorers.py`, `harness.py`, `cli.py`, `analysis.py`
- `tasks/__init__.py`, `tasks/counting_grid.py`, `tasks/line_intersection.py`
- `tasks/touching_circles.py`, `tasks/nested_squares.py`
- `tasks/table_cell_read.py`, `tasks/line_chart_crossing.py`
- `tasks/arrow_following.py`, `tasks/form_checkboxes.py`
- `tasks/line_intersection_text.py`
- `loaders/__init__.py`, `loaders/blindtest_loader.py`
- `requirements.txt`, `sprint_zero.py`
