# Business Workflow Perception Tests — Implementation Plan

Spec: `reference/perception_tasks.md` (28 tests, 7 categories, 9 perceptual primitives)

---

## Inventory: What Exists vs What's New

### Already built (9 tasks → map to spec)

| Spec ID | Task File | Status | Notes |
|---------|-----------|--------|-------|
| T1.1 | `counting_grid.py` | ✅ Done | Recently enhanced with merged cells + multi-question |
| T1.2 | `table_cell_read.py` | ✅ Done | Grid with 2-digit numbers, exact cell lookup |
| T2.1 | `line_chart_crossing.py` | ✅ Done | Two-series chart, crossing count |
| T3.1 | `arrow_following.py` | ✅ Done | Labeled boxes + directed arrows |
| T4.1 | `form_checkboxes.py` | ✅ Done | Checkbox state detection, SET answer |
| T7.1 | `line_intersection.py` | ✅ Done | Clean geometric line intersection |
| T7.2 | `touching_circles.py` | ✅ Done | Binary touching/overlap discrimination |
| T7.3 | `nested_squares.py` | ✅ Done | Nested shape counting |
| — | `line_intersection_text.py` | ✅ Done | Perception vs reasoning diagnostic |

### New tasks to build (19)

| Spec ID | Task Name | Category | Answer Format | Primitives | Tier |
|---------|-----------|----------|---------------|------------|------|
| T1.3 | `realistic_table` | Tables | EXACT str | P1, P8 | 3 |
| T1.4 | `merged_cell_read` | Tables | EXACT str | P1, P3 | 4 |
| T2.2 | `bar_chart_value` | Charts | MC4 | P9, P8, P1 | 2 |
| T2.3 | `line_chart_point` | Charts | MC4 | P1, P2, P9 | 4 |
| T2.4 | `legend_association` | Charts | EXACT str | P7, P8, P9 | 3 |
| T2.5 | `pie_chart` | Charts | MC4 | P9, P7, P8 | 4 |
| T2.6 | `stacked_bar` | Charts | MC4 | P7, P9, P1 | 4 |
| T3.2 | `decision_flowchart` | Diagrams | EXACT str | P2, P6, P8 | 3 |
| T3.3 | `edge_crossing` | Diagrams | Yes/No | P2, P4 | 4 |
| T3.4 | `hierarchy_depth` | Diagrams | EXACT int | P3, P2, P1 | 4 |
| T4.2 | `radio_button` | Forms | EXACT str | P5, P8, P1 | 4 |
| T4.3 | `form_field` | Forms | EXACT str | P8, P1 | 3 |
| T4.4 | `progress_bar` | Forms | MC4 | P9 | 4 |
| T5.1 | `circled_text` | Annotations | EXACT str | P1, P8 | 3 |
| T5.2 | `arrow_annotation` | Annotations | EXACT str | P2, P6, P1 | 4 |
| T5.3 | `strikethrough` | Annotations | SET | P5, P8 | 4 |
| T6.1 | `text_degradation` | Text/OCR | EXACT str | P8 | 2 |
| T6.2 | `rotated_text` | Text/OCR | EXACT str | P8 | 2 |
| T6.3 | `dense_text` | Text/OCR | EXACT str | P8, P1, P3 | 4 |

---

## Framework Changes Required

### F1. Add `mc4` parser (`parsers.py`)

Extract letter A/B/C/D from model response. The existing `letter` parser deprioritizes "A" and "I" which are valid MC4 answers. Need a dedicated parser that:
- Tries `{A}` format first
- Tries `(A)` format
- Tries "answer is A" / standalone A/B/C/D
- No skip list — all four letters are always valid

Ground truth for MC4 tasks = the correct letter (e.g., "B"). The `exact_match` scorer handles the rest.

### F2. Add `exact_string` parser (`parsers.py`)

For tasks where the answer is an arbitrary string (currency, name, text). Strategy:
- Try `{...}` format first (e.g., `{$1,456}`)
- Try quoted string (`"..."` or `'...'`)
- Fall back to full response stripped of preamble ("The answer is ...", "The text says ...")

Pair with `exact_match` scorer (case-insensitive string comparison).

### F3. Add clutter tax analysis (`analysis.py`)

New function `print_clutter_tax(results_path)` that:
- Defines matched pairs: T7.1↔T2.1, T7.2↔T4.1, T7.4↔T3.1
- Computes accuracy delta (clean − business) for each pair
- Prints table with both accuracies and the gap
- Generates grouped bar chart to `report_assets/clutter_tax.png`

### F4. MC4 distractor generation utility

Create `mc4_utils.py` with:
- `generate_distractors(correct_value, other_values, n=3, min_spacing_pct=0.15)` — picks plausible nearby distractors from other elements in the image + random offsets
- `format_mc4_prompt(question, options)` — shuffles options, assigns A/B/C/D, returns (prompt_str, correct_letter)
- Used by T2.2, T2.3, T2.5, T2.6, T4.4

---

## Implementation Phases

### Phase 0: Framework prep
- [x] Add `mc4` parser to `parsers.py`
- [x] Add `exact_string` parser to `parsers.py`
- [x] Create `mc4_utils.py` with distractor generation + prompt formatting
- [x] Verify existing tasks still work after parser additions (run generate on one task)

### Phase 1: Tier 2 — First new signal (4 tasks)

These are the highest-value new tasks not already built.

**T6.1 — `text_degradation.py`** (~35 lines)
- Render short business strings ("Total: $42,387.19", "Invoice #INV-2024-0892") with PIL
- Apply degradation: Gaussian blur (ImageFilter), contrast reduction (ImageEnhance), noise, rotation
- Parser: `exact_string`. Scorer: `exact_match`
- Sweep: font_size [8–28], blur_radius [0–3], rotation [0°, 2°, 5°], contrast [1.0→0.3]
- This is the single most common real-world VLM failure

**T6.2 — `rotated_text.py`** (~20 lines)
- Render text at rotation angles using PIL `Image.rotate()`
- Parser: `exact_string`. Scorer: `exact_match`
- Sweep: rotation [0°, 15°, 30°, 45°, 60°, 90°], font_size [10–24]

**T2.2 — `bar_chart_value.py`** (~40 lines)
- Matplotlib vertical bar chart, no value labels on bars
- MC4 question about a specific bar's height
- Uses `mc4_utils` for distractor generation (heights of adjacent bars ± offsets)
- Parser: `mc4`. Scorer: `exact_match`
- Sweep: n_bars [3–8], value_range, tick_spacing

**T3.1 — arrow_following (already exists, review sweep coverage)**
- Verify existing implementation matches spec
- Ensure sweep axes cover the difficulty range in the spec

### Phase 2: Tier 3 — Chart + form + annotation depth (5 tasks)

**T1.3 — `realistic_table.py`** (~60 lines)
- PIL table with string headers (Name, Q1, Q2, YoY%), mixed data types
- Alternating row shading, header bold
- Parser: `exact_string`. Scorer: `exact_match`
- Sweep: n_rows [4–12], n_cols [3–6], font_size [9–18]

**T3.2 — `decision_flowchart.py`** (~80 lines)
- 4–5 fixed template layouts with decision diamonds + process boxes
- Labels/conditions randomized per sample
- Question provides condition outcomes, asks for final result
- Parser: `exact_string`. Scorer: `exact_match`
- Hardest rendering task — use fixed PIL coordinates per template

**T2.4 — `legend_association.py`** (~45 lines)
- Matplotlib 3–4 series line chart with configurable legend position
- Ask which series has highest peak
- Parser: `exact_string`. Scorer: `exact_match`
- Sweep: n_series [2–4], color_similarity [distinct/similar], legend_position

**T5.1 — `circled_text.py`** (~40 lines)
- PIL rendered sentence with red ellipse around one word
- Parser: `exact_string`. Scorer: `exact_match`
- Sweep: font_size [14–36], ellipse_thickness [1–4]

**T4.3 — `form_field.py`** (~45 lines)
- PIL vertical form with "Label: [value]" pairs in bordered boxes
- Ask for value of a specific labeled field
- Parser: `exact_string`. Scorer: `exact_match`
- Sweep: n_fields [5–12], font_size [10–16], field_style [boxed/underlined]

### Phase 3: Tier 4 — Remaining coverage (10 tasks)

Build in priority order within the tier:

1. **T7.4 — `colored_paths.py`** (~50 lines) — completes matched pairs
2. **T2.5 — `pie_chart.py`** (~35 lines, MC4)
3. **T2.6 — `stacked_bar.py`** (~50 lines, MC4)
4. **T3.3 — `edge_crossing.py`** (~60 lines) — bridge gap convention
5. **T3.4 — `hierarchy_depth.py`** (~50 lines) — org chart tree
6. **T5.2 — `arrow_annotation.py`** (~35 lines)
7. **T5.3 — `strikethrough.py`** (~30 lines)
8. **T6.3 — `dense_text.py`** (~25 lines)
9. **T4.4 — `progress_bar.py`** (~30 lines, MC4, easiest task)
10. **T4.2 — `radio_button.py`** (~40 lines)
11. **T1.4 — `merged_cell_read.py`** (~50 lines)
12. **T2.3 — `line_chart_point.py`** (~40 lines, MC4)

### Phase 4: Evaluation + analysis
- [x] Generate all tasks with `--task all --sweep --min-samples 50` (2,130 samples across 29 tasks)
- [x] Evaluate full dataset (2,230 samples against Claude Haiku 4.5 — 72.1% overall)
- [x] Add clutter tax analysis to `analysis.py` (`print_clutter_tax`, `plot_clutter_tax`)
- [x] Generate accuracy plots and clutter tax chart
- [x] Run matched-pair clutter tax comparison
- [x] Fix strikethrough (0% → 73%): ground truth used letter labels, model returned word names

### Phase 5: Perception vs reasoning diagnostics
- [x] Add text-only controls for: T7.1/T2.1 (line intersection), T7.3 (nested squares), T3.2 (flowchart), T1.1 (grid)
- [x] Compare image vs text accuracy for each
- [x] Classify each failure as perceptual vs reasoning

---

## Risks & Edge Cases

1. **Font availability**: Tasks using PIL text rendering need DejaVuSans. Fall back to PIL default if unavailable. Check on macOS (`/Library/Fonts/` or bundled).
2. **MC4 parser reliability**: Model may respond with "(A) 42" instead of just "A". Parser must handle this.
3. **T3.2 complexity**: Decision flowcharts at ~80 lines are the most complex rendering. Use fixed coordinate templates to keep it tractable.
4. **Exact string matching fragility**: Model might respond "$1,456" when ground truth is "$1,456.00" or "1456". Need to define normalization rules per task or use lenient matching.
5. **Distractor quality**: Bad distractors make MC4 tasks trivially easy. Distractor generation must use values from the same image (adjacent bars, neighboring data points).
6. **Color discrimination**: T2.4 with similar colors (blue vs teal) requires careful color palette design. Define explicit color sets.

---

## Verification Criteria

Each task must pass these checks before marking complete:
1. `render()` produces valid images with correct ground truth for ≥10 random seeds
2. Ground truth matches visual inspection of 3+ generated images
3. Parser handles the expected model response format (test with synthetic responses)
4. Sweep generates balanced samples across ground truth classes
5. Full pipeline works: `generate → evaluate → analyze` produces valid results

---

## File Manifest (after all phases)

```
tasks/
├── __init__.py              # existing (auto-discovery)
├── counting_grid.py         # T1.1 (exists)
├── table_cell_read.py       # T1.2 (exists)
├── realistic_table.py       # T1.3 (new)
├── merged_cell_read.py      # T1.4 (new)
├── line_chart_crossing.py   # T2.1 (exists)
├── bar_chart_value.py       # T2.2 (new, MC4)
├── line_chart_point.py      # T2.3 (new, MC4)
├── legend_association.py    # T2.4 (new)
├── pie_chart.py             # T2.5 (new, MC4)
├── stacked_bar.py           # T2.6 (new, MC4)
├── arrow_following.py       # T3.1 (exists)
├── decision_flowchart.py    # T3.2 (new)
├── edge_crossing.py         # T3.3 (new)
├── hierarchy_depth.py       # T3.4 (new)
├── form_checkboxes.py       # T4.1 (exists)
├── radio_button.py          # T4.2 (new)
├── form_field.py            # T4.3 (new)
├── progress_bar.py          # T4.4 (new, MC4)
├── circled_text.py          # T5.1 (new)
├── arrow_annotation.py      # T5.2 (new)
├── strikethrough.py         # T5.3 (new)
├── text_degradation.py      # T6.1 (new)
├── rotated_text.py          # T6.2 (new)
├── dense_text.py            # T6.3 (new)
├── line_intersection.py     # T7.1 (exists)
├── touching_circles.py      # T7.2 (exists)
├── nested_squares.py        # T7.3 (exists)
├── colored_paths.py         # T7.4 (new)
├── line_intersection_text.py # diagnostic (exists)
mc4_utils.py                 # MC4 distractor gen + prompt formatting (new)
parsers.py                   # + mc4, exact_string parsers (modify)
analysis.py                  # + clutter_tax comparison (modify)
```
