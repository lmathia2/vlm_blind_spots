# Plan: Visual Sketchpad Strategy

## Background

The removed `repl_vision` strategy failed (-11.9p) because it asked the model to *write* image analysis code. The model generated buggy code, fell into mode confusion, and produced worse results than just looking at the image directly.

**Visual Sketchpad** (Hu et al., 2024, arXiv:2410.08165) takes the opposite approach: instead of asking the VLM to write code, run **pre-built vision primitives** programmatically and feed the **annotated image** back to the VLM. The model sees the results visually (highlighted contours, numbered markers, measurement lines) rather than reading text output. This plays to the VLM's strengths (visual perception) while compensating for its weaknesses (precise counting, proportion estimation).

**Key insight**: The problem is not that the model can't see pixels — it's that it can't *organize* its visual reasoning. Externalizing intermediate analysis as image annotations gives the model visual scaffolding.

## Design

### Architecture

```
Input: image + prompt + task_name
  │
  ├─ 1. Run task-specific vision primitives (deterministic, no model)
  │     → produces: annotated_image + text_findings
  │
  ├─ 2. Send annotated_image + findings + original prompt to VLM
  │     → model answers using both visual annotations and text context
  │
  └─ 3. Parse answer as usual
```

This is a **2-call strategy** (like verify): one programmatic analysis step + one VLM call on the annotated image. No iterative loop, no model-generated code.

### Vision Primitives Library

Pre-built, tested functions that analyze images and return both annotations and text findings:

| Primitive | Input | Output | Used By |
|-----------|-------|--------|---------|
| `detect_edges` | image | edge overlay image | counting_grid, nested_squares |
| `count_line_transitions` | image, axis | count + annotated scan lines | counting_grid |
| `detect_contours` | image | contour overlay + count | nested_squares |
| `segment_colors` | image | color masks + area ratios | pie_chart, colored_paths, progress_bar |
| `measure_bar_fill` | image, bar_region | percentage + measurement line | progress_bar |
| `detect_boxes` | image | bbox overlay + positions | hierarchy_depth, realistic_table |
| `cluster_by_y` | boxes | level groupings | hierarchy_depth |
| `crop_and_enhance` | image, region | enhanced crop | text_degradation, realistic_table |
| `trace_colored_paths` | image, colors | path overlay + endpoint labels | colored_paths |
| `detect_points` | image, color | point positions + coordinate labels | scatter_plot |

### Per-Task Analysis Plans

#### counting_grid (10% → target: 40%+)
1. `count_line_transitions(image, axis="horizontal")` → count vertical lines by scanning a middle row for dark-light transitions
2. `count_line_transitions(image, axis="vertical")` → count horizontal lines by scanning a middle column
3. Annotate: draw colored tick marks at each detected transition, write counts on image
4. Text findings: "Detected {h} horizontal lines and {v} vertical lines"

#### pie_chart (25% → target: 50%+)
1. `segment_colors(image)` → find distinct non-white color clusters, compute pixel area for each
2. Annotate: overlay percentage labels on each slice, draw radial reference lines at 25%/50%/75%
3. Text findings: "Slice areas by color: blue=38.2%, gray=31.1%, ..."

#### colored_paths (60% → target: 65%+)
1. `segment_colors(image, exclude_white=True, exclude_black=True)` → isolate each colored path
2. For each color mask: find connected components, trace endpoints
3. Annotate: label each path with its color and endpoint stations (proximity to station markers)
4. Text findings: "Red path: A→C. Blue path: A→E. Green path: B→D."

#### nested_squares (55% → target: 70%+)
1. `detect_contours(image, min_area=50)` → find all closed rectangular contours
2. Sort by area (largest→smallest), number them
3. Annotate: draw each detected contour in a different color, label with number
4. Text findings: "Detected {n} nested squares (numbered 1-{n} from outside to inside)"

#### hierarchy_depth (61% → target: 85%+)
1. `detect_boxes(image)` → find all rectangular regions (boxes in the org chart)
2. `cluster_by_y(boxes)` → group boxes into horizontal rows by y-coordinate
3. Annotate: draw horizontal colored bands across each detected level, label "Level 1", "Level 2", etc.
4. Text findings: "Detected {n} horizontal levels of boxes"

#### realistic_table (75% → target: 85%+)
1. Parse the prompt to extract target row name and column header
2. `detect_boxes(image)` → find table cell boundaries via grid line detection
3. `crop_and_enhance(image, target_cell)` → crop the specific cell, enlarge 3x
4. Annotate: highlight the target cell with a colored border on the full image; include enlarged crop
5. Text findings: "Target cell is at row '{row}', column '{col}'. Enlarged view attached."

#### progress_bar (39% → target: 60%+)
1. `segment_colors(image)` → find the green (filled) and gray (unfilled) regions
2. For each bar region: measure filled_width / total_width
3. Annotate: draw measurement lines showing filled vs total width, write percentage on each bar
4. Text findings: "Bar 'Processing': 42% filled (187px / 445px total)"

#### scatter_plot (70% → target: 75%+)
1. Parse prompt to extract target color and x-value
2. `detect_points(image, target_color)` → find colored markers, map pixel positions to data coordinates using axis ticks
3. Annotate: draw crosshair lines at the target point, label with estimated coordinates
4. Text findings: "Blue point at x=5: y≈23 (pixel position 312, 198)"

#### text_degradation (80% → target: 85%+)
1. `crop_and_enhance(image)` → auto-crop text region, sharpen, increase contrast, de-rotate
2. Annotate: show the enhanced version alongside (or replace) the original
3. Text findings: "Enhanced image attached. Original had blur={b}, contrast={c}."

## Implementation Plan

### Files to modify

| File | Changes |
|------|---------|
| `sketchpad.py` (NEW) | Vision primitives library + per-task analysis plans |
| `strategies.py` | Add `strategy_sketchpad()` that calls sketchpad analysis |
| `cli.py` | Add `"sketchpad"` to strategy choices |
| `benchmark_strategies.py` | Add to ALL_STRATEGIES |
| `tests/test_sketchpad.py` (NEW) | Unit tests for vision primitives |
| `tests/test_strategies.py` | Update registry test, add sketchpad strategy test |

### Implementation sequence

1. Create `sketchpad.py` with vision primitives library
2. Add per-task analysis plans to `sketchpad.py`
3. Add `strategy_sketchpad()` to `strategies.py`
4. Wire into CLI and benchmark runner
5. Write unit tests for primitives
6. Write strategy tests
7. Run full test suite
8. Benchmark on 176 samples

### Key design decisions

- **No external models** — all primitives use PIL, numpy, and optionally OpenCV. No SAM, no GroundingDINO. This keeps the strategy self-contained and runnable locally.
- **Deterministic analysis** — the vision primitives are pure functions. No model calls, no randomness. The only model call is the final VLM query on the annotated image.
- **2-call strategy** — one programmatic step + one VLM call. Simple, fast, predictable cost.
- **Annotated image as primary output** — the VLM sees the annotations visually, not as text. This is the core distinction from code_vision/repl_vision.
- **Text findings as supplement** — a brief text summary accompanies the annotated image, giving the model both visual and textual scaffolding.

### Risks

- **Primitive accuracy**: If edge detection miscounts lines or contour detection misses squares, the annotations will mislead the model. Mitigation: validate primitives against known ground truth before benchmarking.
- **Annotation clutter**: Too many overlays might obscure the original image. Mitigation: use semi-transparent overlays and place labels in margins.
- **Image size**: Annotated images may be larger (higher resolution for annotations), which could exceed the VLM's context window. Mitigation: keep output images at 768px max.
- **Task identification**: Strategy needs `task_name` to select the right analysis plan. Unknown tasks fall back to a generic edge+contour analysis.
