# Plan: Visual Sketchpad Strategy

## Background

The removed `repl_vision` strategy failed (-11.9p) because it asked the model to *write* image analysis code. The model generated buggy code, fell into mode confusion, and produced worse results than just looking at the image directly.

**Visual Sketchpad** (Hu et al., 2024, arXiv:2410.08165) takes the opposite approach: instead of asking the VLM to write code, run **pre-built vision primitives** programmatically and feed the **annotated image** back to the VLM. The model sees the results visually (highlighted contours, numbered markers, measurement lines) rather than reading text output. This plays to the VLM's strengths (visual perception) while compensating for its weaknesses (precise counting, proportion estimation).

**Key insight**: The problem is not that the model can't see pixels — it's that it can't *organize* its visual reasoning. Externalizing intermediate analysis as image annotations gives the model visual scaffolding.

## Design

### Architecture: Multi-Pass Query-Conditioned Visual Sketchpad

```
Input: image + prompt + task_name (may be unknown)

Step 0: Decompose question into sub-questions
  → decompose_question(prompt, task_name) → list of sub-questions
  → Each sub-question targets one aspect of the original query
  → e.g., "What percentage does the blue slice represent?"
       → sub_q1: "What distinct color regions exist in the image?"
       → sub_q2: "What is the area of the blue region relative to total?"
  → e.g., "How many levels are in the hierarchy?"
       → sub_q1: "Where are the boxes/nodes in the image?"
       → sub_q2: "How many horizontal rows do they form?"
  → For simple queries, may return just [original_prompt] (no decomposition)

Step 1: For each sub-question, infer an analysis plan
  → classify_query(sub_question, task_name) → list of (primitive, args) pairs
  → uses keyword matching on the sub-question + task_name if available

Step 2 (Pass 0 — automatic, no model call):
  Execute all sub-question primitive sequences in order
  → annotate sketchpad_image, accumulate findings[] per sub-question
  → Each sub-question's findings are labeled: "Sub-Q1: ...", "Sub-Q2: ..."

Step 3 (Pass 1..N — model-driven, max 2 additional passes):
  1. Send [sketchpad_image] + [all sub-question findings] + [available tools] + [prompt] to VLM
  2. Model responds with either:
     a) TOOL(primitive_name, args) — request another primitive
     b) ANSWER(response) — final answer
  3. If TOOL: execute primitive, update sketchpad_image + findings, go to next pass
  4. If ANSWER: parse and return

Step 4: Aggregate results
  → If multiple sub-questions produced independent findings, combine them
  → Aggregation method depends on question type:
     - Counting: sum or max across sub-questions
     - Comparison: select the relevant sub-question's answer
     - Composition: model synthesizes from all sub-question findings (done in Step 3)
  → For most cases, the model handles aggregation naturally in Step 3's ANSWER

Max passes: 3 (1 automatic + up to 2 model-driven)
```

### Question Decomposition (`decompose_question`)

Before selecting primitives, the question is decomposed into sub-questions that can each be independently analyzed. This handles multi-aspect questions and provides structured intermediate results.

```python
# Task-aware decomposition templates
DECOMPOSITION_TEMPLATES = {
    "pie_chart": [
        "What distinct color regions exist in the image and what are their relative areas?",
        "What is the area of the {target} region as a percentage of the whole?",
    ],
    "counting_grid": [
        "How many horizontal lines are in the grid?",
        "How many vertical lines are in the grid?",
    ],
    "hierarchy_depth": [
        "Where are the boxes or nodes positioned in the image?",
        "How many distinct horizontal rows do the boxes form?",
    ],
    "progress_bar": [
        "Where are the progress bars in the image?",
        "What fraction of each bar is filled vs unfilled?",
    ],
    "colored_paths": [
        "What distinct colored paths exist in the image?",
        "What are the start and end stations for each colored path?",
    ],
    "realistic_table": [
        "Where are the table cell boundaries?",
        "What is the content of the cell at the target row and column?",
    ],
    "nested_squares": [
        "What rectangular contours exist in the image?",
        "How many concentric squares are there from outside to inside?",
    ],
    "scatter_plot": [
        "Where are the data points in the image?",
        "What is the y-value of the {target_color} point at x={target_x}?",
    ],
    "text_degradation": [
        "What text region needs to be read?",
    ],
}

# For unknown tasks or when task_name is unavailable, use query-based heuristics:
# - Questions with "and" or multiple question marks → split into parts
# - Questions asking "what percentage of X" → decompose into "find X" + "measure X"
# - Single-focus questions → no decomposition, pass through as-is
```

**Why decomposition helps:**
- **Counting grid**: Separating horizontal vs vertical line counting prevents the model from confusing the two dimensions
- **Pie chart**: First identifying all slices, then measuring the target slice, avoids skipping small slices
- **Hierarchy**: Separating box detection from level counting lets the model focus on each step
- **Simple questions**: Pass through without decomposition — no overhead for easy tasks

### Query Classification (`classify_query`)

The analysis plan is selected based on **what the question asks**, not just the task name. This makes the strategy work for unknown tasks and avoids hard-coding task_name → primitive mappings.

```python
# Query signals → analysis category → default primitives
QUERY_PATTERNS = [
    # Counting lines/grid
    (r"(grid|lines|rows.*columns|columns.*rows)", "grid_counting",
     [("count_line_transitions", "horizontal"),
      ("count_line_transitions", "vertical")]),

    # Counting shapes (squares, circles, etc.)
    (r"(count|how many).*(square|circle|shape|triangle|rectangle)", "shape_counting",
     [("detect_contours",)]),

    # Hierarchy / levels / depth
    (r"(level|depth|hierarch|layers?\b|how many.*deep)", "hierarchy",
     [("detect_boxes",), ("cluster_by_y",)]),

    # Percentage / proportion (pie chart, progress bar)
    (r"(percentage|percent|proportion|slice.*represent|progress.*bar)", "proportion",
     [("segment_colors",)]),

    # Path counting / connections
    (r"(path|route|connect|from.*to)", "path_tracing",
     [("segment_colors",), ("trace_colored_paths",)]),

    # Table / cell lookup
    (r"(table|row|column|cell|what is the.*for)", "table_lookup",
     [("detect_boxes",), ("crop_and_enhance", "target_cell")]),

    # Scatter plot / coordinate reading
    (r"(y-value|x-value|scatter|point at|coordinate)", "scatter",
     [("detect_points",)]),

    # Text reading
    (r"(text|read|say|written|OCR)", "text_reading",
     [("crop_and_enhance", "text_region")]),
]

# Fallback: if no pattern matches, use generic edge + contour detection
FALLBACK_PLAN = [("detect_edges",), ("detect_contours",)]
```

The `task_name` field (if available) acts as a tiebreaker when multiple patterns match, and as an override for known tasks where the query wording is ambiguous.

**Priority order**: query keywords > task_name hints > fallback

**Why query-conditioned matters:**
- The same image type can have different questions (e.g., pie_chart: "what percentage" vs "how many slices" → different primitives)
- Unknown tasks (not in the 9 benchmarks) still get appropriate analysis based on what the question asks
- The prompt often contains stronger signal than the task name (e.g., "count the number of horizontal rows" directly tells us to do box detection + y-clustering)

**Pass 0** decomposes the question into sub-questions, classifies each, and runs the inferred primitive sequences automatically. This is still deterministic and free (no model call), but the *choice* and *ordering* of primitives is query-conditioned.

**Passes 1-N** remain model-driven: the model can request additional primitives beyond the initial plan, or just answer directly.

**Why multi-pass helps over single-pass:**
- The model can request a **zoom/crop** after seeing the initial analysis (e.g., contour detection found 5 squares but the center looks suspicious → request `crop_and_enhance` on center region)
- The model can **validate** programmatic findings against its own perception (e.g., "the color segmentation says 38% but that looks closer to 30% to me")
- The model can request **different primitives** than the default set (e.g., for an unusual pie chart, request edge detection instead of color segmentation)

**Why this won't degrade like repl_vision:**
- The model doesn't write code — it selects from a fixed menu of tested primitives
- Each primitive is self-contained and correct — no accumulating errors
- The model sees visual results (annotated images), not text stdout
- Max 3 passes prevents self-doubt cascades
- Pass 0 is free (no model call) and provides a strong starting point

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
| `sketchpad.py` (NEW) | Vision primitives library + `decompose_question()` + `classify_query()` + per-task analysis plans |
| `strategies.py` | Add `strategy_sketchpad()` that calls sketchpad analysis |
| `cli.py` | Add `"sketchpad"` to strategy choices |
| `benchmark_strategies.py` | Add to ALL_STRATEGIES |
| `tests/test_sketchpad.py` (NEW) | Unit tests for vision primitives |
| `tests/test_strategies.py` | Update registry test, add sketchpad strategy test |

### Implementation sequence

1. Create `sketchpad.py` with vision primitives library
2. Add `decompose_question()` and `classify_query()` to `sketchpad.py`
3. Add per-task analysis plans to `sketchpad.py`
4. Add `strategy_sketchpad()` to `strategies.py`
5. Wire into CLI and benchmark runner
6. Write unit tests for primitives and decomposition
7. Write strategy tests
8. Run full test suite
9. Benchmark on 176 samples

### Key design decisions

- **No external models** — all primitives use PIL, numpy, and optionally OpenCV. No SAM, no GroundingDINO. This keeps the strategy self-contained and runnable locally.
- **Pass 0 is deterministic** — task-specific primitives run automatically without a model call. This guarantees useful annotations even if the model doesn't request more.
- **Multi-pass with hard cap** — max 3 passes (1 auto + 2 model-driven). Enough for the model to refine analysis without self-doubt cascades.
- **Tool selection, not code generation** — the model picks from a fixed menu (`TOOL(name, args)`) rather than writing arbitrary code. This eliminates syntax errors and API misuse.
- **Annotated image as primary output** — the VLM sees the annotations visually, not as text. This is the core distinction from code_vision/repl_vision.
- **Text findings as supplement** — a brief text summary accompanies the annotated image, giving the model both visual and textual scaffolding.
- **Cumulative sketchpad** — annotations from all passes persist on the same canvas. The model sees everything it has done so far.

### Model prompt format

Pass 1+ prompt structure:
```
You are analyzing an image with a visual sketchpad. The image has been
annotated with analysis results from vision tools.

The question was decomposed into sub-questions. Analysis so far:

Sub-Q1: {sub_question_1}
  Findings: {findings_1}

Sub-Q2: {sub_question_2}
  Findings: {findings_2}

Available tools (reply with TOOL(name, args) to use one):
- crop_and_enhance(region): Zoom into a region ("center", "top-left", etc.)
- detect_contours(): Find and highlight shape boundaries
- segment_colors(): Isolate distinct color regions with area percentages
- count_line_transitions(axis): Count lines along "horizontal" or "vertical"
- measure_bar_fill(): Measure filled vs total width of bars
- detect_boxes(): Find rectangular boxes and their positions

Synthesize the sub-question findings and reply with ANSWER followed by
your response to the original question:
{original_prompt}
```

Parsing: regex `TOOL\((\w+)(?:,\s*(.+?))?\)` for tool calls, `ANSWER\s*(.+)` for final answer.

### Risks

- **Primitive accuracy**: If edge detection miscounts lines or contour detection misses squares, the annotations will mislead the model. Mitigation: validate primitives against known ground truth before benchmarking.
- **Annotation clutter**: Too many overlays might obscure the original image. Mitigation: use semi-transparent overlays and place labels in margins.
- **Image size**: Annotated images may be larger (higher resolution for annotations), which could exceed the VLM's context window. Mitigation: keep output images at 768px max.
- **Task identification**: Strategy needs `task_name` to select the right analysis plan. Unknown tasks fall back to a generic edge+contour analysis.
