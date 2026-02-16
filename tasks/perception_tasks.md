iple choice** to maintain exact scoring.

### Design Principles

1. **Every test has unambiguous ground truth.** No "approximate" answers. Tests are either exact-answer, binary (yes/no), categorical (A/B/C/D), or multiple-choice with one correct option.
2. **Multiple choice for inherently continuous values.** When a test asks about a quantity that's hard to read precisely (bar height, pie slice percentage), present 4 randomized options where distractors are chosen to be plausible but wrong. This preserves exact scoring while testing genuine perceptual ability.
3. **Matched clean/contextual pairs.** For key primitives, test both the raw geometric version and the business-context version to quantify the "clutter tax."
4. **Every test must be implementable in <60 lines of rendering code** using PIL or matplotlib. Tests requiring complex layout engines are either simplified with fixed templates or deprioritized.

---

## Part 1: Business Workflow Domains

### A. Tabular Data & Spreadsheets
Reading tables in PDFs, spreadsheets, financial reports, invoices. Extracting specific cell values, understanding structure, comparing across cells.

### B. Charts & Data Visualization
Interpreting line charts, bar charts, scatter plots, pie charts. Reading values, detecting crossings, comparing series, matching legends to data.

### C. Diagrams & Flowcharts
Following flowcharts, org charts, network diagrams. Tracing directed paths, understanding hierarchy, resolving visual edge crossings.

### D. Document Layout & Forms
Understanding form fields, multi-column text, header hierarchies, captions. Associating labels to values, determining structure.

### E. Annotations & Markup
Reviewing highlights, circles, arrows, strikethroughs, tracked changes. Identifying what's marked and what annotations refer to.

### F. Dashboards & UI
Reading software interfaces — progress bars, status indicators, buttons, menus. Identifying states and reading values from visual encodings.

---

## Part 2: Perceptual Primitives (P1–P9)

Reduced from 10 to 9 by merging P1/P9 and narrowing P6.

| ID | Primitive | Description | BlindTest | BabyVision |
|----|-----------|-------------|-----------|------------|
| P1 | **Spatial Reference Resolution** | Mapping a symbolic/ordinal/relational reference to a visual location. Includes: locating row 3 col 5, determining containment, adjacency, grouping, reading order. | — | Spatial Perception |
| P2 | **Line/Path Following** | Tracing a continuous path from start to end through crossings and turns | Task 7 | Visual Tracking |
| P3 | **Counting & Enumeration** | Counting discrete objects, rows, columns, nesting levels, especially with overlap | Tasks 4,5,6 | — |
| P4 | **Intersection Detection** | Determining whether lines cross and counting crossing points. Composite of P2+P1 but fails independently and catastrophically. | Task 1 | — |
| P5 | **Fine State Discrimination** | Distinguishing binary visual states at small scale: checked/unchecked, filled/unfilled, touching/separated, connected/disconnected | Task 2 | Fine-grained |
| P6 | **Symbol & Marker Recognition** | Identifying specific symbols in context: arrowhead direction, flowchart shape meaning (diamond=decision), chart marker types. Narrowly scoped to business-relevant symbols. | — | Fine-grained |
| P7 | **Color Discrimination & Mapping** | Matching colors to legends, distinguishing similar hues, reading color-coded status across distance | — | Fine-grained |
| P8 | **Text-in-Visual-Context** | Reading text at varied sizes, orientations, contrast levels, and degradation (blur, noise) embedded in visual elements | Task 3 | — |
| P9 | **Scale & Proportion** | Comparing relative lengths, heights, areas; mapping visual magnitude to numeric value. Tested via multiple choice to avoid ground truth ambiguity. | — | (EncQA) |

### How primitives combine

"What is Russia's revenue in 2006?" from a multi-series line chart requires:
- P7: match dotted green line to "Russia" in legend
- P2: follow that line to x=2006
- P1: locate the intersection point with the vertical at x=2006
- P9: map the y-position to the scale → pick from [18%, 24%, 31%, 42%]

---

## Part 3: Synthesizable Tests

### Answer Format Key
- **EXACT**: Single correct string (number, letter, word, Yes/No)
- **MC4**: Multiple choice with 4 randomized options, one correct. Distractors are plausible (nearby values, common confusion errors).
- **SET**: Unordered set of correct items (e.g., "which checkboxes are checked")

---

### Category 1: Tables & Grids

**T1.1 — Grid Structure Counting**
- *Render:* Black grid lines on white background (PIL)
- *Ask:* "How many rows and columns are in this grid?" → EXACT "rows=R columns=C"
- *Primitives:* P3
- *Params:* rows [3–12], cols [3–12], line_width [1–10px], resolution [256–1024]
- *Difficulty:* High counts + thin lines + low resolution
- *Workflow:* A
- *Implementation:* ~20 lines. Trivial.

**T1.2 — Cell Value Lookup (Numbers)**
- *Render:* Grid with 2-digit numbers in each cell (PIL with DejaVuSans)
- *Ask:* "What number is in row 3, column 5?" → EXACT integer
- *Primitives:* P1, P3, P8
- *Params:* rows [4–10], cols [4–8], font_size [8–24], line_width [1–5], resolution [256–1024]
- *Difficulty:* Small font + thin lines + large grid
- *Workflow:* A
- *Implementation:* ~40 lines. Most important business test.

**T1.3 — Cell Value Lookup (Realistic Table)**
- *Render:* Table with string headers (Name, Q1, Q2, YoY%), mixed data types (text, currency, percentage), occasional blank cells
- *Ask:* "What is the Q2 Revenue for Product B?" → EXACT string (e.g., "$1,456")
- *Primitives:* P1, P8, P1 (header-to-cell association)
- *Params:* n_rows [4–12], n_cols [3–6], font_size [9–18], header_bold [yes/no]
- *Difficulty:* More columns, smaller font, similar row values
- *Workflow:* A
- *Implementation:* ~60 lines. Rendering mixed data types and headers adds complexity but is manageable.

**T1.4 — Merged Cell Detection**
- *Render:* Grid where 1–3 cells span multiple rows or columns. Merge regions have internal grid lines removed and centered text.
- *Ask:* "What text is in the cell that spans columns 2–4 in row 1?" → EXACT string
- *Primitives:* P1, P3
- *Workflow:* A (financial tables, HTML tables)
- *Implementation:* ~50 lines. Suppress internal lines within merge, center text. Manageable.
- *Note:* Avoid "how many merged cells" — ambiguous. Ask about specific merged cells.

---

### Category 2: Charts & Visualization

**T2.1 — Line Chart Crossing Count**
- *Render:* Two line series (blue "Revenue", red "Cost") on matplotlib axes with gridlines, legend, axis labels
- *Ask:* "How many times do the Revenue and Cost lines cross?" → EXACT integer
- *Primitives:* P4, P2, P7
- *Params:* n_crossings [0–3], line_width [1–5], gridline_density [none/light/heavy], near_miss_count [0–2]
- *Near-misses:* Lines approach within a few pixels but don't cross. Critical distractor.
- *Ground truth:* Computed analytically via segment-segment intersection on the underlying data points.
- *Workflow:* B
- *Implementation:* ~50 lines. Construct curves by interlacing ascending/descending segments to control exact crossing count.
- **Matched pair with T7.1** — same P4 primitive, clean vs. chart context. The accuracy gap = "clutter tax."

**T2.2 — Bar Chart Value Reading (MC)**
- *Render:* Vertical bar chart with y-axis ticks (matplotlib). NO value labels on bars.
- *Ask:* "What is the value of the bar labeled 'Q3'? (A) 35 (B) 42 (C) 48 (D) 55" → MC4
- *Primitives:* P9, P8, P1
- *Params:* n_bars [3–8], value_range [0–100], bar_width, tick_spacing
- *Distractor design:* Correct answer is exact bar height. Distractors are heights of adjacent bars ± small offsets. Randomize option order.
- *Workflow:* B
- *Implementation:* ~40 lines.

**T2.3 — Line Chart Point Value (MC)**
- *Render:* Line chart with visible data point markers at exact x positions
- *Ask:* "What is the y-value at the data point marked at x=2019? (A) 12.5 (B) 18.3 (C) 24.1 (D) 31.7" → MC4
- *Primitives:* P1, P2, P9
- *Params:* n_points [5–15], axis_scale [linear], marker_size [3–8], gridlines [on/off]
- *Important:* Only ask about data points that are explicitly marked with a dot/marker. Never ask about interpolated positions.
- *Distractor design:* Correct = exact y-value. Distractors = y-values of neighboring points or ±15% of correct value.
- *Workflow:* B
- *Implementation:* ~40 lines.

**T2.4 — Legend-Data Association**
- *Render:* 3–4 series line chart. Legend placed at configurable distance from data.
- *Ask:* "Which series has the highest peak — Revenue, Cost, Profit, or Tax?" → EXACT (series name)
- *Primitives:* P7 (cross-distance color matching), P8, P9 (relative comparison)
- *Params:* n_series [2–4], color_similarity [distinct/similar], legend_position [inset/right/bottom]
- *Difficulty:* Similar colors (e.g., blue vs. teal) + legend far from data
- *Workflow:* B
- *Implementation:* ~45 lines.

**T2.5 — Pie Chart Relative Comparison (MC)**
- *Render:* Pie chart with 4–6 labeled slices. No percentage labels.
- *Ask:* "What approximate percentage does the 'Marketing' slice represent? (A) 8% (B) 15% (C) 24% (D) 37%" → MC4
- *Primitives:* P9 (angular proportion), P7, P8
- *Params:* n_slices [3–7], min_slice_pct [5%], slice_similarity (how close slice sizes are)
- *Distractor design:* Correct = true percentage. Distractors = percentages of other slices or ±50% offset. Ensure options are well-spaced (at least 7% apart).
- *Workflow:* B
- *Implementation:* ~35 lines. Matplotlib pie chart.

**T2.6 — Stacked Bar Segment Reading (MC)**
- *Render:* Stacked bar chart with 2–3 colored segments per bar, legend
- *Ask:* "In the '2023' bar, what is the approximate value of the blue segment? (A) 12 (B) 23 (C) 35 (D) 48" → MC4
- *Primitives:* P7, P9, P1, P8
- *Ground truth:* The height difference between the top and bottom of the blue segment.
- *Distractor design:* Total bar height, heights of other segments, cumulative height up to the segment.
- *Workflow:* B
- *Implementation:* ~50 lines.

---

### Category 3: Diagrams & Flowcharts

**T3.1 — Arrow Following (Simple)**
- *Render:* 4–6 labeled boxes at fixed grid positions connected by single-segment arrows (PIL rectangles + lines with arrowheads)
- *Ask:* "Starting at box A, follow the arrow. What box do you reach?" → EXACT letter
- *Primitives:* P2, P6 (arrowhead direction), P8
- *Params:* n_boxes [3–6], arrow_width [1–5], layout [2×2 grid / 2×3 grid / 3×3 grid]
- *Ground truth:* Deterministic from the directed edge list.
- *Workflow:* C
- *Implementation:* ~50 lines. Fixed grid positions avoid layout problems.

**T3.2 — Decision Flowchart Traversal (Fixed Templates)**
- *Render:* Use 4–5 fixed flowchart templates with decision diamonds and process boxes. Vary: node labels, condition outcomes (Yes/No), and which path is correct.
- Templates:
  - Linear: Start → Process → Decision → End1/End2
    - Two-decision: Start → D1 → D2 → End1/End2/End3
      - Diamond chain: Start → D1 → Process → D2 → End1/End2
        - Loop-with-exit: Start → Process → Decision → (Yes→Process, No→End)
	- *Ask:* "If 'Amount > $500' is Yes and 'Manager Approved' is No, what is the outcome?" → EXACT (outcome label)
	- *Primitives:* P2, P6 (diamond shape recognition), P8
	- *Workflow:* C
	- *Implementation:* ~80 lines. Templates are hand-coded layouts. Labels/conditions randomized.

**T3.3 — Edge Crossing Disambiguation**
- *Render:* 4–6 nodes with edges that visually cross. At each crossing, one edge has a small bridge gap (standard engineering convention) to show it passes over.
- *Ask:* "Is node A directly connected to node D? Yes/No" → EXACT
- *Primitives:* P2, P4
- *Params:* n_nodes [4–6], n_edges [4–8], n_crossings [1–3], bridge_gap_size [3–10px]
- *Ground truth:* From the adjacency matrix.
- *Workflow:* C (network diagrams, ER diagrams)
- *Implementation:* ~60 lines. Fixed node positions, random edges. Bridge gaps rendered as white rectangles behind the "over" edge.

**T3.4 — Hierarchy Depth**
- *Render:* Tree/org chart with labeled nodes (PIL rectangles + vertical/horizontal connecting lines)
- *Ask:* "How many levels deep is this hierarchy?" → EXACT integer. Or: "Who reports to [name]?" → EXACT name.
- *Primitives:* P3, P2, P1
- *Params:* depth [2–5], branching_factor [2–3], node_width, edge_width
- *Workflow:* C (org charts, file trees)
- *Implementation:* ~50 lines. Top-down tree layout with even horizontal spacing per level.

---

### Category 4: Forms & UI Elements

**T4.1 — Checkbox State Detection**
- *Render:* Vertical list of labeled items with square boxes (15–25px). Checked boxes contain a ✓ rendered as two short lines. Unchecked boxes are empty.
- *Ask:* "Which options are checked? List all that apply." → SET (e.g., "A,C,D")
- *Primitives:* P5, P8
- *Params:* n_options [4–8], n_checked [1–4], box_size [10–25px], font_size [10–16], spacing [tight/normal]
- *Difficulty:* Small boxes + many options + tight spacing
- *Workflow:* D (forms processing — multi-billion dollar market)
- *Implementation:* ~40 lines.
- **Matched pair with T7.2** — checkbox is the business version of "fine state discrimination" (touching circles). The accuracy gap = practical impact of the primitive failure.

**T4.2 — Radio Button Selection**
- *Render:* Groups of 3–4 radio buttons (circles). One per group has a filled inner circle. Groups are labeled.
- *Ask:* "In the 'Payment Method' group, which option is selected?" → EXACT (option label)
- *Primitives:* P5, P8, P1 (group association)
- *Params:* n_groups [1–3], options_per_group [3–4], circle_size [8–20px], fill_ratio [0.4–0.7]
- *Workflow:* D
- *Implementation:* ~40 lines.

**T4.3 — Form Field Extraction**
- *Render:* Vertical form layout with "Label: [value]" pairs. Values inside light-bordered boxes or on underlines.
- *Ask:* "What is the value in the 'Company Name' field?" → EXACT string
- *Primitives:* P8, P1 (label-to-value association)
- *Params:* n_fields [5–12], font_size [10–16], field_style [boxed/underlined], label_position [left/above]
- *Workflow:* D (invoice processing, intake forms)
- *Implementation:* ~45 lines.

**T4.4 — Progress Bar Reading (MC)**
- *Render:* 3–4 horizontal progress bars at different fill levels, each labeled
- *Ask:* "What percentage is the 'Upload' progress bar at? (A) 25% (B) 45% (C) 65% (D) 85%" → MC4
- *Primitives:* P9
- *Params:* n_bars [2–4], fill_levels [10–90%], bar_height [15–30px], show_percentage [no]
- *Distractor design:* Fill levels of other bars in the same image. Ensure ≥15% spacing between options.
- *Workflow:* F
- *Implementation:* ~30 lines. Easiest test to render. Good sanity check.

---

### Category 5: Annotations & Markup

**T5.1 — Circled Text Identification**
- *Render:* Rendered word/sentence with a red ellipse drawn around one letter or word
- *Ask:* "Which letter/word is circled?" → EXACT
- *Primitives:* P1, P8
- *Params:* word_list, target_position, ellipse_thickness [1–4], font_size [14–36], gap_ratio [tight/loose]
- *Workflow:* E
- *Implementation:* ~40 lines. Adaptation of BlindTest Task 3.

**T5.2 — Arrow Annotation Target**
- *Render:* 3–5 words/labels on a canvas. One arrow points from the margin to a specific word.
- *Ask:* "What word does the red arrow point to?" → EXACT
- *Primitives:* P2, P6, P1, P8
- *Params:* n_words [3–6], arrow_length, arrow_width [1–4], word_spacing
- *Workflow:* E (annotated screenshots, tutorial markup)
- *Implementation:* ~35 lines.

**T5.3 — Strikethrough Detection**
- *Render:* 5–8 words in a row. 1–3 have a horizontal line drawn through their center.
- *Ask:* "Which words are struck through? List all." → SET
- *Primitives:* P5, P8
- *Params:* n_words [5–8], n_struck [1–3], line_thickness [1–3px], font_size [12–24]
- *Difficulty:* Thin strikethrough on small text
- *Workflow:* E (track changes, redlining)
- *Implementation:* ~30 lines.

---

### Category 6: Text & OCR Robustness

**NEW — tests isolated from prior taxonomy. Addresses the most common real-world VLM failure.**

**T6.1 — Text Readability Under Degradation**
- *Render:* A short text string (e.g., "Total: $42,387.19" or "Invoice #INV-2024-0892") rendered at varying quality levels
- *Ask:* "What does the text say?" → EXACT string
- *Primitives:* P8
- *Params:* font_size [8–28], blur_radius [0–3px], rotation [0°, 2°, 5°], contrast [1.0 → 0.3], noise_level [0–20%], font [DejaVuSans/DejaVuSerif/FreeMono]
- *Difficulty:* Small font + blur + low contrast = simulates scanned PDFs, fax-quality images
- *Workflow:* A, D (document OCR — the single most common failure in production)
- *Implementation:* ~35 lines. Render text with PIL, apply Gaussian blur and contrast adjustment via ImageFilter/ImageEnhance.

**T6.2 — Rotated Text Reading**
- *Render:* Text rendered at specified rotation angles, simulating chart axis labels
- *Ask:* "What does the rotated text say?" → EXACT string
- *Primitives:* P8
- *Params:* rotation [0°, 15°, 30°, 45°, 60°, 90°], font_size [10–24], text_length [1 word / 2–3 words]
- *Difficulty:* 45° and 90° rotations on small text
- *Workflow:* B (chart axis labels), A (rotated table headers)
- *Implementation:* ~20 lines. PIL Image.rotate().

**T6.3 — Dense Small Text Extraction**
- *Render:* A block of 5–10 lines of small text (simulating a footnote, legal disclaimer, or dense table)
- *Ask:* "What is the third line of text?" → EXACT string
- *Primitives:* P8, P1, P3 (line counting)
- *Params:* font_size [7–14], line_spacing [1.0–1.5], n_lines [5–10], resolution [256–768]
- *Workflow:* A, D (fine print, footnotes, dense documents)
- *Implementation:* ~25 lines.

---

### Category 7: Geometric Primitives (Pure Diagnostic)

These strip away all business context to test raw perceptual capability. Each has a **matched business-context counterpart** to measure the "clutter tax."

**T7.1 — Line Intersection Count** (BlindTest Task 1)
- *Render:* Two colored piecewise-linear paths on white background
- *Ask:* "How many times do the blue and red lines intersect?" → EXACT integer
- *Primitives:* P4
- *Params:* linewidth [1–8], resolution [384–1152], n_intersections [0–2]
- **→ Matched with T2.1** (same P4 in chart context)

**T7.2 — Touching/Overlapping Circles** (BlindTest Task 2)
- *Render:* Two colored circles at parameterized distance
- *Ask:* "Are the two circles touching or overlapping? Yes/No" → EXACT
- *Primitives:* P5
- *Params:* distance [-0.25 to +0.25 in 0.05 steps], resolution [384–1152], rotation [horizontal/vertical/diagonal]
- *Key output:* Sigmoid accuracy curve as a function of distance → reveals discrimination threshold
- **→ Matched with T4.1** (checkbox = business version of fine state discrimination)

**T7.3 — Nested Shape Count** (BlindTest Task 5)
- *Render:* Recursively nested squares with random center offsets
- *Ask:* "How many squares are in the image?" → EXACT integer
- *Primitives:* P3, P1
- *Params:* depth [2–5], line_thickness [1–5], resolution [384–1152]

**T7.4 — Colored Path Count** (BlindTest Task 7)
- *Render:* Multiple colored paths between labeled stations on a grid
- *Ask:* "How many paths go from station A to station B?" → EXACT integer
- *Primitives:* P2, P7
- *Params:* n_paths [1–3], thickness [3–20], resolution [512–1024]
- **→ Matched with T3.1** (arrow following = business version of path tracing)

---

### Matched Pairs for "Clutter Tax" Analysis

| Clean Version (Cat 7) | Business Version (Cat 1–6) | Shared Primitive | What the gap tells you |
|---|---|---|---|
| T7.1 Line Intersection | T2.1 Line Chart Crossing | P4 | How much gridlines/axes/legend degrade intersection detection |
| T7.2 Touching Circles | T4.1 Checkbox Detection | P5 | How much label context affects fine state discrimination |
| T7.4 Colored Paths | T3.1 Arrow Following | P2 | How much box labels/layout affect path following |
| — (new: plain text at size X) | T6.1 Text Degradation | P8 | How much blur/rotation/contrast degrade OCR |

---

## Part 4: Coverage Matrix

```
               P1    P2    P3    P4    P5    P6    P7    P8    P9
	                      SpatR Path  Count Xsct  Fine  Sym   Color Text  Scale
			      ──────────────────────────────────────────────────────────────────────
			      T1.1 Grid       ·     ·     ●     ·     ·     ·     ·     ·     ·
			      T1.2 CellNum    ●     ·     ●     ·     ·     ·     ·     ●     ·
			      T1.3 CellReal   ●     ·     ·     ·     ·     ·     ·     ●     ·
			      T1.4 Merged     ●     ·     ●     ·     ·     ·     ·     ●     ·
			      T2.1 LineCross  ·     ●     ·     ●     ·     ·     ●     ·     ·
			      T2.2 BarVal     ●     ·     ·     ·     ·     ·     ·     ●     ●
			      T2.3 PointVal   ●     ·     ·     ·     ·     ·     ·     ·     ●
			      T2.4 Legend     ·     ·     ·     ·     ·     ·     ●     ●     ●
			      T2.5 Pie        ·     ·     ·     ·     ·     ·     ●     ●     ●
			      T2.6 StackBar   ●     ·     ·     ·     ·     ·     ●     ●     ●
			      T3.1 Arrow      ·     ●     ·     ·     ·     ●     ·     ●     ·
			      T3.2 Decision   ·     ●     ·     ·     ·     ●     ·     ●     ·
			      T3.3 EdgeCross  ·     ●     ·     ●     ·     ·     ·     ·     ·
			      T3.4 Hierarchy  ·     ●     ●     ·     ·     ·     ·     ●     ·
			      T4.1 Checkbox   ·     ·     ·     ·     ●     ·     ·     ●     ·
			      T4.2 Radio      ·     ·     ·     ·     ●     ·     ·     ●     ·
			      T4.3 FormField  ●     ·     ·     ·     ·     ·     ·     ●     ·
			      T4.4 Progress   ·     ·     ·     ·     ·     ·     ·     ·     ●
			      T5.1 Circled    ●     ·     ·     ·     ·     ·     ·     ●     ·
			      T5.2 ArrowAnno  ●     ●     ·     ·     ·     ●     ·     ●     ·
			      T5.3 Strike     ·     ·     ·     ·     ●     ·     ·     ●     ·
			      T6.1 TextDeg    ·     ·     ·     ·     ·     ·     ·     ●     ·
			      T6.2 RotText    ·     ·     ·     ·     ·     ·     ·     ●     ·
			      T6.3 DenseText  ●     ·     ●     ·     ·     ·     ·     ●     ·
			      T7.1 LineXsct   ·     ·     ·     ●     ·     ·     ·     ·     ·
			      T7.2 TouchCirc  ·     ·     ·     ·     ●     ·     ·     ·     ·
			      T7.3 NestSqr    ·     ·     ●     ·     ·     ·     ·     ·     ·
			      T7.4 ColorPath  ·     ●     ·     ·     ·     ·     ●     ·     ·
			      ```

**Coverage:** Every primitive tested by ≥3 tests. P8 appears in 20/28 tests — realistic since most business content mixes text with visual structure.

---

## Part 5: Difficulty Parameters

| Parameter | Range | What it degrades | Key tests |
|-----------|-------|-----------------|-----------|
| **Resolution** | 256–1536px | Everything | All |
| **Line/border width** | 1–10px | Grid visibility, connector clarity | T1.*, T2.1, T3.*, T7.* |
| **Font size** | 7–36px | Text readability | T1.2–3, T4.*, T6.* |
| **Element count** | 2–15 | Counting, visual crowding | T1.1, T3.4, T4.1 |
| **Gap/distance** | -0.25 to 0.5 | Fine discrimination threshold | T7.2, T4.1, T4.2 |
| **Blur radius** | 0–3px | OCR degradation | T6.1 |
| **Rotation** | 0–90° | Orientation robustness | T6.2 |
| **Contrast** | 1.0 → 0.3 | Visibility of all elements | T6.1, T1.* |
| **Color similarity** | distinct → similar | Color discrimination | T2.4, T7.4 |
| **Clutter level** | clean / gridlines / full context | Figure-ground separation | T7.1→T2.1 pairs |
| **Element size** | 5–30px | Fine feature detection | T4.1, T4.2 |

**The key diagnostic:** For each test, find the parameter value where accuracy drops from >80% to <50%. This **failure boundary** translates directly to business requirements: "Haiku 4.5 cannot reliably read table cells when font is below Xpx at resolution Y."

---

## Part 6: Perception vs. Reasoning Diagnostic

For tests where the question requires **computation** (not just reading), run a text-only control:

| Test | Text-only control | What it tells you |
|---|---|---|
| T7.1 / T2.1 | Provide line coordinates as point lists, ask for intersection count | If text >> image → perceptual failure. Model can compute intersections but can't see them. |
| T7.3 | Describe nested square coordinates, ask for count | If text ≈ image → reasoning failure (can't count nested structures even from description) |
| T3.2 | Provide flowchart as text adjacency list with conditions, ask for traversal outcome | Isolates path-following reasoning from visual path-following |
| T1.1 | "A grid has lines at x=[0, 10, 20, 30] and y=[0, 15, 30, 45, 60]. How many rows and columns?" | Tests whether counting failure is perceptual or numerical |

**Not applicable for:** T1.2, T4.1, T4.3, T6.1 — for pure reading/discrimination tasks, the answer IS the visual content. Failure on these is definitionally perceptual.

---

## Part 7: Multiple Choice Design Guide

For MC4 tests (T2.2, T2.3, T2.5, T2.6, T4.4), distractor quality determines test validity.

### Distractor Principles

1. **Plausible nearby values.** If correct bar height is 42, use distractors like 35, 48, 55 — not 3, 99, 1000.
2. **Common confusion errors.** Include values of adjacent/nearby visual elements (the bar next to the target, the segment above the target in a stacked bar).
3. **Sufficient spacing.** Options must be at least 15% of the value range apart. If options are too close, even humans can't reliably distinguish them from the visual alone.
4. **Randomized order.** Shuffle A/B/C/D on every generated sample. Never put the correct answer in the same position.
5. **No "none of the above."** Always include the correct answer as one of the four options.
6. **Deterministic distractor generation.** Distractors are computed from the ground truth and other elements in the image, not randomly sampled. This makes the test reproducible.

### MC4 Prompt Format
```
"What is the value of the bar labeled 'Q3'?
(A) 35
(B) 48
(C) 42
(D) 55
Answer with only the letter."
```

Parser: extract single letter A/B/C/D. Scorer: exact match against shuffled correct option.

---

## Part 8: Primitive → Business Impact Mapping

| Primitive Failure | Business Impact | Example Consequence | Test(s) |
|---|---|---|---|
| P1 Spatial Reference fails | Wrong cell read from table | Financial report has wrong numbers | T1.2, T1.3 |
| P2 Path Following fails | Flowchart gives wrong outcome | Process compliance error | T3.1, T3.2 |
| P3 Counting fails | Wrong table dimensions | Table parsed incorrectly | T1.1, T7.3 |
| P4 Intersection fails | Trend crossings missed | Bad investment decision | T2.1, T7.1 |
| P5 Fine Discrimination fails | Checkbox/radio misread | Contract terms wrong | T4.1, T4.2 |
| P6 Symbol Recognition fails | Arrow direction misread | Flowchart traversal error | T3.1, T3.2 |
| P7 Color Mapping fails | Legend-data mismatch | Revenue confused with Cost | T2.4, T2.6 |
| P8 Text Reading fails | OCR errors on small/degraded text | Data entry errors at scale | T6.1, T6.2, T6.3 |
| P9 Scale Reading fails | Wrong value from chart | Dashboard shows wrong KPIs | T2.2, T2.3 |

---

## Part 9: Implementation Priorities

### Tier 1: First 4 hours — get end-to-end signal
1. **T1.1** Grid Counting — simplest possible test, validates entire pipeline
2. **T1.2** Cell Value Lookup — most important business test
3. **T7.1** Line Intersection — validate against BlindTest results
4. **T7.2** Touching Circles — validate against BlindTest results
5. **T4.1** Checkbox Detection — high business value, tests P5

### Tier 2: Hours 4–8 — chart + diagram coverage
6. **T2.1** Line Chart Crossing — chart version of T7.1 (measure clutter tax)
7. **T6.1** Text Degradation — OCR robustness (most common real-world failure)
8. **T3.1** Arrow Following — path following in business context
9. **T2.2** Bar Chart Value (MC) — scale reading
10. **T6.2** Rotated Text — common chart/table failure

### Tier 3: Hours 8–12 — depth and nuance
11. **T1.3** Realistic Table — headers + mixed data types
12. **T3.2** Decision Flowchart — fixed template traversal
13. **T2.4** Legend-Data Association — long-range color matching
14. **T5.1** Circled Text — annotation review
15. **T4.3** Form Field Extraction

### Tier 4: If time allows
16. **T7.3** Nested Squares, **T7.4** Colored Paths
17. **T2.5** Pie Chart, **T2.6** Stacked Bar
18. **T3.3** Edge Crossing, **T3.4** Hierarchy
19. **T5.2** Arrow Annotation, **T5.3** Strikethrough
20. **T6.3** Dense Small Text, **T4.4** Progress Bar
21. **T4.2** Radio Buttons, **T1.4** Merged Cells
22. Text-only perception vs. reasoning controls

### Total test count: 28 tests (down from 35)
- 7 dropped from v1 (trivially easy or redundant)
- 3 new OCR/text degradation tests added
- All remaining tests have unambiguous ground truth

---

## Appendix: Tests Removed from v1 and Why

| Removed Test | Reason |
|---|---|
| T1.5 Row/Column Highlighting | Trivially easy — colored bands are visually obvious |
| T5.4 Status Indicator (red/green dots) | Trivially easy — even weak VLMs identify red vs green |
| T6.3 Highlight Color (v1) | Trivially easy — colored text backgrounds are salient |
| T7.4 Olympic Circle Counting (v1) | Low business relevance — interlocking rings are benchmark-specific |
| T1.3 Cell Comparison (v1) | Redundant with T1.2 — once you read two cells, comparison is LLM reasoning |
| T4.1-T4.4 Document Layout (v1) | Requires complex page layout engine. Primitives (P1, P8) tested by simpler tasks. Replaced by T6.1–T6.3 which test the same OCR primitives more directly. |
| T3.5 Swimlane Diagram (v1) | Requires graph layout. Containment tested by T3.4 hierarchy. |

---

## Appendix: Answer Format Summary

| Format | Description | Tests | Parser |
|---|---|---|---|
| EXACT int | Single integer answer | T1.1, T1.2, T2.1, T3.4, T7.1, T7.3, T7.4 | regex `\{?(\d+)\}?` |
| EXACT str | Single string answer | T1.3, T4.3, T5.1, T6.1, T6.2, T6.3 | exact string match (case-insensitive, strip whitespace) |
| EXACT letter | Single letter A–Z | T3.1, T5.2 | regex `\b([A-Za-z])\b` |
| Yes/No | Binary | T3.3, T7.2 | starts with "yes"/"no" |
| MC4 | Multiple choice, answer is letter A–D | T2.2, T2.3, T2.5, T2.6, T4.4 | regex for A/B/C/D |
| SET | Unordered set of items | T4.1, T4.2, T5.3 | comma-separated, sorted, set equality |
| row_col | "rows=N columns=M" | T1.1 | regex for rows/columns pattern |~