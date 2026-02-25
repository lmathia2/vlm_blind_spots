# Detailed Failure Analysis: Top 4 Worst-Performing Tasks

**Model**: Claude Haiku 4.5 (claude-haiku-4-5-20251001)
**Reasoning**: Extended thinking enabled
**Samples**: 75 per task across full parameter sweeps

This report deep-dives into the four tasks where Haiku 4.5 struggles most, analyzing accuracy by sweep parameter, error distributions, and reasoning text patterns. Each section includes concrete failure examples with the actual images the model was shown.

---

## Overview

| Rank | Task | Accuracy | Dominant Error | Root Cause |
|------|------|----------|---------------|------------|
| 1 | counting_grid | 9.3% (7/75) | Mixed over/under | Grid size beyond perceptual limit |
| 2 | colored_paths | 45.9% (34/74) | Overcounting (92.5%) | Confuses visual paths with graph routes |
| 3 | nested_squares | 57.3% (43/75) | Undercounting (96.9%) | Perceptual ceiling at ~5 squares |
| 4 | text_degradation | 58.7% (44/75) | Hallucination (35%) | Blur + small font destroys OCR |

![Failure Examples Panel](figures/fig_failure_examples.png)
*Figure: Representative failures from all 4 tasks. Each panel shows the actual image presented to the model, with ground truth and model answer below. Annotations describe the failure mode.*

---

## 1. Counting Grid (9.3%)

The model must count rows, columns, total cells, or merged regions in a grid image.

### Accuracy by Sweep Parameter

**By question type:**

| Question Type | Accuracy | Description |
|---|---|---|
| grid_size (count rows, cols) | 2/24 (8.3%) | Report grid dimensions as "rows,cols" |
| total_cells | 1/30 (3.3%) | Count all cells (accounting for merges) |
| merged_count | 4/21 (19.0%) | Count merged regions |

**By grid dimensions:**

| Rows | Accuracy | | Cols | Accuracy |
|------|----------|-|------|----------|
| 4 | 18.2% | | 4 | 13.3% |
| 8 | 7.7% | | 8 | 21.4% |
| 12 | 7.1% | | 12 | 0.0% |
| 18 | 0.0% | | 18 | 6.7% |
| 25 | 6.2% | | 25 | 7.1% |

**By number of merged cells:**

| n_merged | Accuracy |
|---|---|
| 0 | 25.0% |
| 3 | 5.0% |
| 6 | 0.0% |

### What the 7 Successes Have in Common

Every correct answer shares at least one property:
- **Small grid**: 4x8, 8x8, or 4x4 (both dimensions at most 8)
- **No merged cells**: 6/7 successes had n_merged=0

No grid with more than 8 rows AND more than 8 columns was ever answered correctly.

### Error Patterns

**Grid size errors** scale with grid area (r=0.804 correlation between total cells and absolute error). **Total cells** show a strong undercounting bias (-15.7 cells mean error). **Merged regions** are hallucinated in 4/7 grids with zero merged cells (57% false positive rate).

Hedging language in failures: "approximately" (76%), "let me recount" (56%), "appears to be" (84%). Incorrect answers had longer thinking (2284 chars) vs correct (1889 chars) -- more deliberation did not help.

### Failure Examples

**Example 1: 25x25 grid miscounted as 16x30** (top-left in figure)

The model is shown a clean 25x25 grid with no merged cells (`7037223c.png`). It answers **16, 30** -- undercounting rows by 9 and overcounting columns by 5. From its thinking:

> "Let me count the rows and columns in this grid carefully. Looking at the image, I need to count the number of horizontal lines and vertical lines to determine the grid dimensions. Counting the vertical lines... Let me count them carefully..."

The sequential counting strategy breaks down for dense grids. The model loses track around line 16-19 and produces a confident but wrong answer.

**Example 2: Zero merges, model hallucinates 8** (top-center in figure)

A perfect 4x8 grid with no merged cells (`d640ef44.png`). The model answers **8 merges** when the correct answer is **0**:

> "Let me analyze this grid carefully to identify merged regions. I need to identify areas where internal borders have been removed, creating merged regions..."

The model interprets normal grid line rendering as evidence of removed internal borders, fabricating 8 merged regions in a completely regular grid.

### Key Insight

Grid counting is an almost total perceptual failure (9% image vs 96% text). The model understands the concept of grids perfectly but cannot reliably enumerate parallel lines in dense images. This is the clearest case where tool-use delegation (e.g., a code interpreter with pixel analysis) would be the appropriate fix.

---

## 2. Colored Paths (45.9%)

The model must count how many colored ribbons connect station A to station C in a transit-map-style diagram. Some paths connect other station pairs (distractors).

### Accuracy by Sweep Parameter

**By distractor count (dominant factor):**

| Distractors | Accuracy |
|---|---|
| 0 (all paths connect A-C) | 64.9% |
| 1 | 47.1% |
| 2 | 18.2% |
| 3 | **0.0%** |

**By total paths in image:**

| n_paths | Accuracy |
|---|---|
| 1 | 83.3% |
| 2 | 50.0% |
| 3 | 36.4% |
| 4 | 27.8% |

**By thickness:**

| Thickness | Accuracy | MAE |
|---|---|---|
| 3 | 43.8% | 1.00 |
| 6 | 31.8% | 1.14 |
| 10 | 50.0% | 0.86 |
| 20 | 59.1% | 0.50 |

**Resolution does NOT help** (54.2% at 384px vs 41.7% at 768px).

### Error Distribution

**92.5% of errors are overcounting.** The model almost never misses paths -- it invents extra ones.

30% of errors have `parsed_answer == n_paths` (total paths in image), meaning the model counted ALL paths instead of just A-C connecting ones.

### Confusion Matrix

```
          ans=1  ans=2  ans=3  ans=4  ans=5  ans=6
gt=1       12     10      5      4      2      .
gt=2        .     14     11      3      .      .
gt=3        .      3      8      1      .      1
```

When GT=1, the model answers "2" ten times (30%). The +1 overcounting pattern dominates.

### Failure Example

**Counted all 4 paths instead of the 1 connecting A-C** (top-right in figure)

Image `23f31622.png` shows 4 colored paths between 5 stations, but only 1 actually connects station A to station C. The model answers **4** instead of **1**:

> "Let me analyze this graph carefully to find all paths from station A to station C. First, let me identify the stations and connections: Station A (right side), Station B (top), Station C (top left), Station D (bottom left), Station E (bottom center)..."

The model identifies all 4 colored paths in the image and reports the total count, ignoring which stations each path actually connects. This is the single most common error pattern: confusing "number of paths in the image" with "number of paths connecting the queried stations."

In other failures, the model goes further and hallucinates graph-theoretic routes through intermediate stations: *"I get three distinct paths: the direct route C->A, the two-hop C->D->A, and the three-hop C->D->E->A"* -- inventing multi-hop walks instead of counting drawn colored ribbons.

### Key Insight

This is a mixed perceptual-conceptual failure. The model can see the paths, but conflates visual ribbons with graph routes. Higher resolution doesn't help because the bottleneck is semantic, not visual: the model doesn't maintain path identity through crossing points and defaults to counting all visible paths regardless of endpoints.

---

## 3. Nested Squares (57.3%)

The model must count concentric nested squares in an image.

### Accuracy by Sweep Parameter

**By depth (number of squares) -- the dominant factor:**

| Depth | Accuracy |
|---|---|
| 2 | 100% |
| 3 | 100% |
| 4 | 100% |
| 5 | 50% |
| 6 | 11.1% |
| 7 | 0.0% |
| 8 | 14.3% |

A **sharp threshold between 4 and 5 squares**. Depths 2-4 are perfect; depths 6-8 are near-total failure.

**By reduction factor:**

| RF | Accuracy |
|----|----------|
| 0.4 | 43.8% |
| 0.5 | 53.3% |
| 0.6 | 62.5% |
| 0.7 | 68.8% |
| 0.8 | 58.3% |

**Line thickness has minimal effect** (52-62% across 1-3px).

### Error Distribution

**96.9% of errors (31/32) are undercounting.** Only 1 overcounting case.

Mean error: **-1.31** (systematic undercounting). The model consistently misses the innermost squares.

### Model Answer Distribution for GT >= 5

```
When GT=5: {4: 9, 5: 9}
When GT=6: {5: 8, 6: 1}
When GT=7: {5: 8, 6: 1}
When GT=8: {6: 4, 7: 1, 8: 1, 9: 1}
```

The model overwhelmingly answers **"5"** for any image with 5+ squares. It has a perceptual ceiling of approximately 5 distinguishable nested levels.

### Failure Examples

**Example 1: 7 squares with rf=0.8, model sees only 5** (middle-left in figure)

Image `4d26c154.png` shows 7 concentric squares with reduction_factor=0.8, making all inner squares clearly visible. The model answers **5**:

> "1. The outermost square (the largest one) 2. The second square (inside the first) 3. The third square (inside the second) 4. The fourth square... Let me verify by counting..."

With rf=0.8, each inner square is 80% the size of the previous, so even the innermost square is large. Yet the model caps out at 5 -- the labels "large", "medium", "smaller", "even smaller", "smallest" provide only ~5 distinct gradations in its spatial vocabulary.

**Example 2: 6 squares with 168px innermost, still wrong** (middle-center in figure)

Image `70bab5bc.png` has depth=6 with rf=0.8 and line_thickness=2. The innermost square is **168 pixels** across -- easily visible. The model still answers **5**:

> "1. The outermost large square 2. The next square inside it 3. The next square inside that 4. The next square inside that 5. The innermost small square. So I can see 5 individual squares nested within each other..."

This rules out pixel resolution as the cause. The model can see all 6 squares but cannot enumerate beyond 5. The limitation is in spatial segmentation, not visual acuity.

### Key Insight

The model has a hard perceptual ceiling at ~5 nested objects. This is not a reasoning failure (text control is 100%) and not a resolution issue (168px inner squares are easily visible). It appears to be a fundamental limit of the vision encoder's ability to segment tightly-packed concentric shapes.

---

## 4. Text Degradation (58.7%)

The model must read text rendered with varying font size, blur, rotation, and contrast.

### Accuracy by Sweep Parameter

**By font size (strongest mitigator):**

| Font Size | Accuracy |
|-----------|----------|
| 8 | 23.5% |
| 12 | 50.0% |
| 16 | 69.2% |
| 22 | 73.3% |
| 28 | 81.2% |

**By blur radius (most destructive factor):**

| Blur | Accuracy |
|------|----------|
| 0 | 86.4% |
| 1 | 78.9% |
| 2 | 42.1% |
| 3 | 13.3% |

**Rotation has minimal effect** (53-67%). **Contrast has moderate effect** (54-65%).

### Blur x Font Size Interaction

```
            font=8    font=12   font=16   font=22   font=28
blur=0      57%       100%      100%      100%      100%
blur=1       0%        75%       80%      100%      100%
blur=2       0%         0%       50%      100%       50%
blur=3       0%         0%        0%        0%       67%
```

**Blur >= 2 with font <= 12 is 0% accuracy.** Even font=22 drops to 0% at blur=3. Only font=28 partially survives heavy blur.

### Compound Difficulty

| Active Degradation Factors | Accuracy |
|---|---|
| 0 | 100% |
| 1 | 75.0% |
| 2 | 44.4% |
| 3 | 16.7% |

Degradation compounds non-linearly -- three simultaneous factors reduce accuracy by 83 percentage points.

### Failure Mode Taxonomy (31 failures)

| Category | Count (%) | Description |
|---|---|---|
| Complete hallucination | 11 (35%) | Fabricates entirely unrelated text |
| Minor character errors | 14 (45%) | Close but with digit/letter swaps |
| Colon dropped | 4 (13%) | Perfect match except missing ":" |
| Refusal | 2 (6%) | Model says it cannot read the text |

### Failure Examples

**Example 1: "Contract #C-2024-00341" becomes "Cannot all sour cream"** (middle-right in figure)

Image `423c81c1.png` renders "Contract #C-2024-00341" at font=8 with blur=2. The text is small and blurry but structurally present. The model confidently outputs **"Cannot all sour cream"**:

> "The text appears to be quite small and faint, but it looks like it says 'Cannot all sour cream' or possibly 'Cannot all sour cream' -- actually, looking more carefully, it might say something else. Let me try again. It appears..."

Rather than admitting uncertainty, the model pattern-matches the degraded pixels to familiar web/UI phrases from its training distribution. A financial contract number becomes nonsensical food text. This hallucination pattern occurs in 35% of failures and is the most dangerous mode because the output looks plausible.

Other hallucinations from the same conditions include "Add to cart" (for "Rate: $12.75/hr"), "PLEASE ADD YOUR TEXT HERE" (for another invoice number), and "Explore what's inside today" (for a tax line item).

**Example 2: "$1,256.00" becomes "$1,258.00"** (bottom-left in figure)

Image `7d2ec1cd.png` shows "Balance Due: $1,256.00" at font=16, blur=1, contrast=0.3. The model reads it as **"Balance Due: $1,258.00"** -- every character correct except the digit 6, misread as 8:

> "Looking at the image, I can see text that reads: 'Balance Due: $1,258.00'."

No hesitation or self-correction. At low contrast with slight rotation, the open bottom of a `6` becomes ambiguous and is perceived as the closed loops of an `8`. This is the most common digit confusion pattern (9->0 appears 3 times, 6->8 and 5->9 also recur). In financial contexts, such single-digit errors propagate silently.

### Key Insight

Text degradation has two failure modes with very different risk profiles. Complete hallucination (35%) produces obviously wrong output that a downstream system could flag. But minor character errors (45%), especially digit confusion in financial values, are subtle and dangerous -- a $2 error from 6->8 confusion passes cursory review. The interaction between blur and font size is the key predictor: blur >= 2 with font <= 12 is guaranteed failure.

---

## Cross-Cutting Patterns

### 1. The Resolution Paradox

Two of four tasks (colored_paths, nested_squares) show that **higher resolution does not improve accuracy**. This means the bottleneck is not pixel-level perception but higher-level visual reasoning -- the model can see the elements but cannot correctly process their relationships.

### 2. Systematic Counting Failures

Three tasks involve counting (grid cells, paths, nested squares). All show systematic biases:
- **counting_grid**: error scales with grid area (r=0.804)
- **colored_paths**: overcounts by conflating visual objects with graph paths
- **nested_squares**: caps at ~5 perceived objects

The model appears to have a fundamental limit on visual enumeration, especially for overlapping or densely packed objects.

### 3. Confident Fabrication

Both text_degradation and counting_grid show the model making **confident errors** rather than expressing uncertainty. In text_degradation, it hallucinates "Cannot all sour cream" for degraded financial text. In counting_grid, it confidently reports grid dimensions that are off by 50%. Extended thinking makes this worse -- longer reasoning leads to more elaborate but still wrong answers.

### 4. Conceptual vs Perceptual Failures

The four tasks split into two categories:

**Primarily perceptual** (the model can't extract the visual information):
- counting_grid: can't count dense grid lines
- nested_squares: can't segment beyond 5 nesting levels
- text_degradation: can't read blurred/small text

**Primarily conceptual** (the model extracts visual info but reasons about it incorrectly):
- colored_paths: sees the paths but confuses visual ribbons with graph routes

### 5. Parameter Sensitivity Summary

| Task | Dominant Parameter | Effect Size | Irrelevant Parameters |
|------|-------------------|-------------|----------------------|
| counting_grid | Grid size (rows x cols) | 21% -> 0% (small->large) | -- |
| colored_paths | Distractor count | 65% -> 0% (0->3) | Resolution |
| nested_squares | Depth | 100% -> 0% (4->7) | Line thickness |
| text_degradation | Blur radius | 86% -> 13% (0->3) | Rotation |
