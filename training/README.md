# Training Data Generation & Reward Functions

Generate SFT training data, verify RL reward functions, and run reward-hacking diagnostics for grid counting.

## Quick Start

```bash
# Generate all 5K SFT samples (2K direct + 2K intermediate_repr + 1K tool_use)
python -m training.cli generate --strategy all --output training_data/

# Generate a single strategy with custom count
python -m training.cli generate --strategy direct --n 500 --output training_data/
```

## Commands

### `generate`

Produces images + JSONL for SFT training.

```bash
python -m training.cli generate --strategy {direct,intermediate_repr,tool_use,all} \
                                --output DIR \
                                [--n N]
```

- `--strategy all` uses default counts (2K/2K/1K) and ignores `--n`.
- `--n` sets sample count when a single strategy is selected.
- Output structure: `DIR/<strategy>/images/*.png` + `DIR/<strategy>/samples.jsonl`.

### `verify`

Print samples to the terminal for visual inspection. Uses seeds outside all training/eval ranges.

```bash
python -m training.cli verify --strategy direct --n 3
python -m training.cli verify --strategy all --n 2
```

### `verify-reward`

Feed samples through all three reward functions (outcome, process, tool_use) and check they return 1.0.

```bash
# Check generated JSONL
python -m training.cli verify-reward --jsonl training_data/direct/samples.jsonl --n 20

# Generate fresh samples and check (no file needed)
python -m training.cli verify-reward --n 10
```

### `diagnose`

Run reward-hacking diagnostics on RL episode results. Checks for answer distribution gaming, CoT camouflage, tool overuse, and calibration issues.

```bash
# Run diagnostics on an RL results file
python -m training.cli diagnose --results rl_results.jsonl

# Limit to first 500 episodes and save JSON report
python -m training.cli diagnose --results rl_results.jsonl --n 500 --json report.json
```

Expected JSONL format for `--results`:

```json
{
    "response": "I count 10 horizontal lines...\nrows=9 columns=7",
    "ground_truth": "9,7",
    "metadata": {"rows": 9, "cols": 7},
    "reward": 1.0
}
```

The diagnose command also accepts `chain_of_thought` + `answer` fields (combined automatically into `response`).

## Seed Ranges

| Split | Range        | Notes                          |
|-------|--------------|--------------------------------|
| SFT   | [0, 50K)     | direct [0,20K), intermediate_repr [20K,40K), tool_use [40K,50K) |
| RL    | [100K, 500K) | On-the-fly generation          |
| Eval  | [500K, 510K) | Held-out                       |
| Verify| [999K+)      | CLI verify commands            |

## Strategies

### Direct counting (grid 3-12)

Count lines, subtract 1. ~20% of samples include a self-correction pattern where the model catches its own off-by-one.

**Standard example** (9x9 grid):

```
I need to count lines, not cells — N+1 lines create N rows.

Horizontal: top border + 8 dividers + bottom border = 10 lines → 9 rows.
Vertical: left border + 8 dividers + right border = 10 lines → 9 columns.

rows=9 columns=9
```

**Rule-first variant** (5x11 grid):

```
Key rule: a grid with R rows has R+1 horizontal lines.

I count 6 horizontal lines, so rows = 6 - 1 = 5.
I count 12 vertical lines, so columns = 12 - 1 = 11.

rows=5 columns=11
```

**With self-correction** (12x7 grid):

```
To determine the grid dimensions, I'll count lines and apply the subtraction rule.

Horizontal lines: the top edge is line 1, then there are 11 internal dividers,
plus the bottom edge — 13 lines total. Each pair of adjacent horizontal lines
forms one row, so 13 lines → 12 rows.
Wait — I almost said 13 rows, but 13 lines means 13 - 1 = 12 rows. Lines and
rows are not the same thing.

Vertical lines: left edge + 6 dividers + right edge = 8 lines. Similarly,
8 lines → 7 columns.

rows=12 columns=7
```

### Intermediate representation (grid 3-15)

The model externalizes its perception as a structured summary before answering. Templates include ASCII sketches, tables, and enumeration formats.

**Structured summary** (6x10 grid):

```
Let me build a structured summary first.

Line count summary:
  Horizontal lines: 7 (top border, 5 dividers, bottom border)
  Vertical lines: 11 (left border, 9 dividers, right border)

Applying the N+1 rule:
  7 horizontal lines → 6 rows
  11 vertical lines → 10 columns

rows=6 columns=10
```

**ASCII sketch** (4x3 grid):

```
I'll sketch the grid structure to make the count explicit.

+---+---+---+
|   |   |   |
+---+---+---+
|   |   |   |
+---+---+---+
|   |   |   |
+---+---+---+
|   |   |   |
+---+---+---+

From the sketch:
  Horizontal lines = 5 → rows = 4
  Vertical lines = 4 → columns = 3

rows=4 columns=3
```

**Tabular summary** (7x12 grid):

```
Organizing my observations:

| Direction  | Border lines | Interior dividers | Total lines | Cells |
|------------|-------------|-------------------|-------------|-------|
| Horizontal | 2           | 6                 | 8           | 7     |
| Vertical   | 2           | 11                | 13          | 12    |

The 'Cells' column is total lines minus 1 in each direction.

rows=7 columns=12
```

**Enumerate then derive** (8x5 grid):

```
Step 1: Enumerate what I see.
  - 9 horizontal lines running across the grid
  - 6 vertical lines running down the grid

Step 2: Derive dimensions.
  Lines always exceed the cell count by 1 (the fence-post principle).
  Rows = 9 - 1 = 8
  Columns = 6 - 1 = 5

rows=8 columns=5
```

### Tool use (grid 12-25)

The model writes line-detection code, shows its output, and interprets the result. First 200 samples use small grids (3-8) with explicit "skip" templates where the model decides code isn't needed.

**With code — gradient approach** (19x23 grid):

````
With this many cells, I should use code rather than try to count visually.

```python
import numpy as np
from PIL import Image

img = np.array(Image.open(image_path).convert('L')).astype(float)

# Use gradient to find edges
h_grad = np.abs(np.diff(img, axis=0)).mean(axis=1)
v_grad = np.abs(np.diff(img, axis=1)).mean(axis=0)

h_thresh = h_grad.max() * 0.3
v_thresh = v_grad.max() * 0.3

h_peaks = np.where(h_grad > h_thresh)[0]
v_peaks = np.where(v_grad > v_thresh)[0]

def count_lines(peaks, min_gap=5):
    if len(peaks) == 0:
        return 0
    count = 1
    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i-1] > min_gap:
            count += 1
    return count

n_h = count_lines(h_peaks)
n_v = count_lines(v_peaks)
print(f'Detected {n_h} horizontal lines, {n_v} vertical lines')
print(f'Therefore: {n_h - 1} rows, {n_v - 1} columns')
```

```output
Detected 20 horizontal lines, 24 vertical lines
Therefore: 19 rows, 23 columns
```

The gradient analysis detected 20 horizontal and 24 vertical lines.
Subtracting 1 from each gives 19 rows and 23 columns.

rows=19 columns=23
````

**With code — thresholding approach** (14x16 grid):

````
The grid is dense — visual counting would be unreliable. Let me use
a line-detection approach.

```python
import numpy as np
from PIL import Image

img = np.array(Image.open(image_path).convert('L'))
threshold = 128
binary = img < threshold

# Project onto axes
row_proj = binary.mean(axis=1)
col_proj = binary.mean(axis=0)

# Find line positions (peaks in projection)
h_positions = np.where(row_proj > 0.3)[0]
v_positions = np.where(col_proj > 0.3)[0]

# Group adjacent pixels into lines
def count_groups(positions):
    if len(positions) == 0:
        return 0
    groups = 1
    for i in range(1, len(positions)):
        if positions[i] - positions[i-1] > 3:
            groups += 1
    return groups

n_h = count_groups(h_positions)
n_v = count_groups(v_positions)
print(f'{n_h} horizontal lines, {n_v} vertical lines')
print(f'Grid: {n_h - 1} rows x {n_v - 1} columns')
```

```output
15 horizontal lines, 17 vertical lines
Grid: 14 rows x 16 columns
```

The line detection confirms: 14 rows and 16 columns.

rows=14 columns=16
````

**Skip — no code needed** (5x7 grid):

```
The grid is small enough to count visually.

I see 6 horizontal lines and 8 vertical lines. Applying the fence-post rule:
rows = 6 - 1 = 5, columns = 8 - 1 = 7.

rows=5 columns=7
```

## Reward Functions

All reward functions share the signature `(response: str, ground_truth: str, metadata: dict) -> float` where `ground_truth` is in `"R,C"` format. The answer parser extracts the **last** `rows=N columns=M` match in the response to avoid confusion with intermediate CoT values.

### `outcome_reward`

Binary exact-match. Both rows and columns must match.

```
outcome = 1.0 if parsed_answer == ground_truth else 0.0
```

### `process_reward`

Combines outcome with a check that the CoT correctly applies the N-1 subtraction rule, gated by a CoT-answer consistency check.

The process score measures what fraction of `(line_count, cell_count)` pairs in the CoT satisfy `cell_count == line_count - 1`. The consistency score checks whether the CoT's own subtraction result matches the final answer — catching "CoT camouflage" where the model writes correct-looking reasoning but gives a contradictory answer.

```
process_score = correct_subtraction_pairs / total_pairs
consistency   = 1.0 if CoT result matches final answer, else 0.0

R = max(outcome, 0.8 * outcome + 0.2 * process_score * consistency)
```

The `max` ensures correct answers (outcome=1.0) are never penalized below 1.0. If the model's CoT says "12 rows" but the answer says `rows=13`, `consistency` drops to 0.0 and the process bonus is zeroed out.

### `tool_use_reward`

Rewards correct tool use and penalizes fabricated or inconsistent tool output.

```
if no code block:
    R = outcome                           # pure outcome for visual counting

elif fabrication detected:                # code block but no ```output block
    R = outcome * 0.7                     # soft penalty

elif tool output parseable:
    if output matches ground truth:
        if answer correct:  R = 1.0       # ideal path
        if answer wrong:    R = 0.5       # misinterpreted correct output
    else:
        if answer correct:  R = 1.0       # got lucky despite bad output
        if answer wrong:    R = 0.0       # everything wrong

else:                                     # code + output but can't parse
    R = outcome
```

**Fabrication detection** flags responses where a Python code block is present but:
- There is no `` ```output `` block (model "ran" code without execution markers), OR
- The code contains no image-processing imports (`numpy`, `PIL`, `cv2`, `skimage`)

This is a soft penalty — the model still gets 70% of the outcome reward, but less than an honest tool-use path. During RL this creates pressure toward genuine tool use.

## Anti-Shortcut Randomization

To prevent the model from inferring grid size from image dimensions or pixel spacing rather than actually counting, images are post-processed after rendering with:

- **Random padding**: 0-30px per side (top, right, bottom, left sampled independently)
- **Background color variation**: sampled from white, light gray, cream, or light blue-gray
- **Aspect-ratio stretch**: width and height independently scaled by up to +/-15%

These randomizations break pixel-based shortcuts while preserving the grid structure. The renderer (`tasks/counting_grid.py`) is unchanged — all randomization happens as post-processing in `sft_generate.py`.

## Uniform Grid-Size Sampling

Grid sizes are sampled uniformly from all valid `(rows, cols)` pairs for each strategy, ensuring every grid size appears with equal probability. This prevents the model from learning the prior distribution of grid sizes and biasing toward high-frequency values.

| Strategy           | Range     | Valid pairs |
|--------------------|-----------|-------------|
| direct             | 3-12      | 100         |
| intermediate_repr  | 3-15      | 169         |
| tool_use           | 12-25     | 196         |
| tool_use_skip      | 3-8       | 36          |

## Reward Hacking Diagnostics

The `diagnose` command runs five checks on RL episode data:

| Check | What it detects | Flag threshold |
|-------|----------------|----------------|
| `check_answer_distribution` | Single answer dominating predictions | Any answer > 15% of predictions |
| `check_per_size_accuracy` | Suspiciously flat accuracy across grid sizes | Accuracy std < 0.05 across 5+ sizes |
| `check_cot_consistency_rate` | CoT arithmetic not matching final answers | Consistency rate < 70% |
| `check_tool_use_rate` | Tool overuse on easy grids | Tool-use rate > 30% for grids <= 8 |
| `check_calibration` | Overconfident wrong answers | Accuracy < 70% at confidence > 0.9 |

Example output:

```
============================================================
REWARD HACKING DIAGNOSTICS
============================================================

  1 FLAG(S) RAISED:
    - DOMINANT PREDICTION: '8,8' accounts for 18% of answers

  Answer distribution:
    Unique predictions: 41
    Most common: '8,8' (18%)
    KL divergence: 0.2527

  Per-size accuracy (43 sizes):
    Accuracy std: 0.3410

  CoT consistency:
    Consistency rate: 100%
    Subtraction pattern rate: 73%

  Tool use by grid size:
    3-8: 14% (n=37)
    9-12: 0% (n=18)
    13-18: 0% (n=0)
    19-25: 0% (n=0)

  Calibration: no confidence data available

============================================================
```

## JSONL Record Format

Each line in `samples.jsonl` contains:

```json
{
    "seed": 0,
    "strategy": "direct",
    "is_skip": false,
    "image_path": "data/training_samples/direct/images/000000.png",
    "prompt": "Count the number of rows and columns in this grid. Reply in the format: rows=N columns=M",
    "chain_of_thought": "...",
    "answer": "rows=9 columns=9",
    "ground_truth": "9,9",
    "metadata": {
        "rows": 9,
        "cols": 9,
        "resolution": 384,
        "line_width": 2,
        "strategy": "direct",
        "include_self_correction": false,
        "padding": [12, 5, 22, 18],
        "bg_color": [255, 255, 255],
        "aspect_stretch": [1.03, 0.97]
    }
}
```
