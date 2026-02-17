# Training Data Generation

Generate SFT training data and verify RL reward functions for grid counting.

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

## Seed Ranges

| Split | Range        | Notes                          |
|-------|--------------|--------------------------------|
| SFT   | [0, 50K)     | direct [0,20K), intermediate_repr [20K,40K), tool_use [40K,50K) |
| RL    | [100K, 500K) | On-the-fly generation          |
| Eval  | [500K, 510K) | Held-out                       |
| Verify| [999K+)      | CLI verify commands            |

## Strategies

### Direct counting (grid 3–12)

Count lines, subtract 1. ~20% of samples include a self-correction pattern where the model catches its own off-by-one.

**Standard example** (9×9 grid):

```
To determine the grid dimensions, I'll count lines and apply the subtraction rule.

Horizontal lines: the top edge is line 1, then there are 8 internal dividers,
plus the bottom edge — 10 lines total. Each pair of adjacent horizontal lines
forms one row, so 10 lines → 9 rows.

Vertical lines: left edge + 8 dividers + right edge = 10 lines. Similarly,
10 lines → 9 columns.

rows=9 columns=9
```

**With self-correction** (12×7 grid):

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

### Intermediate representation (grid 3–15)

The model externalizes its perception as a structured summary before answering. Templates include ASCII sketches, tables, and enumeration formats.

**ASCII sketch** (4×3 grid):

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

**Tabular summary** (7×12 grid):

```
Organizing my observations:

| Direction  | Border lines | Interior dividers | Total lines | Cells |
|------------|-------------|-------------------|-------------|-------|
| Horizontal | 2           | 6                 | 8           | 7     |
| Vertical   | 2           | 11                | 13          | 12    |

The 'Cells' column is total lines minus 1 in each direction.

rows=7 columns=12
```

### Tool use (grid 12–25)

The model writes line-detection code, shows its output, and interprets the result. First 200 samples use small grids (3–8) with explicit "skip" templates where the model decides code isn't needed.

**With code** (19×23 grid):

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

**Skip (no code needed)** (8×6 grid):

```
The grid is small enough to count visually.

I see 9 horizontal lines and 7 vertical lines. Applying the fence-post rule:
rows = 9 - 1 = 8, columns = 7 - 1 = 6.

rows=8 columns=6
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
        "include_self_correction": false
    }
}
```
