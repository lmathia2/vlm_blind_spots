# Fixing Grid Counting in Haiku 4.5

## What's actually going wrong

Grid counting scores 9% from images and 96% from text. The obvious conclusion is that the model can't see grids — but that's wrong. Look at accuracy by grid size: a 4×4 grid where each cell is 128px (enormous, easily visible) still only gets 43%. The model sees the grid fine. It reasons about it incorrectly.

The dominant bug is **lines-vs-cells confusion**. A 12-row grid has 13 horizontal lines. The model counts the lines correctly, then reports "13 rows." From the thinking traces, this accounts for most errors at resolvable grid sizes. It's an off-by-one error baked into how the model interprets what it sees.

There's a secondary issue — **resolution failure** — that kicks in around 18+ rows. Once cells are smaller than a patch, individual lines can't be resolved, and counts go haywire (25×25 → "16×30"). No reasoning fix helps here; you need a different approach entirely.

Third, the model sometimes **hallucinates merged regions** that don't exist, probably confusing aliasing artifacts with intentional merges. Less frequent but worth addressing.

## The plan

Three things, in order of priority:

**1. SFT to teach the subtraction rule.** The model needs to learn one thing: N+1 lines make N rows. Every training example should count lines explicitly, then subtract. This is the highest-ROI intervention because it targets the dominant failure mode at grid sizes the model can already see.

**2. RL to make it stick.** SFT will teach the rule, but the model will apply it inconsistently. RL (GRPO) generates multiple completions per image and reinforces the ones that get the exact answer. This turns "knows the rule" into "reliably applies the rule."

**3. Tool use for dense grids.** Above ~12 rows, teach the model to recognize it can't count visually and write a simple line-detection script instead. This removes the resolution ceiling entirely for grids where visual counting fails.

We also introduce an **intermediate representation** — having the model write out its line counts as structured text before answering. This makes the subtraction step explicit and checkable, and gives the model a chance to catch its own errors.

Note: we considered a **decomposition** approach (tiling the image into overlapping sub-regions, counting per tile, combining). Tool use subsumes its benefits with less inference-time complexity — a code interpreter can process the full image at pixel resolution, while tiling still requires a system-level preprocessor and the model still struggles with the overlap-combination step. Dropped in favor of tool use.

## Data generation pipeline

The `training/` module implements the full SFT data pipeline: grid rendering, template-based CoT generation, reward functions, and a CLI for generation and verification.

### Seed allocation

All randomness is deterministic. Seeds are partitioned into non-overlapping ranges to guarantee no data leakage between splits:

| Split | Seed range | Substrategy offsets |
|-------|-----------|---------------------|
| SFT   | [0, 100K) | Direct [0, 20K), Intermediate [20K, 40K), Tool use [40K, 50K) |
| RL    | [100K, 500K) | On-the-fly from renderer |
| Eval  | [500K, 510K) | Held-out |

Each seed deterministically controls both the grid rendering and the template/parameter selection (via `Random(seed)`), making any sample fully reproducible.

### Visual parameter randomization

Each training image varies along several visual dimensions to prevent shortcut learning:

- **Grid size**: sampled uniformly within per-strategy ranges (see below)
- **Resolution**: 384/512/768 px for grids ≤15 rows; 512/768/1024 px for denser grids
- **Line width**: 1, 2, or 3 px (random per sample)
- **Merged cells**: disabled for SFT (`n_merged=0`) to keep the counting task clean

All grids use `question_type="grid_size"` — the renderer produces a PIL image plus ground truth in `"R,C"` format.

### Strategy details

5,000 training examples, split across three strategies:

**Strategy 1: Direct counting (2,000 samples, grid sizes 3–12).** Five paraphrase variants of the same reasoning structure — count lines, apply the N−1 subtraction, report answer:

1. *Canonical* — "N+1 lines create N rows" stated upfront, then borders + dividers arithmetic
2. *Borders-first* — leads with total line count, then explains the decomposition
3. *Counting-up* — enumerates lines top-to-bottom, then subtracts
4. *Rule-first* — states the key rule before any counting
5. *Verbose* — walks through each line individually ("top edge is line 1...")

20% of direct samples inject a **self-correction pattern** after the first horizontal line count: the model writes the wrong answer, catches itself, and corrects ("Wait — I almost said 13 rows, but 13 lines means 12 rows"). This gives RL a correction template to reinforce.

**Strategy 2: Intermediate representation (2,000 samples, grid sizes 3–15).** Five variants, each forcing the model to externalize its perception as structured text before computing the answer:

1. *Structured summary* — line counts in a labeled list, then N−1 rule
2. *ASCII sketch* — renders a text grid (capped at 8×8 for readability, with `...` truncation), then reads dimensions from it
3. *Tabular* — markdown table with direction / border lines / interior dividers / total lines / cells
4. *Enumerate-then-derive* — Step 1 lists what the model sees, Step 2 applies the fence-post principle
5. *Visual breakdown* — enumerates lines top-to-bottom and left-to-right with structural labels

**Strategy 3: Tool use (1,000 samples, grid sizes 12–25).** Three code variants, each implementing a different line-detection approach:

1. *Threshold + diff* — binarize image, project dark pixels per row/column, use `np.diff` to find transitions
2. *Threshold + grouping* — same binarization, but groups adjacent dark-pixel positions with a gap threshold
3. *Gradient-based* — computes intensity gradients, peaks at line positions, groups with `min_gap`

All tool-use templates include the full code, a simulated ````output``` block with the correct counts, and a final interpretation step applying N−1.

**Tool-use skip (200 of the 1,000 tool-use samples, grid sizes 3–8).** The model explicitly recognizes the grid is small enough to count visually and skips code. Three skip-template variants. This teaches the model the decision boundary — not every grid needs a tool call. RL optimizes this boundary further.

### Template system

All templates use `str.format()` with named placeholders derived from ground truth:

| Placeholder | Derivation |
|-------------|-----------|
| `{rows}`, `{cols}` | Ground truth dimensions |
| `{h_lines}`, `{v_lines}` | `rows + 1`, `cols + 1` |
| `{h_interior}`, `{v_interior}` | `h_lines - 2`, `v_lines - 2` (dividers between borders) |
| `{ascii_sketch}` | Generated by `build_ascii_sketch()` |

No LLM-generated training data. Templates are hand-written, values are filled deterministically from ground truth.

### Prompt

Every SFT sample uses the same user prompt:

> Count the number of rows and columns in this grid. Reply in the format: rows=N columns=M

### Output format

Each sample is saved as a JSONL record with fields: `seed`, `strategy`, `is_skip`, `image_path`, `prompt`, `chain_of_thought`, `answer`, `ground_truth`, `metadata`. Images are saved as PNGs alongside the JSONL.

### CLI

```
python -m training generate --strategy {direct,intermediate_repr,tool_use,all} --output DIR
python -m training verify --strategy STRAT --n N       # print samples for visual inspection
python -m training verify-reward --jsonl FILE --n N     # check all reward fns return 1.0
```

`verify-reward` runs every generated CoT through all three reward functions (outcome, process, tool_use) and flags any sample that doesn't score 1.0 — a sanity check that the templates are well-formed and parseable.

## SFT training

The training data described above feeds into standard SFT. Key training parameters:

- Mixed batches across all three strategies
- Checkpoint every 500 steps
- Evaluate pass@1 and pass@16 on 200 held-out samples at each checkpoint

## When to stop SFT

Generate 16 completions per image on 200 held-out grids at each checkpoint. The key diagnostic is the gap between pass@1 (single-shot accuracy) and pass@16 (can the model get it right at least once in 16 tries).

If pass@16 is low, the model hasn't learned the subtraction rule yet — keep training. If pass@16 is high but pass@1 is still low, the model knows the rule but applies it inconsistently — that's exactly what RL is for. Stop SFT when pass@16 on grid sizes ≤12 hits 50%+ and stops improving.

Tagging completions by strategy (did it use the subtraction rule? intermediate representation? tool call?) reveals which approaches the model can actually execute from visual input vs. which it's just imitating from templates.

## RL details

GRPO with K=16 completions per prompt, ~15,000 episodes generated on-the-fly from the grid renderer.

**Binary reward.** Both dimensions exactly correct → 1.0, otherwise → 0.0. No partial credit. This is deliberate: the dominant error is off-by-one from the lines-vs-cells bug. If we gave partial credit for off-by-one, we'd be rewarding the exact failure mode we're trying to fix. Binary reward creates a sharp gradient at the decision point where the model either applies the subtraction or doesn't.

**Lightweight process checks (weight 0.2).** The `process_reward` function extracts all `(line_count, cell_count)` pairs from the CoT using regex pattern matching — it catches phrases like "N lines → M rows", "N lines, so M rows", "N - 1 = M", and several other variants. For each pair, it checks whether `cell_count == line_count - 1`. The process score is the fraction of correctly-applied pairs.

Combined reward: `R = max(outcome, 0.8 × outcome + 0.2 × process_score)`. This guarantees a correct final answer always scores at least 1.0, regardless of process check noise. Process checks can only hurt wrong answers and help right-for-the-right-reasons answers.

**Tool-use reward.** A separate `tool_use_reward` function handles tool-use completions. It detects Python code blocks and parses ````output``` blocks for consistency with ground truth. Scoring:

- Tool used, output correct, answer correct → 1.0
- Tool used, output correct, answer wrong (misinterpretation) → 0.5
- Tool used, output wrong, answer wrong → 0.0
- No tool used → pure outcome (appropriate for easy-grid skip examples)

This structure penalizes the specific failure mode where the model runs correct code but misreads its own output.

A note on robustness: the process checks require parsing free-form chain-of-thought, which is inherently noisy. The model might write "13 lines, meaning 12 rows" or "that gives me 13 horizontal lines (so 12 rows)" or many other variants. The implementation handles multiple regex patterns for this, but at weight 0.2, imperfect parsing adds noise without dominating the signal. If parse accuracy proves too low (<70% precision), drop process checks entirely and rely on pure outcome reward.

**Difficulty gating.** Only include grid sizes where post-SFT pass@16 exceeds 20% — below that, GRPO has no contrastive signal (0/16 correct means no positive examples to reinforce). In practice this means visual-counting RL covers sizes 3–15; sizes 16+ only appear in tool-use mode.

KL penalty against the SFT checkpoint (β = 0.05 → 0.01 linear decay over training) prevents forgetting the subtraction rule.


## Risks worth tracking

**SFT overfitting.** If pass@1 rises during training but pass@16 stalls, the model is memorizing templates rather than learning the rule. Use paraphrase variants and monitor both metrics.

**Graded reward temptation.** It's natural to want to give partial credit for near-misses. Don't. The whole point is to fix an off-by-one bug, and rewarding off-by-one is rewarding the bug.

**Tool overuse.** The model might learn to call the code interpreter for every grid, including trivial ones. SFT examples of visual counting at easy sizes plus RL's indifference to method (only accuracy matters) should prevent this, but monitor tool-use rate by grid size.

**Catastrophic forgetting.** Mix 10–15% general VQA replay data during training.

**Reward hacking — shortcut strategies.** The model could learn to exploit regularities in the synthetic grid renderer rather than actually counting. For example, if grid images have a fixed canvas size, the model can infer row count from image dimensions and cell spacing without looking at lines at all. Mitigation: randomize canvas size, padding, line thickness, cell aspect ratio, and background color across the training distribution. Validate on a separate renderer with different visual parameters to confirm the learned strategy transfers.

**Reward hacking — answer distribution gaming.** With binary reward on a finite answer space (typical grids are 3–25 rows), the model could learn the prior distribution of correct answers and bias toward high-frequency values. If training batches are not size-balanced, the model gets more reward signal from common sizes and learns to default to them. Mitigation: uniform sampling across grid sizes in both SFT and RL. Monitor per-size accuracy curves for suspiciously flat predictions (e.g., always answering "8×8").

**Reward hacking — CoT camouflage.** Process reward checks parse the chain-of-thought for the subtraction pattern. The model could learn to write "13 lines → 12 rows" in its CoT while arriving at the answer through some other (possibly wrong) internal process — essentially gaming the process checks without internalizing the rule. This is hard to detect but partly mitigated by the low weight (0.2) on process checks: the outcome reward dominates, so CoT gaming alone doesn't help unless the answer is also correct. Watch for cases where the CoT describes a clean subtraction but the final answer doesn't match the CoT's own arithmetic.

**Reward hacking — tool output fabrication.** In tool-use examples, the model could learn to "hallucinate" plausible tool outputs in its reasoning rather than actually executing code — especially if the code interpreter has latency or the model has learned that certain line-count outputs reliably score well. Mitigation: verify that tool calls actually execute (sandboxed interpreter with logged inputs/outputs), and include adversarial examples where the grid has unusual properties (e.g., double-thick borders) that make guessing tool output unreliable.

**Safety — capability generalization.** Fine-tuning on structured visual reasoning could improve the model's general ability to extract information from images in unintended ways (e.g., reading fine print, decoding obfuscated text, extracting data from screenshots of private content). This is a narrow risk given the task specificity, but worth monitoring: evaluate the fine-tuned model on out-of-distribution visual extraction benchmarks to check for unexpected capability gains.

**Safety — training data leakage into outputs.** The SFT templates are hand-written with fixed phrasing patterns. If the model memorizes and regurgitates these templates verbatim, this is mostly a quality issue, but if any template inadvertently contains identifying information or problematic phrasing, it could surface at inference time. Mitigation: audit all templates before training; strip any metadata from rendered grid images.

**Safety — overconfidence calibration.** RL with binary reward can produce a model that is highly confident in its answers — including wrong ones. A model that always says "12 rows" with high confidence when the answer is 11 is worse than one that hedges, because downstream users may trust it uncritically. Monitor calibration (confidence vs. accuracy) post-RL, and consider whether the deployment context needs the model to express uncertainty on ambiguous grids.

## Implementation steps

1. Grid renderer with deterministic seeds. Disjoint ranges: SFT [0, 100K) (direct [0, 20K), intermediate [20K, 40K), tool_use [40K, 50K)), RL [100K, 500K), eval [500K, 510K).
2. **Done.** `training/sft_generate.py` — generates 5,000 SFT examples across three strategies. `training/cot_templates.py` — 5 direct, 5 intermediate, 3 tool-use, and 3 tool-use-skip template variants.
3. **Done.** `training/rewards.py` — three reward functions (`outcome_reward`, `process_reward`, `tool_use_reward`) with answer parsing that uses last-match to avoid confusion with CoT intermediate values.
4. **Done.** `training/cli.py` — CLI with `generate`, `verify` (visual inspection), and `verify-reward` (reward sanity check) commands.
5. SFT with mixed batches, checkpoint every 500 steps. Evaluate pass@1 and pass@16 on 200 held-out samples. Stop at pass@16 ≥ 50% plateau.
6. RL: GRPO K=16, binary outcome + 0.2-weight process checks, pass@16 > 20% gating, KL β=0.05→0.01, ~15K episodes.
7. Eval on 1,000 held-out samples, uniform across sizes 3–25. Report per-size accuracy, tool-use rate, error type breakdown, pass@K curves.
