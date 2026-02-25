# Code Walkthrough Script

> 30-45 min, two parts. Audience has deep expertise and assigned the problem.

---

## Part 1: BlindTest Evaluation Framework (~15-20 min)

### 1.1 Opening: Problem and Architecture (2 min)

**Say:** "The question is: where exactly do vision-language models fail on
structured visual tasks? Is the bottleneck perception — they can't extract
information from pixels — or reasoning — they can't process it even when
given as text?"

**Architecture on whiteboard:**

```
Task Registry ──render()──> Image + Ground Truth
                                │
                        VisionClient.query()
                                │
                           Raw Response
                                │
                         Parser ─────> Parsed Answer
                                           │
                         Scorer ─────> {correct, score, error}
                                           │
                         Analysis ───> Tables, Plots, Diagnostics
```

**Say:** "Four clean boundaries. The task produces the image and ground truth.
The parser interprets model output. The scorer computes metrics. Analysis
aggregates. Each is a registry — adding a new task or scorer is one
function with a decorator."

---

### 1.2 Task System (3 min)

**Open:** `tasks/__init__.py`

**Say:** "Tasks self-register. The `__init__` module scans every `.py` file
in the directory with `pkgutil.iter_modules()`. If a module exports both
`TASK_CONFIG` and `render()`, it gets registered. No manual imports needed."

**Open:** `tasks/counting_grid.py` → show `TASK_CONFIG`:

```python
TASK_CONFIG = {
    "task_name": "counting_grid",
    "parser": "row_col",
    "scorer": "row_col",
    "default_params": {"rows": 8, "cols": 8, ...},
    "sweep_axes": {
        "rows": [4, 8, 12, 18, 25],
        "cols": [4, 8, 12, 18, 25],
        "question_type": ["grid_size", "total_cells", "merged_count"],
    },
}
```

**Say:** "`sweep_axes` defines the parameter grid. The harness takes the
Cartesian product and generates one sample per combination. This gives us
accuracy curves as a function of difficulty."

**Show:** `render()` signature: `(rows, cols, seed, ...) -> (Image, ground_truth, metadata)`

**Point out:** Seed-based determinism — same seed always produces the same
image. Critical for reproducibility across runs.

---

### 1.3 Runner / Harness (3 min)

**Open:** `harness.py`

**Say:** "The VisionClient wraps the Anthropic API. Base64-encodes an image,
sends it with a text prompt, returns the response."

**Point out the reasoning mode toggle (lines 59-65):**

```python
if self.reasoning:
    kwargs["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET}
    # temperature must not be set when thinking is enabled
else:
    kwargs["temperature"] = self.temperature
```

**Say:** "This lets us A/B test: does extended thinking (chain-of-thought)
actually help, or is the bottleneck purely visual?"

**Point out resume support:**

**Say:** "`evaluate_manifest()` appends results to JSONL. On restart, it
loads completed sample IDs and skips them. Evaluations are expensive — a
full sweep can take hours — so crash-safe resume was essential."

**Key detail:** Results written one-at-a-time with `ThreadPoolExecutor` +
`as_completed`. No batch writes that could lose data on interrupt.

---

### 1.4 Parsers and Scorers (3 min)

**Open:** `parsers.py`

**Say:** "Parsers are decorator-registered. Each maps raw model output to a
structured answer."

**Walk through `parse_row_col` as the example:**
1. Try `rows=N columns=M` (structured format)
2. Try `NxM` (fallback)
3. Return `"R,C"` string or `None`

**Say:** "The design principle: try explicit formats first, degrade to
heuristics. Return `None` on failure — don't guess."

**Open:** `scorers.py`

**Key scorer: `score_integer_distance`:**

```python
error = pred - gt          # positive = overcount, negative = undercount
score = 1.0 / (1 + abs_error)
```

**Say:** "This is important — we track signed error, not just accuracy.
It tells us whether models systematically overcount or undercount,
which directly informed the training approach later."

**Also note `score_row_col`:** gives 0.5 partial credit when one dimension
is correct. More signal than binary accuracy.

---

### 1.5 The Perception vs Reasoning Diagnostic (4 min)

**This is the key insight of Part 1.**

**Open:** `tasks/counting_grid_text.py` alongside `tasks/counting_grid.py`

**Say:** "For every image task, we build a text control. Same logic, same
ground truth — but the visual information is described as text in the
prompt instead of rendered as pixels."

**Example:** For a 5x3 grid:
- **Image task:** Renders actual grid lines, asks "how many rows and columns?"
- **Text control:** Says "horizontal lines at positions 0, 1, 2, 3, 4, 5"
  and asks the same question

**Say:** "If the model gets the text version right but the image version
wrong, the failure is perceptual — it can reason but can't see. If both
fail, the failure is reasoning."

**Open:** `analysis.py`, show `_classify_failure`:

```python
if image_acc > 0.80:        return "not_a_failure"
if text_acc > 0.80 and gap > 0.15: return "perceptual"
if gap < 0.10 and text_acc < 0.80: return "reasoning"
if gap >= 0.10:             return "mixed"
```

**Say:** "Auto-discovered by naming convention: `foo` and `foo_text` are
paired automatically. No manual registration."

**If asked about results:** "For grid counting, the gap was substantial —
models scored well on text but poorly on images for larger grids.
The failure is clearly perceptual, which is why the training work
targets visual processing specifically."

---

### 1.6 MC4 Multiple Choice (1 min)

**Open:** `mc4_utils.py`

**Say:** "Some tasks are easier to evaluate as multiple choice.
`generate_distractors` creates plausible wrong answers in three phases:
first from other visible values in the image, then percentage offsets,
then sequential steps. Spacing enforcement prevents options from being
trivially close."

**Skip details unless asked.** The interesting bit is the distractor
strategy — it makes wrong answers *plausible*.

---

### 1.7 Part 1 Transition (1 min)

**Say:** "So the evaluation framework showed us that grid counting fails
perceptually — the model can do the math when given text, but can't
reliably extract line positions from pixels. The question becomes: can
we teach it? That's Part 2."

---

## Part 2: SFT + RL Training Pipeline (~15-20 min)

### 2.1 Training Strategy Overview (2 min)

**Say:** "We use a two-phase approach. Phase 1 is SFT: teach the model
what good chain-of-thought looks like for grid counting, using
synthetically generated demonstrations. Phase 2 is RL with GRPO:
reinforce strategies that actually work, using shaped reward functions."

**Three reasoning strategies, by grid difficulty:**

| Strategy | Grid Size | Approach |
|----------|-----------|----------|
| Direct counting | 3-12 | Count lines, subtract 1 |
| Intermediate repr | 3-15 | Externalize perception (ASCII sketch, listing) |
| Tool use | 12-25 | Write Python code for line detection |

**Say:** "Small grids: just count. Medium: write down what you see first,
then reason. Large: use code. The ranges overlap deliberately —
the model learns when to switch strategies."

---

### 2.2 CoT Templates (3 min)

**Open:** `training/cot_templates.py`

**Say:** "Each strategy has 3-5 template variants — paraphrases of the
same reasoning approach. This prevents the model from memorizing one
surface form."

**Show a direct template example (walk through one):**

**Say:** "The key arithmetic: horizontal lines = rows + 1, then
subtract 1 to get rows. The templates embed this explicitly."

**Show `fill_template`:**

```python
h_lines = rows + 1
v_lines = cols + 1
```

**Say:** "Given ground truth dimensions, we derive all intermediate values
and fill the template. The model sees the complete reasoning trace."

**Point out self-correction (20% of direct samples):**

**Say:** "We deliberately inject a correction pattern: 'Wait — I said 13
rows, but 13 lines means 12 rows.' This teaches the model to catch
the off-by-one error, which is the most common failure mode."

**If asked about ASCII sketch:** "Intermediate templates optionally
include an ASCII grid visualization, capped at 8x8 to control token
count."

---

### 2.3 Data Synthesis (3 min)

**Open:** `training/sft_generate.py`

**Two things to highlight:**

**1. Uniform grid-size sampling:**

**Say:** "If you sample rows and columns independently with `randint`,
the joint distribution is biased — center sizes appear more often.
We pre-compute all valid `(rows, cols)` pairs and sample uniformly
from that list."

**Show:** `_get_grid_pairs` — cached list of all pairs per strategy.

**2. Anti-shortcut randomization:**

**Say:** "This is a defense against reward hacking. If every grid is
rendered at the same resolution on a white background, the model
could learn to infer grid size from pixel spacing without actually
counting."

**Show `_apply_anti_shortcut`:**

```python
# Random padding (0-30px per side)
img = ImageOps.expand(img, border=padding, fill=bg_color)
# Aspect-ratio stretch (±15%)
img = img.resize((new_w, new_h), Image.LANCZOS)
```

**Say:** "Random padding, background color variation, and aspect-ratio
stretching. All post-processing — the renderer stays clean.
This forces the model to actually parse grid structure."

**Seed ranges:**

**Say:** "Non-overlapping seed ranges: SFT uses 0-50K, RL uses 100K-500K,
eval uses 500K+. No data leakage between stages."

---

### 2.4 Reward Functions (5 min — this is the core)

**Open:** `training/rewards.py`

**Say:** "Three reward functions, each targeting a different training signal.
They share a parser — `_parse_final_answer` — that extracts the last
`rows=N columns=M` from the response. Last match, because CoT
intermediate values shouldn't be confused with the final answer."

---

**Reward 1: `outcome_reward`**

```
R = 1.0 if parsed_answer == ground_truth else 0.0
```

**Say:** "Pure binary. Both dimensions must be correct. This is the
baseline — just get the answer right."

---

**Reward 2: `process_reward`**

```
R = max(outcome, 0.8 * outcome + 0.2 * process_score * consistency)
```

**Say:** "This rewards correct reasoning, not just correct answers."

**Walk through the components:**

- **Process score:** "We extract `(line_count, cell_count)` pairs from
  the CoT — patterns like '10 lines → 9 rows'. Process score is the
  fraction where `cell_count == line_count - 1`."

- **Consistency check:** "Here's the anti-gaming mechanism. We check
  that the CoT's own arithmetic matches the final answer. If the model
  writes '13 lines → 12 rows' but then says `rows=13`, consistency
  drops to zero and the process bonus is wiped out."

**Show `_cot_answer_consistent`:**

**Say:** "It classifies each pair as row-related or column-related by
finding the nearest 'row' or 'col' keyword after the pattern. Then
checks the last pair of each type against the final answer."

- **The `max`:** "The `max(outcome, ...)` ensures correct answers are
  never penalized below 1.0. You get 1.0 for being right, plus
  a potential 0.2 bonus for correct process when wrong."

---

**Reward 3: `tool_use_reward`**

**Say:** "For the code-writing strategy. The interesting part is
fabrication detection."

**Show `_detect_fabrication_risk`:**

**Say:** "If the response has a Python code block but no `output` block,
or lacks image-processing imports, we flag it as likely fabricated.
The model wrote code that looks plausible but was never executed."

**Walk through the reward table:**

```
No code:                R = outcome (standard)
Fabricated code:        R = outcome * 0.7  (30% penalty)
Real code, right output, right answer:   R = 1.0
Real code, right output, wrong answer:   R = 0.5
Real code, wrong output:                 R = 0.0
```

**Say:** "The fabrication penalty is soft — 0.7x, not zero. We want
gradient signal, not a cliff. Over many episodes, this creates
pressure toward genuine tool use without catastrophically punishing
exploration."

**Say:** "The 0.5 for correct-output-wrong-answer catches misinterpretation:
the code did the right thing, but the model read its own output
wrong. We want to reward the code execution while penalizing the
final error."

---

### 2.5 Reward Hacking Diagnostics (3 min)

**Open:** `training/diagnostics.py`

**Say:** "During RL training, we need to watch for five failure modes."

**Walk through each briefly:**

| Check | What It Catches | Flag Threshold |
|-------|----------------|----------------|
| Answer distribution | Model always predicts "8,8" | Any answer > 15% |
| Per-size accuracy | Suspiciously uniform performance | Std < 0.05 across 5+ sizes |
| CoT consistency | Says one thing, answers another | Rate < 70% |
| Tool-use rate | Using code for trivial 4x4 grids | > 30% on grids ≤ 8 |
| Calibration | Confidently wrong | > 90% confidence, < 70% accuracy |

**Say:** "These run as a post-hoc check via `python -m training.cli diagnose`.
If flags fire during training, you know the reward function is being gamed
and need to intervene."

---

### 2.6 End-to-End Workflow (2 min)

**Say:** "Putting it together:"

```bash
# 1. Generate SFT data (2K direct + 2K intermediate + 1K tool_use)
python -m training.cli generate --strategy all --output training_data/

# 2. Sanity check: templates look good?
python -m training.cli verify --strategy all --n 3

# 3. Sanity check: generated samples get perfect reward?
python -m training.cli verify-reward --jsonl training_data/direct/samples.jsonl

# 4. Train SFT, then RL with GRPO using the reward functions
# (external training loop calls outcome_reward / process_reward / tool_use_reward)

# 5. Monitor for gaming
python -m training.cli diagnose --results rl_episodes.jsonl
```

**Say:** "The reward functions are pure Python — no API calls, no model
inference. They take `(response, ground_truth, metadata)` and return
a float. You plug them into whatever RL framework you're using."

---

### 2.7 Closing / Anticipated Questions

**If asked "why not just use outcome reward?":**
"Outcome reward has sparse signal — you only learn from correct answers.
Process reward gives gradient on incorrect answers that used good
reasoning. But process reward without consistency checks is gameable,
which is why we need the consistency and fabrication defenses."

**If asked "how do you know anti-shortcut works?":**
"We verify by checking that accuracy doesn't correlate with resolution
or padding in the eval set. If the model learned pixel spacing, larger
padding would degrade performance."

**If asked "what about other tasks?":**
"The reward functions are specific to grid counting, but the framework
is general. Adding a new task to the eval side is one file with
`TASK_CONFIG` and `render()`. The training side would need new
templates, but the reward architecture generalizes."

**If asked about test coverage:**
"264 unit tests covering parsers, scorers, all three reward functions,
CoT templates, data synthesis, diagnostics, and the CLI. All passing."
