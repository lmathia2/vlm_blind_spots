# Lessons Learned

### 2026-02-15 Match parser to actual model response format
- **Pattern**: The `row_col` parser expected `rows=N columns=M` but the model responded with `rows={N} columns={M}` (curly brackets around numbers, matching the prompt format). This caused 100% parse failure on 264 counting_grid samples.
- **Rule**: When writing parsers, handle the response format the prompt asks for. If the prompt uses `{N}` as an example, the parser must handle curly brackets. Test parsers against actual model output before running large evaluations.
- **Example**: Before: `r"rows?\s*[=:]\s*(\d+)"` / After: `r"rows?\s*[=:]\s*\{?(\d+)\}?"`

### 2026-02-15 Never claim verification without reproducible proof
- **Pattern**: Marked sprint_zero as "verified" based on a single in-session API call that the user could not reproduce (API returned billing error when they tried).
- **Rule**: Never mark a task as verified unless the user can reproduce the result OR you show the full command + output and confirm the user can run it too. If there's an external dependency (API credits, network, etc.), note it as a prerequisite, not as verified.
- **Example**: Before: `- [x] Validate API key` / After: `- [ ] Validate API key — BLOCKED: API credits exhausted`

### 2026-02-15 Null prompts in dynamic-prompt tasks invalidate all results
- **Pattern**: Tasks with `prompt_template: None` (table_cell_read, arrow_following) produced manifests with `"prompt": null` because the cli.py prompt fallback `metadata.get("prompt", config["prompt_template"])` returns None when config["prompt_template"] is None. Evaluations ran with null prompts, producing meaningless 0% and 20% accuracy.
- **Rule**: For dynamic-prompt tasks, always verify the manifest contains non-null prompts before running evaluation. Add a pre-evaluation check or assertion.
- **Example**: table_cell_read went from 0% → 100% after fixing the prompt. The original 0% was entirely a measurement artifact.

### 2026-02-15 Small sample sizes and easy params mask real difficulty
- **Pattern**: nested_squares showed 97.9% accuracy on 48 samples (depth 2-5, thick lines) but dropped to 51.7% on 315 samples with harder params (depth 2-8, reduction 0.4-0.8, thin lines). The original result was misleadingly optimistic.
- **Rule**: When generated images are much easier than reference benchmarks, expand sweep axes to include harder parameter values before drawing conclusions. Match difficulty to the reference dataset range.
