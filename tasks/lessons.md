# Lessons Learned

### 2026-02-15 Match parser to actual model response format
- **Pattern**: The `row_col` parser expected `rows=N columns=M` but the model responded with `rows={N} columns={M}` (curly brackets around numbers, matching the prompt format). This caused 100% parse failure on 264 counting_grid samples.
- **Rule**: When writing parsers, handle the response format the prompt asks for. If the prompt uses `{N}` as an example, the parser must handle curly brackets. Test parsers against actual model output before running large evaluations.
- **Example**: Before: `r"rows?\s*[=:]\s*(\d+)"` / After: `r"rows?\s*[=:]\s*\{?(\d+)\}?"`
- **Pattern**: Marked sprint_zero as "verified" based on a single in-session API call that the user could not reproduce (API returned billing error when they tried).
- **Rule**: Never mark a task as verified unless the user can reproduce the result OR you show the full command + output and confirm the user can run it too. If there's an external dependency (API credits, network, etc.), note it as a prerequisite, not as verified.
- **Example**: Before: `- [x] Validate API key` / After: `- [ ] Validate API key — BLOCKED: API credits exhausted`
