"""RL reward functions for grid counting.

Three reward functions, all with signature:
    (response_str, ground_truth_str, metadata_dict) -> float

Ground truth is in "R,C" format (e.g. "8,6").

Reward hardening notes
----------------------
The following risks are mitigated by code in this module:

- **CoT camouflage**: ``_cot_answer_consistent`` checks that the CoT's own
  arithmetic matches the final answer.  If the model writes "13 lines → 12
  rows" but answers rows=13, the process bonus is zeroed out.
- **Tool output fabrication**: ``_detect_fabrication_risk`` flags responses
  that include code blocks without real execution markers (no ``output``
  block, no image-processing imports).  Flagged responses receive a reduced
  tool-use reward.

Out-of-scope risks (documented for completeness):

- **Capability generalization**: monitor with OOD visual extraction
  benchmarks post-training.  Not addressable at the reward-function level.
- **Training data leakage**: audit SFT templates before training, strip
  image metadata.  Not addressable at the reward-function level.
"""

import re


def _parse_final_answer(response: str) -> str | None:
    """Parse the final rows=N columns=M answer from a response.

    Unlike the general row_col parser, this finds the *last* match to
    avoid confusion with intermediate CoT values like "Rows = 11 - 1 = 10".
    """
    # Find all rows=N / columns=M pairs; use last match
    row_matches = list(re.finditer(
        r"rows?\s*[=:]\s*\{?(\d+)\}?", response, re.IGNORECASE
    ))
    col_matches = list(re.finditer(
        r"col(?:umn)?s?\s*[=:]\s*\{?(\d+)\}?", response, re.IGNORECASE
    ))
    if row_matches and col_matches:
        return f"{row_matches[-1].group(1)},{col_matches[-1].group(1)}"
    # Fallback: NxM format, last match
    nxm = list(re.finditer(r"(\d+)\s*[x×,]\s*(\d+)", response))
    if nxm:
        return f"{nxm[-1].group(1)},{nxm[-1].group(2)}"
    return None


def outcome_reward(response: str, ground_truth: str, metadata: dict) -> float:
    """Binary exact-match reward.

    Parses the last rows=N columns=M in the response. Both rows AND
    columns must match exactly. Returns 1.0 if correct, 0.0 otherwise.
    """
    parsed = _parse_final_answer(response)
    if parsed is None:
        return 0.0
    return 1.0 if parsed == ground_truth else 0.0


def _extract_line_row_patterns(cot: str) -> list[tuple[int, int]]:
    """Extract (line_count, row_or_col_count) pairs from CoT text.

    Looks for patterns like:
      - "N lines → M rows"
      - "N lines, so M rows"
      - "N horizontal lines ... M rows"
      - "N lines means M rows"
      - "N lines make M rows"
      - "N - 1 = M"
    """
    pairs = []

    # Pattern: "N lines → M rows/columns" (various connectors)
    for m in re.finditer(
        r"(\d+)\s+(?:horizontal\s+|vertical\s+)?lines?"
        r"\s*(?:→|->|,\s*so|means?|make|gives?|creates?)\s*"
        r"(\d+)\s+(?:rows?|columns?)",
        cot, re.IGNORECASE,
    ):
        pairs.append((int(m.group(1)), int(m.group(2))))

    # Pattern: "N - 1 = M" (explicit subtraction)
    for m in re.finditer(r"(\d+)\s*-\s*1\s*=\s*(\d+)", cot):
        pairs.append((int(m.group(1)), int(m.group(2))))

    # Pattern: "N lines ... rows = M" or "N lines ... columns = M"
    # (more distant connection, check within 80 chars)
    for m in re.finditer(
        r"(\d+)\s+(?:horizontal\s+|vertical\s+)?lines?"
        r".{0,80}?"
        r"(?:rows?|columns?)\s*=\s*(\d+)",
        cot, re.IGNORECASE | re.DOTALL,
    ):
        n_lines = int(m.group(1))
        n_cells = int(m.group(2))
        # Avoid false positives: only add if subtraction is plausible
        if abs(n_lines - n_cells) <= 2:
            pairs.append((n_lines, n_cells))

    return pairs


def _cot_answer_consistent(response: str, ground_truth: str) -> float:
    """Check whether the CoT's own arithmetic matches the final answer.

    Extracts (line_count, cell_count) pairs from the CoT and checks
    the last horizontal/row pair's cell_count against the final answer's
    row value, and the last vertical/column pair's cell_count against
    the column value.

    Returns 1.0 if consistent (or no signal), 0.0 if mismatch detected.
    """
    parsed = _parse_final_answer(response)
    if parsed is None:
        return 1.0  # no answer to check against

    final_r, final_c = parsed.split(",")

    pairs = _extract_line_row_patterns(response)
    if not pairs:
        return 1.0  # no CoT arithmetic to check

    # Classify pairs by whether they reference rows or columns
    row_pairs = []
    col_pairs = []
    for m in re.finditer(
        r"(\d+)\s+(?:horizontal\s+)?lines?"
        r"\s*(?:→|->|,\s*so|means?|make|gives?|creates?)\s*"
        r"(\d+)\s+(rows?|columns?)",
        response, re.IGNORECASE,
    ):
        val = int(m.group(2))
        dim = m.group(3).lower()
        if dim.startswith("row"):
            row_pairs.append(val)
        else:
            col_pairs.append(val)

    # Also check "N - 1 = M" patterns near "row" or "column" context
    for m in re.finditer(r"(\d+)\s*-\s*1\s*=\s*(\d+)", response):
        result = int(m.group(2))
        # Check what's nearest after the match
        after = response[m.end():m.end() + 60].lower()
        row_pos = after.find("row")
        col_pos = after.find("col")
        if row_pos >= 0 and (col_pos < 0 or row_pos < col_pos):
            row_pairs.append(result)
        elif col_pos >= 0:
            col_pairs.append(result)

    # Check last pair for each dimension against final answer
    if row_pairs and str(row_pairs[-1]) != final_r:
        return 0.0
    if col_pairs and str(col_pairs[-1]) != final_c:
        return 0.0

    return 1.0


def process_reward(response: str, ground_truth: str, metadata: dict) -> float:
    """Combined outcome + process reward with CoT consistency check.

    Checks that the CoT correctly applies the N-1 rule:
    whenever "A lines → B rows" appears, B should equal A-1.

    Also checks CoT-answer consistency: the CoT's own subtraction
    result must match the final answer.  If the CoT says "12 rows"
    but the answer says rows=13, the process bonus is zeroed.

    Combined: R = max(outcome, 0.8 * outcome + 0.2 * process_score * consistency)
    Correct answers are never penalized below 1.0.
    """
    outcome = outcome_reward(response, ground_truth, metadata)

    pairs = _extract_line_row_patterns(response)
    if not pairs:
        # No process signal — fall back to pure outcome
        return outcome

    correct_pairs = sum(1 for n_lines, n_cells in pairs if n_cells == n_lines - 1)
    total_pairs = len(pairs)
    process_score = correct_pairs / total_pairs

    consistency = _cot_answer_consistent(response, ground_truth)

    combined = 0.8 * outcome + 0.2 * process_score * consistency
    return max(outcome, combined)


def _detect_code_blocks(response: str) -> bool:
    """Check if the response contains Python code blocks."""
    return bool(re.search(r"```python", response, re.IGNORECASE))


def _tool_output_consistent(response: str, ground_truth: str) -> bool | None:
    """Check if tool output is consistent with ground truth.

    Looks for patterns like "N rows x M columns" or "Rows: N, Columns: M"
    in code output blocks.

    Returns True if consistent, False if inconsistent, None if no output found.
    """
    # Find content in ```output blocks
    output_blocks = re.findall(
        r"```output\s*\n(.*?)```",
        response, re.DOTALL | re.IGNORECASE,
    )
    if not output_blocks:
        return None

    gt_r, gt_c = ground_truth.split(",")

    for block in output_blocks:
        # Check for "N rows" and "M columns" patterns
        row_m = re.search(r"(\d+)\s+rows?", block, re.IGNORECASE)
        col_m = re.search(r"(\d+)\s+columns?", block, re.IGNORECASE)
        if row_m and col_m:
            return row_m.group(1) == gt_r and col_m.group(1) == gt_c

        # Check for "NxM" pattern
        m = re.search(r"(\d+)\s*[x×]\s*(\d+)", block)
        if m:
            return m.group(1) == gt_r and m.group(2) == gt_c

    return None


def _detect_fabrication_risk(response: str) -> bool:
    """Heuristic check for likely fabricated tool output.

    Flags a response as suspicious when:
    - A Python code block is present but there is no ```output block
      (model wrote code but "ran" it without execution markers).
    - The code block contains no image-processing imports (numpy, PIL,
      cv2) — suggesting decorative code rather than real analysis.

    Returns True if fabrication is likely.
    """
    has_code = _detect_code_blocks(response)
    if not has_code:
        return False

    # Check for output block
    has_output = bool(re.search(r"```output", response, re.IGNORECASE))

    # Check for image-processing imports
    has_img_imports = bool(re.search(
        r"(?:import|from)\s+(?:numpy|PIL|cv2|skimage)",
        response, re.IGNORECASE,
    ))

    # Fabrication: code present but no output AND no real imports
    if not has_output and not has_img_imports:
        return True

    # Weaker signal: code with no output block at all
    if not has_output:
        return True

    return False


def tool_use_reward(response: str, ground_truth: str, metadata: dict) -> float:
    """Outcome + tool consistency reward with fabrication penalty.

    If tool output appears fabricated: outcome * 0.7
    If tool is used but output is misinterpreted: outcome * 0.5
    If no tool used: pure outcome (appropriate for easy grids).
    """
    outcome = outcome_reward(response, ground_truth, metadata)
    has_code = _detect_code_blocks(response)

    if not has_code:
        # No tool use — pure outcome
        return outcome

    # Check for fabricated tool output before checking consistency
    if _detect_fabrication_risk(response):
        return outcome * 0.7

    consistency = _tool_output_consistent(response, ground_truth)

    if consistency is None:
        # Tool used but can't parse output — pure outcome
        return outcome

    if consistency and outcome == 1.0:
        # Tool used correctly, answer correct
        return 1.0
    elif not consistency and outcome == 1.0:
        # Tool output wrong but answer correct (weird but possible)
        return 1.0
    elif consistency and outcome == 0.0:
        # Tool output correct but answer wrong (misinterpretation)
        return 0.5
    else:
        # Tool output wrong and answer wrong
        return 0.0
