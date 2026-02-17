"""RL reward functions for grid counting.

Three reward functions, all with signature:
    (response_str, ground_truth_str, metadata_dict) -> float

Ground truth is in "R,C" format (e.g. "8,6").
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


def process_reward(response: str, ground_truth: str, metadata: dict) -> float:
    """Combined outcome + process reward.

    Checks that the CoT correctly applies the N-1 rule:
    whenever "A lines → B rows" appears, B should equal A-1.

    Combined: R = max(outcome, 0.8 * outcome + 0.2 * process_score)
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

    combined = 0.8 * outcome + 0.2 * process_score
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


def tool_use_reward(response: str, ground_truth: str, metadata: dict) -> float:
    """Outcome + tool consistency reward.

    If tool is used but output is misinterpreted: outcome * 0.5
    If no tool used: pure outcome (appropriate for easy grids).
    """
    outcome = outcome_reward(response, ground_truth, metadata)
    has_code = _detect_code_blocks(response)

    if not has_code:
        # No tool use — pure outcome
        return outcome

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
