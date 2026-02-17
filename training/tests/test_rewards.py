"""Tests for training.rewards — all reward functions and helpers."""

import pytest

from training.rewards import (
    _cot_answer_consistent,
    _detect_code_blocks,
    _detect_fabrication_risk,
    _extract_line_row_patterns,
    _parse_final_answer,
    _tool_output_consistent,
    outcome_reward,
    process_reward,
    tool_use_reward,
)


# ---------------------------------------------------------------------------
# _parse_final_answer
# ---------------------------------------------------------------------------

class TestParseAnswer:
    def test_rows_columns_format(self):
        assert _parse_final_answer("rows=8 columns=6") == "8,6"

    def test_case_insensitive(self):
        assert _parse_final_answer("Rows=8 Columns=6") == "8,6"

    def test_curly_brackets(self):
        assert _parse_final_answer("rows={8} columns={6}") == "8,6"

    def test_colon_format(self):
        assert _parse_final_answer("rows: 8 columns: 6") == "8,6"

    def test_nxm_fallback(self):
        assert _parse_final_answer("The grid is 8x6") == "8,6"

    def test_nxm_unicode(self):
        assert _parse_final_answer("The grid is 8×6") == "8,6"

    def test_last_match_wins(self):
        # CoT mentions "rows = 11 - 1 = 10" then final "rows=10"
        response = "rows = 11, then rows=10 columns=5"
        result = _parse_final_answer(response)
        assert result == "10,5"

    def test_no_answer(self):
        assert _parse_final_answer("I don't know") is None

    def test_empty_string(self):
        assert _parse_final_answer("") is None

    def test_partial_row_only(self):
        # Only rows, no columns → no match (need both)
        assert _parse_final_answer("rows=8") is None


# ---------------------------------------------------------------------------
# outcome_reward
# ---------------------------------------------------------------------------

class TestOutcomeReward:
    def test_correct(self):
        assert outcome_reward("rows=8 columns=6", "8,6", {}) == 1.0

    def test_wrong(self):
        assert outcome_reward("rows=9 columns=6", "8,6", {}) == 0.0

    def test_unparseable(self):
        assert outcome_reward("I can't tell", "8,6", {}) == 0.0

    def test_swapped(self):
        # rows and cols swapped → wrong
        assert outcome_reward("rows=6 columns=8", "8,6", {}) == 0.0


# ---------------------------------------------------------------------------
# _extract_line_row_patterns
# ---------------------------------------------------------------------------

class TestExtractLineRowPatterns:
    def test_arrow_pattern(self):
        pairs = _extract_line_row_patterns("10 lines → 9 rows")
        assert (10, 9) in pairs

    def test_dash_arrow(self):
        pairs = _extract_line_row_patterns("10 lines -> 9 rows")
        assert (10, 9) in pairs

    def test_so_pattern(self):
        pairs = _extract_line_row_patterns("10 lines, so 9 rows")
        assert (10, 9) in pairs

    def test_subtraction(self):
        pairs = _extract_line_row_patterns("10 - 1 = 9")
        assert (10, 9) in pairs

    def test_horizontal_qualifier(self):
        pairs = _extract_line_row_patterns("10 horizontal lines → 9 rows")
        assert (10, 9) in pairs

    def test_column_variant(self):
        pairs = _extract_line_row_patterns("7 lines → 6 columns")
        assert (7, 6) in pairs

    def test_distant_pattern(self):
        pairs = _extract_line_row_patterns("10 lines blah blah rows = 9")
        assert (10, 9) in pairs

    def test_no_patterns(self):
        assert _extract_line_row_patterns("just some text") == []

    def test_wrong_subtraction(self):
        # N - 1 = M where M != N-1 should still be extracted
        pairs = _extract_line_row_patterns("10 - 1 = 10")
        assert (10, 10) in pairs


# ---------------------------------------------------------------------------
# _cot_answer_consistent
# ---------------------------------------------------------------------------

class TestCotAnswerConsistent:
    def test_consistent_arrow(self):
        resp = "13 lines → 12 rows. 7 lines → 6 columns. rows=12 columns=6"
        assert _cot_answer_consistent(resp, "12,6") == 1.0

    def test_inconsistent_row(self):
        resp = "13 lines → 12 rows. rows=13 columns=6"
        assert _cot_answer_consistent(resp, "12,6") == 0.0

    def test_inconsistent_col(self):
        resp = "7 lines → 6 columns. rows=5 columns=7"
        assert _cot_answer_consistent(resp, "5,6") == 0.0

    def test_subtraction_with_context(self):
        resp = "6 - 1 = 5 rows. 7 - 1 = 6 columns. rows=5 columns=6"
        assert _cot_answer_consistent(resp, "5,6") == 1.0

    def test_subtraction_inconsistent(self):
        resp = "6 - 1 = 5 rows. 7 - 1 = 6 columns. rows=6 columns=6"
        assert _cot_answer_consistent(resp, "5,6") == 0.0

    def test_no_cot_patterns(self):
        # No signal → return 1.0 (benefit of doubt)
        assert _cot_answer_consistent("rows=5 columns=6", "5,6") == 1.0

    def test_no_answer(self):
        assert _cot_answer_consistent("some rambling text", "5,6") == 1.0


# ---------------------------------------------------------------------------
# process_reward
# ---------------------------------------------------------------------------

class TestProcessReward:
    def test_correct_answer_correct_process(self):
        resp = "10 lines → 9 rows. 7 lines → 6 columns.\nrows=9 columns=6"
        assert process_reward(resp, "9,6", {}) == 1.0

    def test_correct_answer_wrong_process(self):
        # Answer correct but CoT says 10 → 10 (wrong subtraction)
        resp = "10 lines → 10 rows.\nrows=9 columns=6"
        # outcome=1.0, process=0.0, combined=0.8, max(1.0, 0.8)=1.0
        assert process_reward(resp, "9,6", {}) == 1.0

    def test_wrong_answer_correct_process(self):
        resp = "10 lines → 9 rows.\nrows=8 columns=6"
        # outcome=0.0, process=1.0, consistency depends on mismatch
        # CoT says 9, answer says 8 → consistency=0.0 → combined=0.0
        assert process_reward(resp, "9,6", {}) == 0.0

    def test_no_process_signal(self):
        resp = "I count 9 rows and 6 columns.\nrows=9 columns=6"
        assert process_reward(resp, "9,6", {}) == 1.0

    def test_camouflage_zeroes_bonus(self):
        # CoT says 12 rows, answer says 13 → inconsistent
        resp = "13 lines → 12 rows.\nrows=13 columns=6"
        # outcome=0.0 (answer 13,6 != 12,6), consistency=0.0
        reward = process_reward(resp, "12,6", {})
        assert reward == 0.0

    def test_process_bonus_adds_when_all_consistent(self):
        # Wrong answer but correct process + consistent CoT
        resp = "10 lines → 9 rows.\nrows=9 columns=5"
        # outcome=0.0, process=1.0, consistency=1.0
        # combined=0.8*0+0.2*1.0*1.0=0.2
        reward = process_reward(resp, "9,6", {})
        assert reward == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# _detect_code_blocks / _tool_output_consistent / _detect_fabrication_risk
# ---------------------------------------------------------------------------

class TestCodeBlockDetection:
    def test_has_python_block(self):
        assert _detect_code_blocks("```python\nprint('hi')\n```")

    def test_no_code_block(self):
        assert not _detect_code_blocks("just text")

    def test_case_insensitive(self):
        assert _detect_code_blocks("```Python\ncode\n```")


class TestToolOutputConsistent:
    def test_consistent_rows_cols(self):
        resp = "```output\n8 rows x 6 columns\n```"
        assert _tool_output_consistent(resp, "8,6") is True

    def test_inconsistent(self):
        resp = "```output\n9 rows x 6 columns\n```"
        assert _tool_output_consistent(resp, "8,6") is False

    def test_nxm_format(self):
        resp = "```output\n8x6\n```"
        assert _tool_output_consistent(resp, "8,6") is True

    def test_no_output_block(self):
        assert _tool_output_consistent("no output here", "8,6") is None

    def test_unparseable_output(self):
        resp = "```output\nsome random text\n```"
        assert _tool_output_consistent(resp, "8,6") is None


class TestFabricationDetection:
    def test_no_code(self):
        assert not _detect_fabrication_risk("just text")

    def test_code_with_output(self):
        resp = "```python\nfrom PIL import Image\n```\n```output\nresult\n```"
        assert not _detect_fabrication_risk(resp)

    def test_code_no_output_no_imports(self):
        resp = "```python\nlines = count()\nprint(lines)\n```"
        assert _detect_fabrication_risk(resp)

    def test_code_no_output_with_imports(self):
        resp = "```python\nimport numpy as np\nresult = np.sum(x)\n```"
        assert _detect_fabrication_risk(resp)

    def test_code_with_pil_import_and_output(self):
        resp = "```python\nfrom PIL import Image\n```\n```output\n8 rows\n```"
        assert not _detect_fabrication_risk(resp)


# ---------------------------------------------------------------------------
# tool_use_reward
# ---------------------------------------------------------------------------

class TestToolUseReward:
    def test_no_code_correct(self):
        assert tool_use_reward("rows=8 columns=6", "8,6", {}) == 1.0

    def test_no_code_wrong(self):
        assert tool_use_reward("rows=9 columns=6", "8,6", {}) == 0.0

    def test_fabricated_correct(self):
        resp = "```python\nlines = count()\n```\nrows=8 columns=6"
        assert tool_use_reward(resp, "8,6", {}) == pytest.approx(0.7)

    def test_fabricated_wrong(self):
        resp = "```python\nlines = count()\n```\nrows=9 columns=6"
        assert tool_use_reward(resp, "8,6", {}) == pytest.approx(0.0)

    def test_real_tool_correct_output_correct_answer(self):
        resp = (
            "```python\nfrom PIL import Image\n```\n"
            "```output\n8 rows x 6 columns\n```\n"
            "rows=8 columns=6"
        )
        assert tool_use_reward(resp, "8,6", {}) == 1.0

    def test_real_tool_correct_output_wrong_answer(self):
        resp = (
            "```python\nfrom PIL import Image\n```\n"
            "```output\n8 rows x 6 columns\n```\n"
            "rows=9 columns=6"
        )
        assert tool_use_reward(resp, "8,6", {}) == 0.5

    def test_real_tool_wrong_output_wrong_answer(self):
        resp = (
            "```python\nfrom PIL import Image\n```\n"
            "```output\n9 rows x 7 columns\n```\n"
            "rows=9 columns=7"
        )
        assert tool_use_reward(resp, "8,6", {}) == 0.0

    def test_real_tool_unparseable_output(self):
        resp = (
            "```python\nfrom PIL import Image\n```\n"
            "```output\nsome text\n```\n"
            "rows=8 columns=6"
        )
        assert tool_use_reward(resp, "8,6", {}) == 1.0
