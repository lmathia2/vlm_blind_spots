"""Tests for training.cot_templates — template filling and ASCII sketch."""

from random import Random

from training.cot_templates import (
    DIRECT_COT_TEMPLATES,
    INTERMEDIATE_COT_TEMPLATES,
    TOOL_USE_COT_TEMPLATES,
    TOOL_USE_SKIP_TEMPLATES,
    SELF_CORRECTION_INSERT,
    build_ascii_sketch,
    fill_template,
)


# ---------------------------------------------------------------------------
# Template lists
# ---------------------------------------------------------------------------

class TestTemplateLists:
    def test_direct_has_variants(self):
        assert len(DIRECT_COT_TEMPLATES) >= 3

    def test_intermediate_has_variants(self):
        assert len(INTERMEDIATE_COT_TEMPLATES) >= 3

    def test_tool_use_has_variants(self):
        assert len(TOOL_USE_COT_TEMPLATES) >= 3

    def test_tool_use_skip_has_variants(self):
        assert len(TOOL_USE_SKIP_TEMPLATES) >= 3

    def test_self_correction_insert_exists(self):
        assert "{h_lines}" in SELF_CORRECTION_INSERT
        assert "{rows}" in SELF_CORRECTION_INSERT


# ---------------------------------------------------------------------------
# build_ascii_sketch
# ---------------------------------------------------------------------------

class TestBuildAsciiSketch:
    def test_small_grid(self):
        sketch = build_ascii_sketch(3, 3)
        lines = sketch.split("\n")
        # 3 rows × 3 cols = 4 horizontal + 3 cell rows = 7 lines
        assert len(lines) == 7
        assert lines[0].startswith("+")
        assert "..." not in sketch

    def test_truncated_rows(self):
        sketch = build_ascii_sketch(10, 3)
        assert "..." in sketch

    def test_truncated_cols(self):
        sketch = build_ascii_sketch(3, 10)
        assert "..." in sketch

    def test_both_truncated(self):
        sketch = build_ascii_sketch(12, 12)
        assert "..." in sketch

    def test_exact_8x8(self):
        sketch = build_ascii_sketch(8, 8)
        # Should not truncate
        lines = sketch.split("\n")
        # 8 rows → 9 horizontal borders + 8 cell rows = 17 lines
        assert len(lines) == 17
        assert "..." not in sketch

    def test_1x1(self):
        sketch = build_ascii_sketch(1, 1)
        lines = sketch.split("\n")
        # 1 row: border + cell + border = 3 lines
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# fill_template
# ---------------------------------------------------------------------------

class TestFillTemplate:
    def test_basic_fill(self):
        template = "{rows} rows, {cols} columns, {h_lines} h, {v_lines} v"
        result = fill_template(template, 5, 3, Random(42))
        assert "5 rows" in result
        assert "3 columns" in result
        assert "6 h" in result  # 5+1
        assert "4 v" in result  # 3+1

    def test_interior_values(self):
        template = "h_interior={h_interior} v_interior={v_interior}"
        result = fill_template(template, 5, 3, Random(42))
        assert "h_interior=4" in result  # h_lines(6) - 2
        assert "v_interior=2" in result  # v_lines(4) - 2

    def test_all_direct_templates_fill(self):
        for i, template in enumerate(DIRECT_COT_TEMPLATES):
            result = fill_template(template, 8, 6, Random(i))
            assert "rows=8" in result
            assert "columns=6" in result

    def test_all_intermediate_templates_fill(self):
        for i, template in enumerate(INTERMEDIATE_COT_TEMPLATES):
            result = fill_template(template, 8, 6, Random(i))
            assert "rows=8" in result
            assert "columns=6" in result

    def test_all_tool_use_templates_fill(self):
        for i, template in enumerate(TOOL_USE_COT_TEMPLATES):
            result = fill_template(template, 15, 20, Random(i))
            assert "rows=15" in result
            assert "columns=20" in result

    def test_all_skip_templates_fill(self):
        for i, template in enumerate(TOOL_USE_SKIP_TEMPLATES):
            result = fill_template(template, 5, 4, Random(i))
            assert "rows=5" in result
            assert "columns=4" in result

    def test_self_correction_inserts(self):
        # Use the verbose template (variant 5) which has "horizontal"
        template = DIRECT_COT_TEMPLATES[4]
        result = fill_template(template, 8, 6, Random(42), include_self_correction=True)
        assert "Wait" in result
        assert "not the same thing" in result

    def test_self_correction_absent_when_false(self):
        template = DIRECT_COT_TEMPLATES[4]
        result = fill_template(template, 8, 6, Random(42), include_self_correction=False)
        assert "Wait" not in result

    def test_ascii_sketch_in_intermediate(self):
        # Template variant 2 uses {ascii_sketch}
        template = INTERMEDIATE_COT_TEMPLATES[1]
        result = fill_template(template, 4, 3, Random(42))
        assert "+---+" in result
