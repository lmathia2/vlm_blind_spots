"""Tests for scorers.py — evaluation scoring functions."""

import pytest

from scorers import (
    SCORER_REGISTRY,
    score_exact_match,
    score_integer_distance,
    score_row_col,
    score_set_match,
    score_set_member,
)


class TestScorerRegistry:
    def test_all_scorers_registered(self):
        expected = {"exact_match", "set_member", "integer_distance", "row_col", "set_match"}
        assert expected.issubset(set(SCORER_REGISTRY.keys()))


class TestScoreExactMatch:
    def test_correct(self):
        result = score_exact_match("42", "42")
        assert result["correct"] is True
        assert result["score"] == 1.0

    def test_case_insensitive(self):
        result = score_exact_match("Yes", "yes")
        assert result["correct"] is True

    def test_wrong(self):
        result = score_exact_match("41", "42")
        assert result["correct"] is False
        assert result["score"] == 0.0

    def test_none(self):
        result = score_exact_match(None, "42")
        assert result["correct"] is False

    def test_whitespace(self):
        result = score_exact_match("  42  ", "42")
        assert result["correct"] is True


class TestScoreSetMember:
    def test_member(self):
        result = score_set_member("B", "A,B,C")
        assert result["correct"] is True

    def test_not_member(self):
        result = score_set_member("D", "A,B,C")
        assert result["correct"] is False

    def test_case_insensitive(self):
        result = score_set_member("b", "A,B,C")
        assert result["correct"] is True

    def test_none(self):
        result = score_set_member(None, "A,B")
        assert result["correct"] is False


class TestScoreIntegerDistance:
    def test_exact(self):
        result = score_integer_distance("42", "42")
        assert result["correct"] is True
        assert result["score"] == 1.0
        assert result["error"] == 0

    def test_overcount(self):
        result = score_integer_distance("44", "42")
        assert result["correct"] is False
        assert result["error"] == 2
        assert result["abs_error"] == 2

    def test_undercount(self):
        result = score_integer_distance("40", "42")
        assert result["error"] == -2

    def test_score_decays(self):
        r1 = score_integer_distance("43", "42")
        r2 = score_integer_distance("50", "42")
        assert r1["score"] > r2["score"]

    def test_none(self):
        result = score_integer_distance(None, "42")
        assert result["correct"] is False
        assert result["error"] is None

    def test_non_numeric(self):
        result = score_integer_distance("abc", "42")
        assert result["correct"] is False


class TestScoreRowCol:
    def test_both_correct(self):
        result = score_row_col("8,6", "8,6")
        assert result["correct"] is True
        assert result["score"] == 1.0
        assert result["row_correct"] is True
        assert result["col_correct"] is True

    def test_row_only(self):
        result = score_row_col("8,5", "8,6")
        assert result["correct"] is False
        assert result["score"] == 0.5
        assert result["row_correct"] is True
        assert result["col_correct"] is False

    def test_col_only(self):
        result = score_row_col("9,6", "8,6")
        assert result["correct"] is False
        assert result["score"] == 0.5

    def test_both_wrong(self):
        result = score_row_col("9,5", "8,6")
        assert result["score"] == 0.0

    def test_none(self):
        result = score_row_col(None, "8,6")
        assert result["correct"] is False


class TestScoreSetMatch:
    def test_exact(self):
        result = score_set_match("A,B,C", "A,B,C")
        assert result["correct"] is True
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_subset(self):
        result = score_set_match("A,B", "A,B,C")
        assert result["correct"] is False
        assert result["precision"] == 1.0
        assert result["recall"] == pytest.approx(2 / 3)

    def test_superset(self):
        result = score_set_match("A,B,C,D", "A,B,C")
        assert result["correct"] is False
        assert result["precision"] == pytest.approx(3 / 4)
        assert result["recall"] == 1.0

    def test_disjoint(self):
        result = score_set_match("X,Y", "A,B")
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_none(self):
        result = score_set_match(None, "A,B")
        assert result["correct"] is False

    def test_empty_gt(self):
        result = score_set_match("", "")
        assert result["correct"] is True

    def test_case_insensitive(self):
        result = score_set_match("a,b,c", "A,B,C")
        assert result["correct"] is True
