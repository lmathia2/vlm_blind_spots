"""Tests for analysis.py — pure utility functions."""

import json

import pytest

from analysis import load_results, wilson_ci


class TestLoadResults:
    def test_loads_jsonl(self, tmp_path):
        p = tmp_path / "results.jsonl"
        records = [{"a": 1}, {"a": 2}, {"a": 3}]
        with open(p, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        result = load_results(p)
        assert len(result) == 3
        assert result[0]["a"] == 1

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "results.jsonl"
        with open(p, "w") as f:
            f.write('{"a": 1}\n\n{"a": 2}\n')
        result = load_results(p)
        assert len(result) == 2

    def test_empty_file(self, tmp_path):
        p = tmp_path / "results.jsonl"
        p.write_text("")
        result = load_results(p)
        assert result == []


class TestWilsonCi:
    def test_perfect_score(self):
        lo, hi = wilson_ci(100, 100)
        assert lo > 0.95
        assert hi == pytest.approx(1.0)

    def test_zero_score(self):
        lo, hi = wilson_ci(0, 100)
        assert lo == 0.0
        assert hi < 0.05

    def test_empty(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_half(self):
        lo, hi = wilson_ci(50, 100)
        assert 0.3 < lo < 0.5
        assert 0.5 < hi < 0.7

    def test_bounds_in_range(self):
        lo, hi = wilson_ci(10, 20)
        assert 0.0 <= lo <= hi <= 1.0
