"""Tests for mc4_utils.py — MC4 distractor and prompt generation."""

from random import Random

import pytest

from mc4_utils import format_mc4_prompt, generate_distractors


class TestGenerateDistractors:
    def test_returns_n_distractors(self):
        result = generate_distractors(100, [80, 120, 60, 140], n=3, rng=Random(42))
        assert len(result) == 3

    def test_does_not_include_correct(self):
        result = generate_distractors(100, [100, 80, 120], n=3, rng=Random(42))
        assert 100 not in result

    def test_spacing(self):
        result = generate_distractors(100, [80, 120, 60, 140], n=3,
                                       min_spacing_pct=0.15, rng=Random(42))
        all_vals = [100] + result
        for i, a in enumerate(all_vals):
            for b in all_vals[i + 1:]:
                assert abs(a - b) >= 1

    def test_zero_correct(self):
        result = generate_distractors(0, [10, 20, 30], n=3, rng=Random(42))
        assert len(result) == 3
        assert 0 not in result

    def test_no_other_values(self):
        result = generate_distractors(50, [], n=3, rng=Random(42))
        assert len(result) == 3

    def test_few_other_values(self):
        result = generate_distractors(50, [60], n=3, rng=Random(42))
        assert len(result) == 3


class TestFormatMc4Prompt:
    def test_contains_all_options(self):
        prompt, letter = format_mc4_prompt(
            "What is the value?", 100, [80, 120, 60],
            rng=Random(42),
        )
        assert "(A)" in prompt
        assert "(B)" in prompt
        assert "(C)" in prompt
        assert "(D)" in prompt

    def test_correct_letter_valid(self):
        _, letter = format_mc4_prompt(
            "Q?", 100, [80, 120, 60], rng=Random(42),
        )
        assert letter in {"A", "B", "C", "D"}

    def test_correct_value_present(self):
        prompt, letter = format_mc4_prompt(
            "Q?", 42, [10, 20, 30], rng=Random(42),
        )
        assert "42" in prompt

    def test_ends_with_instruction(self):
        prompt, _ = format_mc4_prompt(
            "Q?", 42, [10, 20, 30], rng=Random(42),
        )
        assert "Answer with only the letter" in prompt

    def test_value_format(self):
        prompt, _ = format_mc4_prompt(
            "Q?", 1000, [500, 1500, 2000],
            value_format="${:,.0f}",
            rng=Random(42),
        )
        assert "$1,000" in prompt

    def test_deterministic(self):
        p1, l1 = format_mc4_prompt("Q?", 42, [10, 20, 30], rng=Random(99))
        p2, l2 = format_mc4_prompt("Q?", 42, [10, 20, 30], rng=Random(99))
        assert p1 == p2
        assert l1 == l2
