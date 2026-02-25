"""Tests for parsers.py — response parsing for VLM evaluation."""

import pytest

from parsers import (
    PARSER_REGISTRY,
    parse_csv_cell_labels,
    parse_csv_letters,
    parse_csv_words,
    parse_exact_string,
    parse_integer,
    parse_letter,
    parse_mc4,
    parse_row_col,
    parse_yes_no,
)


class TestParserRegistry:
    def test_all_parsers_registered(self):
        expected = {
            "integer", "yes_no", "letter", "row_col", "mc4",
            "exact_string", "csv_words", "csv_letters", "csv_cell_labels",
        }
        assert expected.issubset(set(PARSER_REGISTRY.keys()))


class TestParseInteger:
    def test_curly_brackets(self):
        assert parse_integer("The answer is {42}") == "42"

    def test_plain_number(self):
        assert parse_integer("I count 42 cells") == "42"

    def test_last_number_wins(self):
        assert parse_integer("First 10, then 20, finally 42") == "42"

    def test_no_number(self):
        assert parse_integer("no numbers here") is None


class TestParseYesNo:
    def test_yes(self):
        assert parse_yes_no("Yes, that's correct") == "Yes"

    def test_no(self):
        assert parse_yes_no("No, it doesn't match") == "No"

    def test_embedded_yes(self):
        assert parse_yes_no("I think yes it does") == "Yes"

    def test_embedded_no(self):
        assert parse_yes_no("I would say no") == "No"

    def test_neither(self):
        assert parse_yes_no("maybe") is None


class TestParseLetter:
    def test_curly_brackets(self):
        assert parse_letter("The answer is {B}") == "B"

    def test_answer_is_pattern(self):
        assert parse_letter("The answer is C") == "C"

    def test_standalone_letter(self):
        assert parse_letter("B") == "B"

    def test_lowercase(self):
        assert parse_letter("{c}") == "C"

    def test_no_letter(self):
        assert parse_letter("12345") is None


class TestParseRowCol:
    def test_rows_columns_format(self):
        assert parse_row_col("rows=8 columns=6") == "8,6"

    def test_curly_format(self):
        assert parse_row_col("rows={8} columns={6}") == "8,6"

    def test_nxm_format(self):
        assert parse_row_col("8x6") == "8,6"

    def test_no_match(self):
        assert parse_row_col("no grid here") is None


class TestParseMc4:
    def test_curly(self):
        assert parse_mc4("The answer is {A}") == "A"

    def test_paren(self):
        assert parse_mc4("I choose (B)") == "B"

    def test_answer_is(self):
        assert parse_mc4("answer is C") == "C"

    def test_bold(self):
        assert parse_mc4("**D**") == "D"

    def test_standalone(self):
        assert parse_mc4("After analysis: B") == "B"

    def test_no_match(self):
        assert parse_mc4("I don't know") is None


class TestParseExactString:
    def test_curly_brackets(self):
        assert parse_exact_string("The value is {hello world}") == "hello world"

    def test_quoted(self):
        assert parse_exact_string('It says "hello"') == "hello"

    def test_answer_is_prefix(self):
        assert parse_exact_string("The answer is: foobar") == "foobar"

    def test_short_response(self):
        assert parse_exact_string("hello") == "hello"

    def test_empty(self):
        assert parse_exact_string("") is None

    def test_whitespace_only(self):
        assert parse_exact_string("   ") is None

    def test_latex_dollar_delimiters(self):
        """Qwen3-VL wraps answers in {$...$} LaTeX-style."""
        assert parse_exact_string("{$J. Smith$}") == "J. Smith"

    def test_latex_single_dollar_preserved(self):
        """Single leading $ is currency, not a LaTeX delimiter — keep it."""
        assert parse_exact_string("{$1,085}") == "$1,085"

    def test_latex_dollar_with_currency(self):
        assert parse_exact_string("{$1,551}") == "$1,551"

    def test_no_dollar_stripping_when_not_delimiter(self):
        """Don't strip $ when it's part of a currency value like $50."""
        assert parse_exact_string("{$50.00}") == "$50.00"

    def test_paired_dollars_stripped(self):
        """Paired $...$ are LaTeX delimiters and should be stripped."""
        assert parse_exact_string("{$Gamma$}") == "Gamma"

    def test_curly_normal_no_dollar(self):
        """Normal curly brace extraction still works."""
        assert parse_exact_string("{Gamma}") == "Gamma"


class TestParseCsvWords:
    def test_curly_comma(self):
        assert parse_csv_words("{Apple, Banana, Cherry}") == "Apple,Banana,Cherry"

    def test_sorted(self):
        assert parse_csv_words("{Cherry, Apple, Banana}") == "Apple,Banana,Cherry"

    def test_empty_braces(self):
        assert parse_csv_words("{}") == ""

    def test_none_value(self):
        assert parse_csv_words("{None}") == ""

    def test_no_match(self):
        assert parse_csv_words("random text") is None


class TestParseCsvLetters:
    def test_curly(self):
        assert parse_csv_letters("{A, C, E}") == "A,C,E"

    def test_sorted_deduped(self):
        assert parse_csv_letters("{C, A, C, E}") == "A,C,E"

    def test_no_match(self):
        assert parse_csv_letters("12345") is None


class TestParseCsvCellLabels:
    def test_curly(self):
        assert parse_csv_cell_labels("{A1, B2, C3}") == "A1,B2,C3"

    def test_sorted(self):
        assert parse_csv_cell_labels("{C3, A1, B2}") == "A1,B2,C3"

    def test_plain_text(self):
        assert parse_csv_cell_labels("The cells are A1 and B2") == "A1,B2"

    def test_no_match(self):
        assert parse_csv_cell_labels("no cells") is None
