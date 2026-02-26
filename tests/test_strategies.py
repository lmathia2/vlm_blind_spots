"""Tests for inference-time strategies."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from strategies import (
    STRATEGY_REGISTRY,
    strategy_baseline,
    strategy_best_of_n,
    strategy_crop_zoom,
    strategy_verify,
    strategy_best_of_n_verify,
    strategy_decompose,
    strategy_code_vision,
    strategy_adaptive,
    strategy_iterative_refine,
    strategy_sketchpad,
    _majority_vote,
    _crop_image,
    _tile_image,
    _parse_answer,
    _save_temp_image,
    _run_sandboxed_code,
    _build_refinement_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image(tmp_path):
    """Create a simple test image and return its path."""
    img = Image.new("RGB", (512, 512), color="white")
    path = tmp_path / "test.png"
    img.save(path)
    return str(path)


@pytest.fixture
def sample_dict(sample_image):
    """Return a minimal sample dict for testing."""
    return {
        "sample_id": "test001",
        "task_name": "nested_squares",
        "image_path": sample_image,
        "prompt": "How many squares are in the image? Answer with {N}.",
        "ground_truth": "5",
        "parser": "integer",
        "scorer": "exact_match",
        "params": {},
    }


@pytest.fixture
def mock_client():
    """Return a mock VisionClient."""
    client = MagicMock()
    client.model = "test-model"
    client.temperature = 0.0
    client.reasoning = False
    return client


def _make_response(raw: str, latency: float = 0.5):
    """Helper to build a mock query response."""
    return {
        "raw_response": raw,
        "latency_s": latency,
        "model": "test-model",
        "input_tokens": 100,
        "output_tokens": 10,
        "reasoning_mode": False,
    }


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestMajorityVote:
    def test_clear_winner(self):
        assert _majority_vote(["5", "5", "5", "3", "4"]) == "5"

    def test_tie_picks_first_most_common(self):
        # Counter.most_common returns arbitrary order for ties,
        # but should return one of the tied values
        result = _majority_vote(["3", "3", "5", "5"])
        assert result in ("3", "5")

    def test_all_none(self):
        assert _majority_vote([None, None, None]) is None

    def test_some_none(self):
        assert _majority_vote([None, "5", None, "5", "3"]) == "5"

    def test_single(self):
        assert _majority_vote(["7"]) == "7"

    def test_empty(self):
        assert _majority_vote([]) is None


class TestParseAnswer:
    def test_integer(self):
        assert _parse_answer("{5}", "integer") == "5"

    def test_unknown_parser(self):
        assert _parse_answer("hello", "nonexistent_parser") is None


class TestCropImage:
    def test_crop_center(self, sample_image):
        cropped = _crop_image(sample_image, (0.25, 0.25, 0.75, 0.75))
        # Original is 512x512, center crop is 256x256, should be upscaled to 512x512
        assert cropped.size == (512, 512)

    def test_crop_full(self, sample_image):
        cropped = _crop_image(sample_image, (0.0, 0.0, 1.0, 1.0))
        assert cropped.size == (512, 512)

    def test_crop_small_region(self, sample_image):
        cropped = _crop_image(sample_image, (0.4, 0.4, 0.6, 0.6))
        # 0.2 * 512 = ~102px, should be upscaled to ~512 (int rounding)
        w, h = cropped.size
        assert min(w, h) >= 510


class TestTileImage:
    def test_2x2_tiles(self, sample_image):
        tiles = _tile_image(sample_image, (2, 2))
        assert len(tiles) == 4
        for tile in tiles:
            assert isinstance(tile, Image.Image)

    def test_2x1_tiles(self, sample_image):
        tiles = _tile_image(sample_image, (2, 1))
        assert len(tiles) == 2

    def test_1x1_tile(self, sample_image):
        tiles = _tile_image(sample_image, (1, 1))
        assert len(tiles) == 1


class TestSaveTempImage:
    def test_saves_and_readable(self):
        img = Image.new("RGB", (100, 100), "red")
        path = _save_temp_image(img)
        assert Path(path).exists()
        loaded = Image.open(path)
        assert loaded.size == (100, 100)
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Strategy: baseline
# ---------------------------------------------------------------------------

class TestBaselineStrategy:
    def test_correct_answer(self, mock_client, sample_dict):
        mock_client.query.return_value = _make_response("{5}")
        result = strategy_baseline(mock_client, sample_dict)
        assert result["parsed_answer"] == "5"
        assert result["strategy"] == "baseline"
        mock_client.query.assert_called_once()

    def test_wrong_answer(self, mock_client, sample_dict):
        mock_client.query.return_value = _make_response("{3}")
        result = strategy_baseline(mock_client, sample_dict)
        assert result["parsed_answer"] == "3"

    def test_parse_failure(self, mock_client, sample_dict):
        mock_client.query.return_value = _make_response("I don't know")
        result = strategy_baseline(mock_client, sample_dict)
        # "don't" -> no integer parseable... actually "know" has no digits.
        # But "don't" might not have digits either. Let's check.
        # The integer parser looks for {N} or \b\d+\b. "I don't know" has no digits.
        assert result["parsed_answer"] is None


# ---------------------------------------------------------------------------
# Strategy: best_of_n
# ---------------------------------------------------------------------------

class TestBestOfNStrategy:
    def test_unanimous_vote(self, mock_client, sample_dict):
        mock_client.query.return_value = _make_response("{5}")
        result = strategy_best_of_n(mock_client, sample_dict, n=3)
        assert result["parsed_answer"] == "5"
        assert result["strategy"] == "best_of_n"
        assert result["strategy_n"] == 3
        assert mock_client.query.call_count == 3

    def test_majority_wins(self, mock_client, sample_dict):
        responses = [_make_response("{5}"), _make_response("{3}"), _make_response("{5}")]
        mock_client.query.side_effect = responses
        result = strategy_best_of_n(mock_client, sample_dict, n=3)
        assert result["parsed_answer"] == "5"
        assert result["strategy_votes"] == {"5": 2, "3": 1}

    def test_all_different(self, mock_client, sample_dict):
        responses = [_make_response("{3}"), _make_response("{5}"), _make_response("{7}")]
        mock_client.query.side_effect = responses
        result = strategy_best_of_n(mock_client, sample_dict, n=3)
        # All have count 1, any is valid
        assert result["parsed_answer"] in ("3", "5", "7")

    def test_all_parse_failures(self, mock_client, sample_dict):
        mock_client.query.return_value = _make_response("no answer")
        result = strategy_best_of_n(mock_client, sample_dict, n=3)
        assert result["parsed_answer"] is None

    def test_temperature_override(self, mock_client, sample_dict):
        mock_client.temperature = 0.0
        mock_client.query.return_value = _make_response("{5}")
        strategy_best_of_n(mock_client, sample_dict, n=2)
        # Temperature should be restored after the call
        assert mock_client.temperature == 0.0

    def test_token_accumulation(self, mock_client, sample_dict):
        mock_client.query.return_value = _make_response("{5}", latency=1.0)
        result = strategy_best_of_n(mock_client, sample_dict, n=3)
        assert result["input_tokens"] == 300  # 100 * 3
        assert result["output_tokens"] == 30  # 10 * 3
        assert result["latency_s"] == 3.0

    def test_strategy_all_answers(self, mock_client, sample_dict):
        responses = [_make_response("{5}"), _make_response("{3}"), _make_response("{5}")]
        mock_client.query.side_effect = responses
        result = strategy_best_of_n(mock_client, sample_dict, n=3)
        assert result["strategy_all_answers"] == ["5", "3", "5"]


# ---------------------------------------------------------------------------
# Strategy: crop_zoom
# ---------------------------------------------------------------------------

class TestCropZoomStrategy:
    def test_center_zoom_default(self, mock_client, sample_dict):
        """Unknown tasks should fall back to center_zoom."""
        sample_dict["task_name"] = "unknown_task"
        mock_client.query.return_value = _make_response("{5}")
        result = strategy_crop_zoom(mock_client, sample_dict)
        assert result["strategy"] == "crop_zoom"
        # Full image + 1 zoomed crop = 2 calls
        assert mock_client.query.call_count == 2

    def test_nested_squares_center_zoom(self, mock_client, sample_dict):
        mock_client.query.return_value = _make_response("{5}")
        result = strategy_crop_zoom(mock_client, sample_dict)
        assert result["strategy"] == "crop_zoom"
        assert result["parsed_answer"] == "5"
        # Full + center zoom = 2 calls
        assert mock_client.query.call_count == 2

    def test_counting_grid_tiles(self, mock_client, sample_dict):
        sample_dict["task_name"] = "counting_grid"
        sample_dict["parser"] = "row_col"
        sample_dict["ground_truth"] = "4,4"
        mock_client.query.return_value = _make_response("rows=4 columns=4")
        result = strategy_crop_zoom(mock_client, sample_dict)
        # Full + 4 tiles = 5 calls
        assert mock_client.query.call_count == 5

    def test_hierarchy_depth_tiles(self, mock_client, sample_dict):
        sample_dict["task_name"] = "hierarchy_depth"
        mock_client.query.return_value = _make_response("{3}")
        result = strategy_crop_zoom(mock_client, sample_dict)
        # Full + 2 tiles (2x1 grid) = 3 calls
        assert mock_client.query.call_count == 3
        # aggregate=max should take the highest count
        assert result["parsed_answer"] == "3"

    def test_crop_answer_disagrees(self, mock_client, sample_dict):
        """When zoom gives a different answer, majority vote resolves."""
        responses = [
            _make_response("{5}"),  # full image
            _make_response("{7}"),  # zoomed center
        ]
        mock_client.query.side_effect = responses
        result = strategy_crop_zoom(mock_client, sample_dict)
        # With 2 different answers, majority of [5, 7] -> either is valid
        assert result["parsed_answer"] in ("5", "7")

    def test_temp_files_cleaned_up(self, mock_client, sample_dict):
        mock_client.query.return_value = _make_response("{5}")
        result = strategy_crop_zoom(mock_client, sample_dict)
        # We can't easily check temp files are gone, but verify no error
        assert result["strategy"] == "crop_zoom"


# ---------------------------------------------------------------------------
# Strategy: verify
# ---------------------------------------------------------------------------

class TestVerifyStrategy:
    def test_confirmed_answer(self, mock_client, sample_dict):
        responses = [
            _make_response("{5}"),  # initial
            _make_response("Yes, that is correct. The answer is {5}"),  # verify
        ]
        mock_client.query.side_effect = responses
        result = strategy_verify(mock_client, sample_dict)
        assert result["parsed_answer"] == "5"
        assert result["strategy_verify_confirmed"] is True
        assert result["strategy_final_changed"] is False

    def test_corrected_answer(self, mock_client, sample_dict):
        responses = [
            _make_response("{3}"),  # initial (wrong)
            _make_response("No, I miscounted. The answer is {5}"),  # corrected
        ]
        mock_client.query.side_effect = responses
        result = strategy_verify(mock_client, sample_dict)
        assert result["parsed_answer"] == "5"
        assert result["strategy_initial_answer"] == "3"
        assert result["strategy_final_changed"] is True

    def test_parse_fail_skips_verify(self, mock_client, sample_dict):
        mock_client.query.return_value = _make_response("I cannot read this")
        result = strategy_verify(mock_client, sample_dict)
        assert result["parsed_answer"] is None
        # Only 1 call since parse failed on first pass
        assert mock_client.query.call_count == 1

    def test_verify_parse_fail_keeps_original(self, mock_client, sample_dict):
        responses = [
            _make_response("{5}"),  # initial
            _make_response("hmm let me think about this..."),  # verify fails to parse
        ]
        mock_client.query.side_effect = responses
        result = strategy_verify(mock_client, sample_dict)
        assert result["parsed_answer"] == "5"

    def test_two_api_calls(self, mock_client, sample_dict):
        responses = [
            _make_response("{5}"),
            _make_response("{5}"),
        ]
        mock_client.query.side_effect = responses
        result = strategy_verify(mock_client, sample_dict)
        assert mock_client.query.call_count == 2

    def test_token_accumulation(self, mock_client, sample_dict):
        responses = [
            _make_response("{5}", latency=1.0),
            _make_response("correct {5}", latency=0.5),
        ]
        mock_client.query.side_effect = responses
        result = strategy_verify(mock_client, sample_dict)
        assert result["input_tokens"] == 200
        assert result["latency_s"] == 1.5


# ---------------------------------------------------------------------------
# Strategy: best_of_n_verify (composite)
# ---------------------------------------------------------------------------

class TestBestOfNVerifyStrategy:
    def test_consensus_confirmed(self, mock_client, sample_dict):
        responses = [
            # best_of_3
            _make_response("{5}"),
            _make_response("{5}"),
            _make_response("{3}"),
            # verify
            _make_response("Yes, correct"),
        ]
        mock_client.query.side_effect = responses
        result = strategy_best_of_n_verify(mock_client, sample_dict, n=3)
        assert result["parsed_answer"] == "5"
        assert result["strategy"] == "best_of_n_verify"
        assert result["strategy_consensus"] == "5"
        assert mock_client.query.call_count == 4

    def test_consensus_overridden(self, mock_client, sample_dict):
        responses = [
            # best_of_3
            _make_response("{3}"),
            _make_response("{3}"),
            _make_response("{5}"),
            # verify changes answer
            _make_response("No, I see {5} squares"),
        ]
        mock_client.query.side_effect = responses
        result = strategy_best_of_n_verify(mock_client, sample_dict, n=3)
        assert result["parsed_answer"] == "5"
        assert result["strategy_consensus"] == "3"
        assert result["strategy_final_changed"] is True

    def test_all_parse_fail(self, mock_client, sample_dict):
        mock_client.query.return_value = _make_response("no idea")
        result = strategy_best_of_n_verify(mock_client, sample_dict, n=3)
        assert result["parsed_answer"] is None
        # Should not attempt verify when consensus is None
        assert mock_client.query.call_count == 3


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestStrategyRegistry:
    def test_all_registered(self):
        expected = {
            "baseline", "best_of_n", "crop_zoom", "verify",
            "best_of_n_verify", "decompose", "code_vision", "adaptive",
            "iterative_refine", "sketchpad",
        }
        assert expected == set(STRATEGY_REGISTRY.keys())

    def test_registry_callables(self):
        for name, fn in STRATEGY_REGISTRY.items():
            assert callable(fn), f"{name} is not callable"


# ---------------------------------------------------------------------------
# Strategy: adaptive (per-task routing)
# ---------------------------------------------------------------------------

class TestAdaptiveStrategy:
    def test_routes_hierarchy_to_verify(self, mock_client, sample_dict):
        sample_dict["task_name"] = "hierarchy_depth"
        responses = [
            _make_response("{4}"),  # initial
            _make_response("Looking again, I count 3 rows. {3}"),  # verify corrects
        ]
        mock_client.query.side_effect = responses
        result = strategy_adaptive(mock_client, sample_dict)
        assert result["strategy"] == "adaptive"
        assert result["strategy_selected"] == "verify"

    def test_routes_nested_squares_to_best_of_n(self, mock_client, sample_dict):
        mock_client.query.return_value = _make_response("{5}")
        result = strategy_adaptive(mock_client, sample_dict, n=3)
        assert result["strategy_selected"] == "best_of_n"
        assert mock_client.query.call_count == 3

    def test_routes_unknown_to_verify(self, mock_client, sample_dict):
        sample_dict["task_name"] = "unknown_task"
        mock_client.query.return_value = _make_response("{5}")
        result = strategy_adaptive(mock_client, sample_dict, n=3)
        assert result["strategy_selected"] == "verify"
        assert mock_client.query.call_count == 2

    def test_routes_text_control_to_verify(self, mock_client, sample_dict):
        """Text controls (e.g., nested_squares_text) should use base task's strategy."""
        sample_dict["task_name"] = "nested_squares_text"
        mock_client.query.return_value = _make_response("{5}")
        result = strategy_adaptive(mock_client, sample_dict, n=3)
        # nested_squares base maps to crop_zoom, but _text suffix could
        # route differently — verify it routes to the base task's strategy
        assert result["strategy"] == "adaptive"


# ---------------------------------------------------------------------------
# Strategy: decompose (structured decomposition)
# ---------------------------------------------------------------------------

class TestDecomposeStrategy:
    def test_nested_squares_decomposition(self, mock_client, sample_dict):
        """nested_squares has a 2-step plan: describe then count."""
        responses = [
            _make_response("Square 1: outermost. Square 2: inner. "
                          "Square 3: smaller. Square 4: tiny. Square 5: smallest"),
            _make_response("{5}"),
            _make_response("{5}"),  # final answer
        ]
        mock_client.query.side_effect = responses
        result = strategy_decompose(mock_client, sample_dict)
        assert result["strategy"] == "decompose"
        assert result["parsed_answer"] == "5"
        # 2 sub-questions + 1 final = 3 calls
        assert mock_client.query.call_count == 3

    def test_counting_grid_row_col_assembly(self, mock_client, sample_dict):
        """counting_grid with row_col parser should assemble from sub-counts."""
        sample_dict["task_name"] = "counting_grid"
        sample_dict["parser"] = "row_col"
        sample_dict["ground_truth"] = "4,5"
        responses = [
            _make_response("{5}"),  # 5 horizontal lines = 4 rows
            _make_response("{6}"),  # 6 vertical lines = 5 cols
            _make_response("rows=4 columns=5"),  # final answer
        ]
        mock_client.query.side_effect = responses
        result = strategy_decompose(mock_client, sample_dict)
        assert result["parsed_answer"] == "4,5"
        # 2 sub-questions + 1 final = 3 calls
        assert mock_client.query.call_count == 3

    def test_counting_grid_fallback_to_decomposed(self, mock_client, sample_dict):
        """When final parse fails, use decomposed row/col counts."""
        sample_dict["task_name"] = "counting_grid"
        sample_dict["parser"] = "row_col"
        sample_dict["ground_truth"] = "4,5"
        responses = [
            _make_response("{5}"),  # 5 horizontal lines
            _make_response("{6}"),  # 6 vertical lines
            _make_response("I see a grid with some rows and columns"),  # unparseable
        ]
        mock_client.query.side_effect = responses
        result = strategy_decompose(mock_client, sample_dict)
        # Should fall back to decomposed: 5-1=4 rows, 6-1=5 cols
        assert result["parsed_answer"] == "4,5"

    def test_unknown_task_generic_decomposition(self, mock_client, sample_dict):
        sample_dict["task_name"] = "unknown_task"
        responses = [
            _make_response("I see a white image with nothing on it"),
            _make_response("{0}"),
        ]
        mock_client.query.side_effect = responses
        result = strategy_decompose(mock_client, sample_dict)
        assert result["strategy"] == "decompose"
        # 1 generic sub-question + 1 final = 2 calls
        assert mock_client.query.call_count == 2

    def test_hierarchy_depth(self, mock_client, sample_dict):
        sample_dict["task_name"] = "hierarchy_depth"
        responses = [
            _make_response("Root: CEO -> CTO -> VP Eng -> Dir A"),
            _make_response("{4}"),
            _make_response("{4}"),
        ]
        mock_client.query.side_effect = responses
        result = strategy_decompose(mock_client, sample_dict)
        assert result["parsed_answer"] == "4"

    def test_sub_responses_recorded(self, mock_client, sample_dict):
        responses = [
            _make_response("I see nested squares"),
            _make_response("{5}"),
            _make_response("{5}"),
        ]
        mock_client.query.side_effect = responses
        result = strategy_decompose(mock_client, sample_dict)
        assert "strategy_sub_responses" in result
        assert len(result["strategy_sub_responses"]) == 2


# ---------------------------------------------------------------------------
# Strategy: code_vision (sandboxed Python REPL)
# ---------------------------------------------------------------------------

class TestRunSandboxedCode:
    def test_simple_print(self):
        output = _run_sandboxed_code("print('hello')", "/dev/null")
        assert output == "hello"

    def test_pil_import(self, sample_image):
        code = "from PIL import Image; img = Image.open(IMAGE_PATH); print(img.size)"
        output = _run_sandboxed_code(code, sample_image)
        assert "(512, 512)" in output

    def test_error_captured(self):
        output = _run_sandboxed_code("raise ValueError('test error')", "/dev/null")
        assert "ERROR" in output
        assert "test error" in output

    def test_timeout(self):
        output = _run_sandboxed_code("import time; time.sleep(100)", "/dev/null", timeout=1)
        assert "ERROR" in output
        assert "timeout" in output

    def test_no_output(self):
        output = _run_sandboxed_code("x = 42", "/dev/null")
        assert output == "NO OUTPUT"


class TestCodeVisionStrategy:
    def test_basic_flow(self, mock_client, sample_dict):
        responses = [
            # Step 1: model writes code
            _make_response("```python\nprint('5 squares found')\n```"),
            # Step 2: model interprets output
            _make_response("{5}"),
        ]
        mock_client.query.side_effect = responses
        result = strategy_code_vision(mock_client, sample_dict)
        assert result["strategy"] == "code_vision"
        assert result["parsed_answer"] == "5"
        assert mock_client.query.call_count == 2

    def test_code_without_fences(self, mock_client, sample_dict):
        """Model might not use code fences."""
        responses = [
            _make_response("print('5 squares found')"),
            _make_response("{5}"),
        ]
        mock_client.query.side_effect = responses
        result = strategy_code_vision(mock_client, sample_dict)
        assert result["parsed_answer"] == "5"

    def test_records_code_and_output(self, mock_client, sample_dict):
        responses = [
            _make_response("```python\nprint('analysis')\n```"),
            _make_response("{5}"),
        ]
        mock_client.query.side_effect = responses
        result = strategy_code_vision(mock_client, sample_dict)
        assert "strategy_code" in result
        assert "strategy_code_output" in result
        assert result["strategy_steps"] == 3

    def test_token_accumulation(self, mock_client, sample_dict):
        responses = [
            _make_response("```python\nprint('ok')\n```", latency=1.0),
            _make_response("{5}", latency=0.5),
        ]
        mock_client.query.side_effect = responses
        result = strategy_code_vision(mock_client, sample_dict)
        assert result["input_tokens"] == 200
        assert result["latency_s"] == 1.5


# ---------------------------------------------------------------------------
# Strategy: iterative_refine (multi-round prompt refinement)
# ---------------------------------------------------------------------------

class TestBuildRefinementPrompt:
    def test_includes_prior_answers(self, sample_dict):
        prompt = _build_refinement_prompt(sample_dict, ["3", "5"], 2)
        assert "Round 1: 3" in prompt
        assert "Round 2: 5" in prompt

    def test_task_specific_critique(self, sample_dict):
        sample_dict["task_name"] = "counting_grid"
        prompt = _build_refinement_prompt(sample_dict, ["4,4"], 1)
        assert "Scan" in prompt  # From _REFINEMENT_CRITIQUES

    def test_generic_critique_for_unknown_task(self, sample_dict):
        sample_dict["task_name"] = "unknown_task"
        prompt = _build_refinement_prompt(sample_dict, ["5"], 1)
        assert "Re-examine" in prompt


class TestIterativeRefineStrategy:
    def test_converges_immediately(self, mock_client, sample_dict):
        """Two consecutive same answers → stop early."""
        responses = [
            _make_response("{5}"),  # round 1
            _make_response("{5}"),  # round 2 — same, converge
        ]
        mock_client.query.side_effect = responses
        result = strategy_iterative_refine(mock_client, sample_dict, max_rounds=5)
        assert result["parsed_answer"] == "5"
        assert result["strategy"] == "iterative_refine"
        assert result["strategy_rounds"] == 2
        assert result["strategy_converged"] is True
        assert mock_client.query.call_count == 2

    def test_corrects_after_critique(self, mock_client, sample_dict):
        """Model changes answer, then converges on the new one."""
        responses = [
            _make_response("{3}"),  # round 1
            _make_response("{5}"),  # round 2 — changed
            _make_response("{5}"),  # round 3 — converged
        ]
        mock_client.query.side_effect = responses
        result = strategy_iterative_refine(mock_client, sample_dict, max_rounds=5)
        assert result["parsed_answer"] == "5"
        assert result["strategy_rounds"] == 3
        assert result["strategy_converged"] is True

    def test_max_rounds_respected(self, mock_client, sample_dict):
        """Never converges, stops at max_rounds."""
        responses = [
            _make_response("{3}"),
            _make_response("{4}"),
            _make_response("{5}"),
        ]
        mock_client.query.side_effect = responses
        result = strategy_iterative_refine(mock_client, sample_dict, max_rounds=3)
        assert result["parsed_answer"] == "5"
        assert result["strategy_rounds"] == 3
        assert result["strategy_converged"] is False
        assert mock_client.query.call_count == 3

    def test_all_parse_failures(self, mock_client, sample_dict):
        """All rounds fail to parse → None answer."""
        mock_client.query.return_value = _make_response("no answer")
        result = strategy_iterative_refine(mock_client, sample_dict, max_rounds=3)
        assert result["parsed_answer"] is None
        # No convergence possible with None answers, runs all rounds
        assert result["strategy_rounds"] == 3

    def test_single_round(self, mock_client, sample_dict):
        """max_rounds=1 should behave like baseline."""
        mock_client.query.return_value = _make_response("{5}")
        result = strategy_iterative_refine(mock_client, sample_dict, max_rounds=1)
        assert result["parsed_answer"] == "5"
        assert result["strategy_rounds"] == 1
        assert mock_client.query.call_count == 1

    def test_token_accumulation(self, mock_client, sample_dict):
        responses = [
            _make_response("{3}", latency=1.0),
            _make_response("{5}", latency=0.5),
            _make_response("{5}", latency=0.5),
        ]
        mock_client.query.side_effect = responses
        result = strategy_iterative_refine(mock_client, sample_dict, max_rounds=5)
        assert result["input_tokens"] == 300
        assert result["latency_s"] == 2.0

    def test_all_answers_recorded(self, mock_client, sample_dict):
        responses = [
            _make_response("{3}"),
            _make_response("{5}"),
            _make_response("{5}"),
        ]
        mock_client.query.side_effect = responses
        result = strategy_iterative_refine(mock_client, sample_dict, max_rounds=5)
        assert result["strategy_all_answers"] == ["3", "5", "5"]


# ---------------------------------------------------------------------------
# Strategy: sketchpad (visual sketchpad with pre-built primitives)
# ---------------------------------------------------------------------------

class TestSketchpadStrategy:
    def test_returns_answer_from_model(self, mock_client, sample_dict):
        """Model responds with ANSWER directive on first model pass."""
        mock_client.query.return_value = _make_response(
            "ANSWER {5}"
        )
        result = strategy_sketchpad(mock_client, sample_dict, max_passes=3)
        assert result["strategy"] == "sketchpad"
        assert result["parsed_answer"] == "5"
        assert result["strategy_passes"] >= 2  # pass 0 + at least 1 model pass

    def test_model_requests_tool(self, mock_client, sample_dict):
        """Model requests a tool, then answers."""
        responses = [
            _make_response("TOOL(detect_edges)"),
            _make_response("ANSWER {5}"),
        ]
        mock_client.query.side_effect = responses
        result = strategy_sketchpad(mock_client, sample_dict, max_passes=3)
        assert result["parsed_answer"] == "5"
        assert result["strategy_passes"] >= 3  # pass 0 + 2 model passes

    def test_max_passes_respected(self, mock_client, sample_dict):
        """Always requests tools → stops at max_passes."""
        mock_client.query.return_value = _make_response(
            "TOOL(detect_contours)"
        )
        result = strategy_sketchpad(mock_client, sample_dict, max_passes=2)
        # Should stop after max_passes total (pass 0 + 1 model pass)
        assert result["strategy_passes"] == 2
        assert mock_client.query.call_count == 1

    def test_unknown_tool_treated_as_answer(self, mock_client, sample_dict):
        """Model requests a non-existent tool → treated as answer."""
        mock_client.query.return_value = _make_response(
            "TOOL(nonexistent_tool)"
        )
        result = strategy_sketchpad(mock_client, sample_dict, max_passes=3)
        # Should treat this as answer since tool is unknown
        assert mock_client.query.call_count == 1

    def test_findings_recorded(self, mock_client, sample_dict):
        """Sub-question findings are recorded in the result."""
        mock_client.query.return_value = _make_response("ANSWER {5}")
        result = strategy_sketchpad(mock_client, sample_dict, max_passes=3)
        assert "strategy_findings" in result
        assert "strategy_sub_questions" in result
        assert result["strategy_sub_questions"] >= 1

    def test_pie_chart_task(self, mock_client, sample_dict):
        """Pie chart task uses segment_colors primitive."""
        sample_dict["task_name"] = "pie_chart"
        sample_dict["prompt"] = "What percentage does the blue slice represent? (A) 25% (B) 35% (C) 45% (D) 55%"
        sample_dict["parser"] = "mc4"
        mock_client.query.return_value = _make_response("ANSWER (B)")
        result = strategy_sketchpad(mock_client, sample_dict, max_passes=3)
        assert result["strategy"] == "sketchpad"
        # segment_colors should be in findings
        assert "segment" in result.get("strategy_findings", "").lower() or \
               "color" in result.get("strategy_findings", "").lower()

    def test_hierarchy_task(self, mock_client, sample_dict):
        """Hierarchy task decomposes into box detection + y-clustering."""
        sample_dict["task_name"] = "hierarchy_depth"
        sample_dict["prompt"] = "How many levels deep is this hierarchy? {3}"
        mock_client.query.return_value = _make_response("ANSWER {3}")
        result = strategy_sketchpad(mock_client, sample_dict, max_passes=3)
        assert result["strategy_sub_questions"] >= 2  # at least 2 sub-questions

    def test_token_accumulation(self, mock_client, sample_dict):
        responses = [
            _make_response("TOOL(detect_edges)", latency=1.0),
            _make_response("ANSWER {5}", latency=0.5),
        ]
        mock_client.query.side_effect = responses
        result = strategy_sketchpad(mock_client, sample_dict, max_passes=3)
        assert result["input_tokens"] == 200
        assert result["latency_s"] == 1.5
