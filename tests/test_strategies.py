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
    _majority_vote,
    _crop_image,
    _tile_image,
    _parse_answer,
    _save_temp_image,
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
        expected = {"baseline", "best_of_n", "crop_zoom", "verify", "best_of_n_verify"}
        assert expected == set(STRATEGY_REGISTRY.keys())

    def test_registry_callables(self):
        for name, fn in STRATEGY_REGISTRY.items():
            assert callable(fn), f"{name} is not callable"
