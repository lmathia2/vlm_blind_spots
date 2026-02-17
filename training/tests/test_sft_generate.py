"""Tests for training.sft_generate — data generation and anti-shortcut."""

from random import Random

import pytest

from training.sft_generate import (
    _BG_COLORS,
    _GRID_RANGES,
    _apply_anti_shortcut,
    _get_grid_pairs,
    _sample_grid_params,
    _select_template,
    generate_one_sample,
    generate_sft_dataset,
)


# ---------------------------------------------------------------------------
# _get_grid_pairs
# ---------------------------------------------------------------------------

class TestGetGridPairs:
    def test_direct_range(self):
        pairs = _get_grid_pairs("direct")
        lo, hi = _GRID_RANGES["direct"]
        expected = (hi - lo + 1) ** 2
        assert len(pairs) == expected

    def test_tool_use_skip(self):
        pairs = _get_grid_pairs("tool_use", is_skip=True)
        lo, hi = _GRID_RANGES["tool_use_skip"]
        expected = (hi - lo + 1) ** 2
        assert len(pairs) == expected

    def test_all_pairs_in_range(self):
        pairs = _get_grid_pairs("direct")
        lo, hi = _GRID_RANGES["direct"]
        for r, c in pairs:
            assert lo <= r <= hi
            assert lo <= c <= hi

    def test_caching(self):
        p1 = _get_grid_pairs("intermediate_repr")
        p2 = _get_grid_pairs("intermediate_repr")
        assert p1 is p2  # same object (cached)


# ---------------------------------------------------------------------------
# _sample_grid_params
# ---------------------------------------------------------------------------

class TestSampleGridParams:
    def test_returns_required_keys(self):
        params = _sample_grid_params(Random(42), "direct")
        for key in ("rows", "cols", "resolution", "line_width",
                     "padding", "bg_color", "aspect_stretch"):
            assert key in params

    def test_rows_cols_in_range(self):
        rng = Random(42)
        lo, hi = _GRID_RANGES["direct"]
        for _ in range(50):
            params = _sample_grid_params(rng, "direct")
            assert lo <= params["rows"] <= hi
            assert lo <= params["cols"] <= hi

    def test_padding_range(self):
        rng = Random(42)
        for _ in range(50):
            params = _sample_grid_params(rng, "direct")
            for p in params["padding"]:
                assert 0 <= p <= 30

    def test_bg_color_from_palette(self):
        rng = Random(42)
        for _ in range(50):
            params = _sample_grid_params(rng, "direct")
            assert params["bg_color"] in _BG_COLORS

    def test_aspect_stretch_range(self):
        rng = Random(42)
        for _ in range(50):
            params = _sample_grid_params(rng, "direct")
            w, h = params["aspect_stretch"]
            assert 0.85 <= w <= 1.15
            assert 0.85 <= h <= 1.15

    def test_uniform_distribution(self):
        """All (rows, cols) pairs should appear with roughly equal probability."""
        rng = Random(42)
        from collections import Counter
        counter = Counter()
        n = 1000
        for _ in range(n):
            params = _sample_grid_params(rng, "direct")
            counter[(params["rows"], params["cols"])] += 1

        lo, hi = _GRID_RANGES["direct"]
        n_pairs = (hi - lo + 1) ** 2
        expected_per_pair = n / n_pairs
        # No pair should be more than 4x expected (loose bound)
        for count in counter.values():
            assert count < expected_per_pair * 4

    def test_high_resolution_for_dense_grids(self):
        rng = Random(42)
        for _ in range(20):
            params = _sample_grid_params(rng, "tool_use")
            if max(params["rows"], params["cols"]) > 15:
                assert params["resolution"] in [512, 768, 1024]

    def test_skip_uses_small_range(self):
        rng = Random(42)
        lo, hi = _GRID_RANGES["tool_use_skip"]
        for _ in range(20):
            params = _sample_grid_params(rng, "tool_use", is_skip=True)
            assert lo <= params["rows"] <= hi
            assert lo <= params["cols"] <= hi


# ---------------------------------------------------------------------------
# _select_template
# ---------------------------------------------------------------------------

class TestSelectTemplate:
    def test_direct(self):
        from training.cot_templates import DIRECT_COT_TEMPLATES
        t = _select_template(Random(42), "direct")
        assert t in DIRECT_COT_TEMPLATES

    def test_intermediate_repr(self):
        from training.cot_templates import INTERMEDIATE_COT_TEMPLATES
        t = _select_template(Random(42), "intermediate_repr")
        assert t in INTERMEDIATE_COT_TEMPLATES

    def test_tool_use(self):
        from training.cot_templates import TOOL_USE_COT_TEMPLATES
        t = _select_template(Random(42), "tool_use")
        assert t in TOOL_USE_COT_TEMPLATES

    def test_tool_use_skip(self):
        from training.cot_templates import TOOL_USE_SKIP_TEMPLATES
        t = _select_template(Random(42), "tool_use", is_skip=True)
        assert t in TOOL_USE_SKIP_TEMPLATES

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            _select_template(Random(42), "unknown")


# ---------------------------------------------------------------------------
# _apply_anti_shortcut
# ---------------------------------------------------------------------------

class TestApplyAntiShortcut:
    def _make_img(self, w=100, h=100):
        from PIL import Image
        return Image.new("RGB", (w, h), "white")

    def test_padding_increases_size(self):
        img = self._make_img()
        params = {
            "padding": (10, 10, 10, 10),
            "bg_color": (255, 255, 255),
            "aspect_stretch": (1.0, 1.0),
        }
        result = _apply_anti_shortcut(img, params)
        assert result.width == 120
        assert result.height == 120

    def test_no_padding(self):
        img = self._make_img()
        params = {
            "padding": (0, 0, 0, 0),
            "bg_color": (255, 255, 255),
            "aspect_stretch": (1.0, 1.0),
        }
        result = _apply_anti_shortcut(img, params)
        assert result.size == (100, 100)

    def test_stretch_changes_size(self):
        img = self._make_img(200, 200)
        params = {
            "padding": (0, 0, 0, 0),
            "bg_color": (255, 255, 255),
            "aspect_stretch": (1.1, 0.9),
        }
        result = _apply_anti_shortcut(img, params)
        assert result.width == 220
        assert result.height == 180

    def test_asymmetric_padding(self):
        img = self._make_img()
        params = {
            "padding": (5, 10, 15, 20),  # top, right, bottom, left
            "bg_color": (240, 240, 240),
            "aspect_stretch": (1.0, 1.0),
        }
        result = _apply_anti_shortcut(img, params)
        assert result.width == 100 + 10 + 20  # right + left
        assert result.height == 100 + 5 + 15  # top + bottom


# ---------------------------------------------------------------------------
# generate_one_sample
# ---------------------------------------------------------------------------

class TestGenerateOneSample:
    def test_returns_required_keys(self):
        sample = generate_one_sample(999_000, "direct", Random(999_000))
        for key in ("image", "image_base64", "prompt", "chain_of_thought",
                     "answer", "ground_truth", "strategy", "seed", "metadata"):
            assert key in sample

    def test_ground_truth_format(self):
        sample = generate_one_sample(999_000, "direct", Random(999_000))
        parts = sample["ground_truth"].split(",")
        assert len(parts) == 2
        assert all(p.isdigit() for p in parts)

    def test_answer_format(self):
        sample = generate_one_sample(999_000, "direct", Random(999_000))
        assert "rows=" in sample["answer"]
        assert "columns=" in sample["answer"]

    def test_metadata_has_anti_shortcut_params(self):
        sample = generate_one_sample(999_000, "direct", Random(999_000))
        for key in ("padding", "bg_color", "aspect_stretch"):
            assert key in sample["metadata"]

    def test_tool_use_skip_for_early_seeds(self):
        # Seed 40_000 (offset 0 within tool_use) → is_skip=True
        sample = generate_one_sample(40_000, "tool_use", Random(40_000))
        assert sample["is_skip"] is True

    def test_tool_use_no_skip_for_later_seeds(self):
        # Seed 40_200+ → is_skip=False
        sample = generate_one_sample(40_200, "tool_use", Random(40_200))
        assert sample["is_skip"] is False

    def test_deterministic(self):
        s1 = generate_one_sample(999_000, "direct", Random(999_000))
        s2 = generate_one_sample(999_000, "direct", Random(999_000))
        assert s1["ground_truth"] == s2["ground_truth"]
        assert s1["chain_of_thought"] == s2["chain_of_thought"]

    def test_image_is_pil(self):
        from PIL import Image
        sample = generate_one_sample(999_000, "direct", Random(999_000))
        assert isinstance(sample["image"], Image.Image)


# ---------------------------------------------------------------------------
# generate_sft_dataset
# ---------------------------------------------------------------------------

class TestGenerateSftDataset:
    def test_generates_files(self, tmp_path):
        import json
        jsonl_path = generate_sft_dataset(tmp_path, "direct", n_samples=5)
        assert jsonl_path.exists()

        # Check JSONL has 5 lines
        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 5

        # Check each line is valid JSON
        for line in lines:
            record = json.loads(line)
            assert "ground_truth" in record
            assert "chain_of_thought" in record

    def test_images_saved(self, tmp_path):
        generate_sft_dataset(tmp_path, "direct", n_samples=3)
        img_dir = tmp_path / "direct" / "images"
        assert img_dir.exists()
        pngs = list(img_dir.glob("*.png"))
        assert len(pngs) == 3
