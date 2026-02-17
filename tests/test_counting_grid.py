"""Tests for tasks/counting_grid.py — grid renderer."""

from PIL import Image

from tasks.counting_grid import render, _generate_merges, TASK_CONFIG


class TestRender:
    def test_returns_tuple(self):
        img, gt, meta = render(rows=4, cols=4, seed=42)
        assert isinstance(img, Image.Image)
        assert isinstance(gt, str)
        assert isinstance(meta, dict)

    def test_grid_size_question(self):
        img, gt, meta = render(rows=5, cols=8, question_type="grid_size", seed=42)
        assert gt == "5,8"
        assert meta["parser"] == "row_col"

    def test_total_cells_no_merges(self):
        img, gt, meta = render(rows=4, cols=4, n_merged=0,
                                question_type="total_cells", seed=42)
        assert gt == "16"

    def test_merged_count(self):
        img, gt, meta = render(rows=8, cols=8, n_merged=3,
                                question_type="merged_count", seed=42)
        assert int(gt) <= 3  # might get fewer if placement fails

    def test_image_dimensions(self):
        img, _, _ = render(rows=4, cols=4, resolution=512, seed=42)
        assert img.size == (512, 512)

    def test_deterministic(self):
        img1, gt1, _ = render(rows=5, cols=5, seed=99)
        img2, gt2, _ = render(rows=5, cols=5, seed=99)
        assert gt1 == gt2
        assert list(img1.getdata()) == list(img2.getdata())

    def test_metadata_fields(self):
        _, _, meta = render(rows=6, cols=8, resolution=384, line_width=2,
                            n_merged=0, question_type="grid_size", seed=42)
        assert meta["rows"] == 6
        assert meta["cols"] == 8
        assert meta["resolution"] == 384
        assert meta["line_width"] == 2
        assert meta["question_type"] == "grid_size"

    def test_total_cells_with_merges(self):
        _, gt, meta = render(rows=8, cols=8, n_merged=3,
                              question_type="total_cells", seed=42)
        total = int(gt)
        # total cells <= 64, reduced by merges
        assert total <= 64
        assert total == meta["total_cells"]


class TestGenerateMerges:
    def test_returns_list(self):
        from random import Random
        merges = _generate_merges(8, 8, 3, Random(42))
        assert isinstance(merges, list)

    def test_respects_count(self):
        from random import Random
        merges = _generate_merges(8, 8, 5, Random(42))
        assert len(merges) <= 5

    def test_no_merges(self):
        from random import Random
        merges = _generate_merges(8, 8, 0, Random(42))
        assert merges == []

    def test_non_overlapping(self):
        from random import Random
        merges = _generate_merges(8, 8, 5, Random(42))
        occupied = set()
        for r, c, sr, sc in merges:
            cells = {(r + dr, c + dc) for dr in range(sr) for dc in range(sc)}
            assert not (cells & occupied), "Merges overlap"
            occupied |= cells

    def test_multi_cell(self):
        from random import Random
        merges = _generate_merges(8, 8, 3, Random(42))
        for _, _, sr, sc in merges:
            assert sr > 1 or sc > 1, "Merge must span >1 cell"


class TestTaskConfig:
    def test_required_keys(self):
        for key in ("task_name", "parser", "scorer", "default_params", "sweep_axes"):
            assert key in TASK_CONFIG

    def test_question_types(self):
        assert "grid_size" in TASK_CONFIG["sweep_axes"]["question_type"]
        assert "total_cells" in TASK_CONFIG["sweep_axes"]["question_type"]
        assert "merged_count" in TASK_CONFIG["sweep_axes"]["question_type"]
