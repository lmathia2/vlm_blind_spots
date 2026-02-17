"""Tests for config.py — configuration constants."""

from pathlib import Path

import config


class TestConfig:
    def test_model_is_string(self):
        assert isinstance(config.MODEL, str)

    def test_temperature_zero(self):
        assert config.TEMPERATURE == 0.0

    def test_max_tokens_positive(self):
        assert config.MAX_TOKENS > 0

    def test_max_workers_positive(self):
        assert config.MAX_WORKERS > 0

    def test_project_root_exists(self):
        assert config.PROJECT_ROOT.exists()

    def test_data_dir_exists(self):
        assert config.DATA_DIR.exists()

    def test_results_dir_exists(self):
        assert config.RESULTS_DIR.exists()

    def test_paths_are_absolute(self):
        assert config.PROJECT_ROOT.is_absolute()
        assert config.DATA_DIR.is_absolute()
        assert config.RESULTS_DIR.is_absolute()
