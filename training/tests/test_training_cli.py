"""Tests for training.cli — CLI subcommands."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def run_cli(*args, check=True):
    """Run training CLI as subprocess, return CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", "training.cli", *args],
        capture_output=True,
        text=True,
        check=check,
    )


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_generate_small(self, tmp_path):
        result = run_cli("generate", "--strategy", "direct",
                         "--n", "3", "--output", str(tmp_path))
        assert result.returncode == 0
        jsonl = tmp_path / "direct" / "samples.jsonl"
        assert jsonl.exists()
        with open(jsonl) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_generate_images(self, tmp_path):
        run_cli("generate", "--strategy", "direct",
                "--n", "2", "--output", str(tmp_path))
        img_dir = tmp_path / "direct" / "images"
        pngs = list(img_dir.glob("*.png"))
        assert len(pngs) == 2


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

class TestVerify:
    def test_verify_prints_samples(self):
        result = run_cli("verify", "--strategy", "direct", "--n", "2")
        assert result.returncode == 0
        assert "Strategy: direct" in result.stdout
        assert "Sample 1" in result.stdout
        assert "Sample 2" in result.stdout

    def test_verify_all(self):
        result = run_cli("verify", "--strategy", "all", "--n", "1")
        assert result.returncode == 0
        assert "direct" in result.stdout
        assert "intermediate_repr" in result.stdout
        assert "tool_use" in result.stdout


# ---------------------------------------------------------------------------
# verify-reward
# ---------------------------------------------------------------------------

class TestVerifyReward:
    def test_verify_reward_fresh(self):
        result = run_cli("verify-reward", "--n", "5")
        assert result.returncode == 0
        assert "All rewards returned 1.0" in result.stdout

    def test_verify_reward_from_jsonl(self, tmp_path):
        # Generate, then verify from file
        run_cli("generate", "--strategy", "direct",
                "--n", "5", "--output", str(tmp_path))
        jsonl = str(tmp_path / "direct" / "samples.jsonl")
        result = run_cli("verify-reward", "--jsonl", jsonl, "--n", "3")
        assert result.returncode == 0
        assert "Checked 3 samples" in result.stdout

    def test_verify_reward_missing_file(self):
        result = run_cli("verify-reward", "--jsonl", "/nonexistent.jsonl",
                         check=False)
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------

class TestDiagnose:
    def _write_episodes(self, path, episodes):
        with open(path, "w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")

    def test_diagnose_basic(self, tmp_path):
        jsonl = tmp_path / "episodes.jsonl"
        episodes = [
            {
                "response": f"rows={r} columns={c}",
                "ground_truth": f"{r},{c}",
                "metadata": {"rows": r, "cols": c},
                "reward": 1.0,
            }
            for r in range(3, 8) for c in range(3, 8)
        ]
        self._write_episodes(jsonl, episodes)

        result = run_cli("diagnose", "--results", str(jsonl))
        assert result.returncode == 0
        assert "REWARD HACKING DIAGNOSTICS" in result.stdout

    def test_diagnose_with_n_limit(self, tmp_path):
        jsonl = tmp_path / "episodes.jsonl"
        episodes = [
            {
                "response": f"rows=5 columns=5",
                "ground_truth": "5,5",
                "metadata": {"rows": 5, "cols": 5},
                "reward": 1.0,
            }
            for _ in range(20)
        ]
        self._write_episodes(jsonl, episodes)

        result = run_cli("diagnose", "--results", str(jsonl), "--n", "5")
        assert result.returncode == 0
        assert "Loaded 5 episodes" in result.stdout

    def test_diagnose_json_output(self, tmp_path):
        jsonl = tmp_path / "episodes.jsonl"
        json_out = tmp_path / "report.json"
        episodes = [
            {
                "response": "rows=5 columns=5",
                "ground_truth": "5,5",
                "metadata": {"rows": 5, "cols": 5},
                "reward": 1.0,
            }
        ]
        self._write_episodes(jsonl, episodes)

        result = run_cli("diagnose", "--results", str(jsonl),
                         "--json", str(json_out))
        assert result.returncode == 0
        assert json_out.exists()
        with open(json_out) as f:
            report = json.load(f)
        assert "flags" in report

    def test_diagnose_missing_file(self):
        result = run_cli("diagnose", "--results", "/nonexistent.jsonl",
                         check=False)
        assert result.returncode != 0

    def test_diagnose_cot_answer_format(self, tmp_path):
        """Test that chain_of_thought + answer fields are combined."""
        jsonl = tmp_path / "episodes.jsonl"
        episodes = [
            {
                "chain_of_thought": "I count 6 lines.",
                "answer": "rows=5 columns=5",
                "ground_truth": "5,5",
                "metadata": {"rows": 5, "cols": 5},
                "reward": 1.0,
            }
        ]
        self._write_episodes(jsonl, episodes)

        result = run_cli("diagnose", "--results", str(jsonl))
        assert result.returncode == 0
