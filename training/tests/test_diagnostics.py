"""Tests for training.diagnostics — reward hacking monitors."""

from training.diagnostics import (
    check_answer_distribution,
    check_calibration,
    check_cot_consistency_rate,
    check_per_size_accuracy,
    check_tool_use_rate,
    print_diagnostic_report,
    run_all_diagnostics,
)


def _make_episode(response, gt, rows=None, cols=None, reward=1.0, confidence=None):
    """Helper to build an episode dict."""
    if rows is None:
        rows = int(gt.split(",")[0])
    if cols is None:
        cols = int(gt.split(",")[1])
    meta = {"rows": rows, "cols": cols}
    if confidence is not None:
        meta["confidence"] = confidence
    return {
        "response": response,
        "ground_truth": gt,
        "metadata": meta,
        "reward": reward,
    }


def _correct_episode(r, c):
    """Helper: episode with correct answer."""
    return _make_episode(f"rows={r} columns={c}", f"{r},{c}")


def _wrong_episode(r, c, pred_r, pred_c):
    """Helper: episode with wrong answer."""
    return _make_episode(f"rows={pred_r} columns={pred_c}", f"{r},{c}")


# ---------------------------------------------------------------------------
# check_answer_distribution
# ---------------------------------------------------------------------------

class TestCheckAnswerDistribution:
    def test_no_data(self):
        result = check_answer_distribution([])
        assert result["status"] == "no_data"

    def test_uniform_no_flag(self):
        episodes = [_correct_episode(r, c) for r in range(3, 8) for c in range(3, 8)]
        result = check_answer_distribution(episodes)
        assert not result["dominant_prediction_flag"]

    def test_dominant_prediction_flag(self):
        # 20 episodes all predicting 8,8
        episodes = [_wrong_episode(r, 5, 8, 8) for r in range(3, 23)]
        result = check_answer_distribution(episodes)
        assert result["dominant_prediction_flag"]
        assert result["most_common_prediction"] == "8,8"

    def test_kl_zero_when_perfect(self):
        episodes = [_correct_episode(r, c) for r in range(3, 6) for c in range(3, 6)]
        result = check_answer_distribution(episodes)
        assert result["kl_divergence"] == 0.0

    def test_unique_predictions_counted(self):
        episodes = [_correct_episode(r, c) for r in range(3, 6) for c in range(3, 6)]
        result = check_answer_distribution(episodes)
        assert result["n_unique_predictions"] == 9

    def test_unparsed_tracked(self):
        episodes = [_make_episode("no answer here", "5,5")]
        result = check_answer_distribution(episodes)
        assert "UNPARSED" in result["prediction_distribution"]


# ---------------------------------------------------------------------------
# check_per_size_accuracy
# ---------------------------------------------------------------------------

class TestCheckPerSizeAccuracy:
    def test_no_data(self):
        result = check_per_size_accuracy([])
        assert result["status"] == "no_data"

    def test_perfect_accuracy(self):
        episodes = [_correct_episode(5, 5) for _ in range(10)]
        result = check_per_size_accuracy(episodes)
        assert result["accuracy_by_size"]["5,5"]["accuracy"] == 1.0

    def test_zero_accuracy(self):
        episodes = [_wrong_episode(5, 5, 8, 8) for _ in range(10)]
        result = check_per_size_accuracy(episodes)
        assert result["accuracy_by_size"]["5,5"]["accuracy"] == 0.0

    def test_flat_prediction_flag(self):
        # All sizes at exactly 50% accuracy → std ≈ 0
        episodes = []
        for r in range(3, 9):
            for c in range(3, 9):
                # 3 correct + 3 wrong = 6 per size, 50% each
                for _ in range(3):
                    episodes.append(_correct_episode(r, c))
                    episodes.append(_wrong_episode(r, c, r + 1, c))
        result = check_per_size_accuracy(episodes)
        # std should be very low since all are 50%
        assert result["flat_prediction_flag"]

    def test_varied_accuracy_no_flag(self):
        # Easy grids 100%, hard grids 0%
        episodes = []
        for r in range(3, 6):
            for c in range(3, 6):
                episodes.extend([_correct_episode(r, c)] * 5)
        for r in range(10, 13):
            for c in range(10, 13):
                episodes.extend([_wrong_episode(r, c, r + 1, c)] * 5)
        result = check_per_size_accuracy(episodes)
        assert not result["flat_prediction_flag"]


# ---------------------------------------------------------------------------
# check_cot_consistency_rate
# ---------------------------------------------------------------------------

class TestCheckCotConsistencyRate:
    def test_no_data(self):
        result = check_cot_consistency_rate([])
        assert result["status"] == "no_data"

    def test_all_consistent(self):
        episodes = [
            _make_episode("6 lines → 5 rows. rows=5 columns=3", "5,3")
            for _ in range(10)
        ]
        result = check_cot_consistency_rate(episodes)
        assert result["consistency_rate"] == 1.0
        assert not result["low_consistency_flag"]

    def test_low_consistency_flag(self):
        # CoT says 5 but answer says 6 → inconsistent
        episodes = [
            _make_episode("6 lines → 5 rows. rows=6 columns=3", "5,3")
            for _ in range(10)
        ]
        result = check_cot_consistency_rate(episodes)
        assert result["consistency_rate"] < 0.7
        assert result["low_consistency_flag"]

    def test_subtraction_pattern_detected(self):
        episodes = [
            _make_episode("6 - 1 = 5 rows. rows=5 columns=3", "5,3")
        ]
        result = check_cot_consistency_rate(episodes)
        assert result["subtraction_pattern_rate"] == 1.0

    def test_no_subtraction_patterns(self):
        episodes = [_make_episode("rows=5 columns=3", "5,3")]
        result = check_cot_consistency_rate(episodes)
        assert result["subtraction_pattern_rate"] == 0.0


# ---------------------------------------------------------------------------
# check_tool_use_rate
# ---------------------------------------------------------------------------

class TestCheckToolUseRate:
    def test_no_tool_use(self):
        episodes = [_correct_episode(5, 5) for _ in range(10)]
        result = check_tool_use_rate(episodes)
        assert result["tool_use_by_bucket"]["3-8"]["rate"] == 0.0
        assert not result["easy_grid_overuse_flag"]

    def test_tool_overuse_flag(self):
        # All easy grids use code
        episodes = []
        for _ in range(10):
            resp = "```python\ncode\n```\nrows=5 columns=5"
            episodes.append(_make_episode(resp, "5,5"))
        result = check_tool_use_rate(episodes)
        assert result["tool_use_by_bucket"]["3-8"]["rate"] == 1.0
        assert result["easy_grid_overuse_flag"]

    def test_large_grid_tool_use_ok(self):
        # Tool use on large grids is fine
        episodes = []
        for _ in range(10):
            resp = "```python\ncode\n```\nrows=20 columns=20"
            episodes.append(_make_episode(resp, "20,20"))
        result = check_tool_use_rate(episodes)
        assert result["tool_use_by_bucket"]["19-25"]["rate"] == 1.0
        assert not result["easy_grid_overuse_flag"]

    def test_bucket_assignment(self):
        episodes = [
            _make_episode("rows=5 columns=5", "5,5"),      # 3-8
            _make_episode("rows=10 columns=10", "10,10"),   # 9-12
            _make_episode("rows=15 columns=15", "15,15"),   # 13-18
            _make_episode("rows=20 columns=20", "20,20"),   # 19-25
        ]
        result = check_tool_use_rate(episodes)
        for bucket in ["3-8", "9-12", "13-18", "19-25"]:
            assert result["tool_use_by_bucket"][bucket]["n"] == 1


# ---------------------------------------------------------------------------
# check_calibration
# ---------------------------------------------------------------------------

class TestCheckCalibration:
    def test_no_confidence_data(self):
        episodes = [_correct_episode(5, 5)]
        result = check_calibration(episodes)
        assert result["status"] == "no_confidence_data"

    def test_with_confidence(self):
        episodes = [
            _make_episode("rows=5 columns=5", "5,5", confidence=0.95),
            _make_episode("rows=5 columns=5", "5,5", confidence=0.95),
        ]
        result = check_calibration(episodes)
        assert result["n_with_confidence"] == 2
        assert "0.9-1.0" in result["accuracy_by_confidence_bin"]

    def test_overconfidence_flag(self):
        # High confidence but wrong answers
        episodes = [
            _make_episode("rows=8 columns=8", "5,5", confidence=0.95)
            for _ in range(10)
        ]
        result = check_calibration(episodes)
        assert result["overconfidence_flag"]

    def test_well_calibrated_no_flag(self):
        # High confidence, correct answers
        episodes = [
            _make_episode("rows=5 columns=5", "5,5", confidence=0.95)
            for _ in range(10)
        ]
        result = check_calibration(episodes)
        assert not result["overconfidence_flag"]


# ---------------------------------------------------------------------------
# run_all_diagnostics
# ---------------------------------------------------------------------------

class TestRunAllDiagnostics:
    def test_returns_all_sections(self):
        episodes = [_correct_episode(5, 5) for _ in range(10)]
        report = run_all_diagnostics(episodes)
        for key in ("answer_distribution", "per_size_accuracy",
                     "cot_consistency", "tool_use", "calibration",
                     "flags", "n_flags"):
            assert key in report

    def test_no_flags_on_clean_data(self):
        episodes = [_correct_episode(r, c)
                     for r in range(3, 8) for c in range(3, 8)]
        report = run_all_diagnostics(episodes)
        assert report["n_flags"] == 0

    def test_flags_on_gaming_data(self):
        # All predictions are 8,8
        episodes = [_wrong_episode(r, c, 8, 8)
                     for r in range(3, 23) for c in [5]]
        report = run_all_diagnostics(episodes)
        assert report["n_flags"] >= 1
        assert any("DOMINANT" in f for f in report["flags"])


# ---------------------------------------------------------------------------
# print_diagnostic_report
# ---------------------------------------------------------------------------

class TestPrintDiagnosticReport:
    def test_prints_without_error(self, capsys):
        episodes = [_correct_episode(r, c)
                     for r in range(3, 8) for c in range(3, 8)]
        report = run_all_diagnostics(episodes)
        print_diagnostic_report(report)
        captured = capsys.readouterr()
        assert "REWARD HACKING DIAGNOSTICS" in captured.out
        assert "No flags raised" in captured.out

    def test_prints_flags(self, capsys):
        episodes = [_wrong_episode(r, 5, 8, 8) for r in range(3, 23)]
        report = run_all_diagnostics(episodes)
        print_diagnostic_report(report)
        captured = capsys.readouterr()
        assert "FLAG(S) RAISED" in captured.out

    def test_prints_no_confidence(self, capsys):
        episodes = [_correct_episode(5, 5)]
        report = run_all_diagnostics(episodes)
        print_diagnostic_report(report)
        captured = capsys.readouterr()
        assert "no confidence data" in captured.out
