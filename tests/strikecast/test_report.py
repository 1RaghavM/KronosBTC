import json
from pathlib import Path

import pytest


def _make_score_results():
    from strikecast.eval.scoring import ScoreResult

    return [
        ScoreResult(
            estimator="randomwalk",
            moneyness_bucket="all",
            brier=0.2500,
            logloss=0.6931,
            ece=0.005,
            directional_accuracy=0.500,
            brier_skill_score=0.0,
            n_windows=100,
            ci_brier=(0.2400, 0.2600),
            ci_logloss=(0.6800, 0.7100),
        ),
        ScoreResult(
            estimator="garch_mc",
            moneyness_bucket="all",
            brier=0.2450,
            logloss=0.6850,
            ece=0.010,
            directional_accuracy=0.520,
            brier_skill_score=0.02,
            n_windows=100,
            ci_brier=(0.2350, 0.2550),
            ci_logloss=(0.6750, 0.7000),
        ),
    ]


class TestRunReport:
    def test_construction(self) -> None:
        from strikecast.eval.report import RunReport

        scores = _make_score_results()
        report = RunReport(
            run_id="abc123_2026-05-30T12:00:00",
            data_window=("2025-12-01", "2026-05-30"),
            model_checkpoint=None,
            git_commit="abc123",
            seed=42,
            scores=scores,
            kill_criterion_passed=None,
            timestamp="2026-05-30T12:00:00Z",
        )

        assert report.run_id == "abc123_2026-05-30T12:00:00"
        assert len(report.scores) == 2


class TestJSONReport:
    def test_valid_json(self) -> None:
        from strikecast.eval.report import RunReport, to_json

        scores = _make_score_results()
        report = RunReport(
            run_id="test_run",
            data_window=("2025-12-01", "2026-05-30"),
            model_checkpoint=None,
            git_commit="abc123",
            seed=42,
            scores=scores,
            kill_criterion_passed=None,
            timestamp="2026-05-30T12:00:00Z",
        )

        json_str = to_json(report)
        parsed = json.loads(json_str)

        assert parsed["run_id"] == "test_run"
        assert parsed["seed"] == 42
        assert len(parsed["scores"]) == 2
        assert parsed["scores"][0]["estimator"] == "randomwalk"

    def test_write_json_to_file(self, tmp_path: Path) -> None:
        from strikecast.eval.report import RunReport, write_report

        scores = _make_score_results()
        report = RunReport(
            run_id="test_run",
            data_window=("2025-12-01", "2026-05-30"),
            model_checkpoint=None,
            git_commit="abc123",
            seed=42,
            scores=scores,
            kill_criterion_passed=None,
            timestamp="2026-05-30T12:00:00Z",
        )

        write_report(report, tmp_path)

        json_path = tmp_path / "test_run.json"
        assert json_path.exists()
        parsed = json.loads(json_path.read_text())
        assert parsed["run_id"] == "test_run"


class TestMarkdownReport:
    def test_contains_kpi_table(self) -> None:
        from strikecast.eval.report import RunReport, to_markdown

        scores = _make_score_results()
        report = RunReport(
            run_id="test_run",
            data_window=("2025-12-01", "2026-05-30"),
            model_checkpoint=None,
            git_commit="abc123",
            seed=42,
            scores=scores,
            kill_criterion_passed=None,
            timestamp="2026-05-30T12:00:00Z",
        )

        md = to_markdown(report)

        assert "Brier" in md
        assert "randomwalk" in md
        assert "garch_mc" in md
        assert "abc123" in md

    def test_kill_criterion_flagged(self) -> None:
        from strikecast.eval.report import RunReport, to_markdown

        scores = _make_score_results()
        report = RunReport(
            run_id="test_run",
            data_window=("2025-12-01", "2026-05-30"),
            model_checkpoint=None,
            git_commit="abc123",
            seed=42,
            scores=scores,
            kill_criterion_passed=False,
            timestamp="2026-05-30T12:00:00Z",
        )

        md = to_markdown(report)
        assert "FAILED-KILL-CRITERION" in md

    def test_write_markdown_to_file(self, tmp_path: Path) -> None:
        from strikecast.eval.report import RunReport, write_report

        scores = _make_score_results()
        report = RunReport(
            run_id="test_run",
            data_window=("2025-12-01", "2026-05-30"),
            model_checkpoint=None,
            git_commit="abc123",
            seed=42,
            scores=scores,
            kill_criterion_passed=None,
            timestamp="2026-05-30T12:00:00Z",
        )

        write_report(report, tmp_path)

        md_path = tmp_path / "test_run.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "Brier" in content
