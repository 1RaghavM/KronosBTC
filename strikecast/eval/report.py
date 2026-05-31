from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from strikecast.eval.scoring import ScoreResult


@dataclass
class RunReport:
    """Complete run report (FR-045)."""

    run_id: str
    data_window: tuple[str, str]
    model_checkpoint: str | None
    git_commit: str
    seed: int
    scores: list[ScoreResult]
    kill_criterion_passed: bool | None
    timestamp: str
    label_source: str = "coinbase"


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def make_run_id(git_commit: str | None = None) -> str:
    commit = git_commit or get_git_commit()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{commit}_{ts}"


def to_json(report: RunReport) -> str:
    data = {
        "run_id": report.run_id,
        "data_window": list(report.data_window),
        "model_checkpoint": report.model_checkpoint,
        "git_commit": report.git_commit,
        "seed": report.seed,
        "kill_criterion_passed": report.kill_criterion_passed,
        "timestamp": report.timestamp,
        "label_source": report.label_source,
        "scores": [asdict(s) for s in report.scores],
    }
    return json.dumps(data, indent=2)


def to_markdown(report: RunReport) -> str:
    lines: list[str] = []
    lines.append(f"# Strikecast Run Report: {report.run_id}")
    lines.append("")
    lines.append(f"- **Data window:** {report.data_window[0]} to {report.data_window[1]}")
    lines.append(f"- **Git commit:** `{report.git_commit}`")
    lines.append(f"- **Seed:** {report.seed}")
    lines.append(f"- **Timestamp:** {report.timestamp}")
    lines.append(f"- **Label source:** {report.label_source}")

    if report.label_source == "coinbase":
        lines.append(
            "  - _Outcome = Coinbase close > window-open price (model-internal). "
            "NOT the Chainlink oracle; do not read as Polymarket paper-PnL._"
        )

    if report.model_checkpoint:
        lines.append(f"- **Model checkpoint:** `{report.model_checkpoint}`")

    lines.append("")

    if report.kill_criterion_passed is False:
        lines.append("## **FAILED-KILL-CRITERION**")
        lines.append("")
        lines.append(
            "Kronos Brier skill score vs GARCH-MC <= 0 on test set. "
            "The foundation-model approach is not justified for this horizon."
        )
        lines.append("")

    if report.kill_criterion_passed is True:
        lines.append("## Kill criterion: PASSED")
        lines.append("")

    lines.append("## KPI Table")
    lines.append("")
    lines.append(
        "| Estimator | Bucket | Brier | Brier CI | Log Loss | Log Loss CI "
        "| ECE | Dir Acc | BSS | N |"
    )
    lines.append(
        "|-----------|--------|-------|----------|----------|----------"
        "---|-----|---------|-----|---|"
    )

    for s in report.scores:
        bss_str = f"{s.brier_skill_score:+.4f}" if s.brier_skill_score is not None else "n/a"
        lines.append(
            f"| {s.estimator} | {s.moneyness_bucket} "
            f"| {s.brier:.4f} | [{s.ci_brier[0]:.4f}, {s.ci_brier[1]:.4f}] "
            f"| {s.logloss:.4f} | [{s.ci_logloss[0]:.4f}, {s.ci_logloss[1]:.4f}] "
            f"| {s.ece:.4f} | {s.directional_accuracy:.4f} "
            f"| {bss_str} | {s.n_windows} |"
        )

    lines.append("")
    return "\n".join(lines)


def write_report(report: RunReport, output_dir: str | Path) -> None:
    """Write JSON and Markdown reports to the output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{report.run_id}.json"
    json_path.write_text(to_json(report))

    md_path = output_dir / f"{report.run_id}.md"
    md_path.write_text(to_markdown(report))
