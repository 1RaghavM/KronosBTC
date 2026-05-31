"""CORP/PAV reliability diagrams and ECE figures (FR-022).

A reliability diagram plots predicted probability against observed outcome
frequency. Classical equal-width binning is unstable; the CORP approach
(Consistent, Optimally binned, Reproducible, PAV-based) fits an isotonic
regression of outcomes on predictions and plots the resulting consistent
recalibration curve.

This module renders both the *uncalibrated* and *calibrated* reliability curves
side by side so the effect of calibration is visually auditable, and reports
the Expected Calibration Error for each. ECE is computed by
:func:`strikecast.eval.scoring.expected_calibration_error`, the single source
of truth, so the figure and the KPI table never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never to a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from strikecast.constants import DEFAULT_ECE_BINS  # noqa: E402
from strikecast.eval.scoring import expected_calibration_error  # noqa: E402


@dataclass(frozen=True)
class ReliabilityResult:
    """Output of a reliability-diagram run.

    Attributes:
        ece_raw: ECE of the uncalibrated probabilities (probability units).
        ece_cal: ECE of the calibrated probabilities (probability units).
        png_path: Path to the written PNG figure.
        n_windows: Number of (probability, outcome) pairs plotted.
    """

    ece_raw: float
    ece_cal: float
    png_path: Path
    n_windows: int


def _corp_curve(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the CORP (PAV isotonic) recalibration curve for predictions ``p``.

    Args:
        p: Predicted probabilities in [0, 1].
        y: Binary outcomes (0.0/1.0).

    Returns:
        ``(x_sorted, y_recalibrated)`` where ``x_sorted`` is the unique sorted
        predictions and ``y_recalibrated`` the PAV-fitted observed frequency.
    """
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(p, y)
    order = np.argsort(p)
    x_sorted = p[order]
    y_recal = np.clip(iso.predict(x_sorted), 0.0, 1.0)
    return x_sorted, y_recal


def make_reliability_diagram(
    p_raw: np.ndarray,
    p_cal: np.ndarray,
    outcomes: np.ndarray,
    output_path: str | Path,
    n_bins: int = DEFAULT_ECE_BINS,
) -> ReliabilityResult:
    """Render a CORP reliability diagram for raw vs calibrated probabilities.

    Args:
        p_raw: Uncalibrated probabilities in [0, 1] (1-D).
        p_cal: Calibrated probabilities in [0, 1] (same windows, 1-D).
        outcomes: Binary outcomes (0.0/1.0) for the same windows.
        output_path: Destination PNG path. Parent directories are created.
        n_bins: Number of bins for the reported ECE (default
            :data:`~strikecast.constants.DEFAULT_ECE_BINS`).

    Returns:
        A :class:`ReliabilityResult` with the raw and calibrated ECE and the
        PNG path. ECE values match
        :func:`strikecast.eval.scoring.expected_calibration_error`.
    """
    p_raw = np.asarray(p_raw, dtype=float).reshape(-1)
    p_cal = np.asarray(p_cal, dtype=float).reshape(-1)
    y = np.asarray(outcomes, dtype=float).reshape(-1)

    ece_raw = expected_calibration_error(p_raw, y, n_bins=n_bins)
    ece_cal = expected_calibration_error(p_cal, y, n_bins=n_bins)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.0, label="perfect calibration")

    raw_x, raw_y = _corp_curve(p_raw, y)
    cal_x, cal_y = _corp_curve(p_cal, y)
    ax.plot(raw_x, raw_y, color="tab:red", linewidth=2.0, label=f"raw (ECE={ece_raw:.4f})")
    ax.plot(cal_x, cal_y, color="tab:blue", linewidth=2.0, label=f"calibrated (ECE={ece_cal:.4f})")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted probability P(close > strike)")
    ax.set_ylabel("Observed outcome frequency (PAV / CORP)")
    ax.set_title("CORP reliability diagram: raw vs calibrated")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    return ReliabilityResult(
        ece_raw=ece_raw,
        ece_cal=ece_cal,
        png_path=output_path,
        n_windows=len(y),
    )
