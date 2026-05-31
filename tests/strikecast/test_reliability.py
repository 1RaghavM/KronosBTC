"""Tests for the CORP/PAV reliability-diagram generator (FR-022).

The generator writes a PNG comparing uncalibrated vs calibrated reliability and
reports ECE for both. The reported ECE values must match
``scoring.expected_calibration_error`` exactly (it is the single source of
truth for ECE).
"""

from __future__ import annotations

import numpy as np


def _data(n: int = 6000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    true_p = rng.uniform(0.0, 1.0, n)
    outcomes = (rng.uniform(0.0, 1.0, n) < true_p).astype(float)
    p_raw = true_p**2
    return p_raw, outcomes


class TestReliabilityDiagram:
    def test_png_written(self, tmp_path) -> None:  # noqa: ANN001
        from strikecast.calibration.calibrator import fit_calibrator
        from strikecast.calibration.reliability import make_reliability_diagram

        p_raw, outcomes = _data()
        cal = fit_calibrator("isotonic", p_raw, outcomes)
        p_cal = np.asarray(cal.apply(p_raw))

        out = tmp_path / "reliability.png"
        result = make_reliability_diagram(p_raw, p_cal, outcomes, out)

        assert out.exists()
        assert out.stat().st_size > 0
        assert result.png_path == out

    def test_ece_matches_scoring(self, tmp_path) -> None:  # noqa: ANN001
        from strikecast.calibration.calibrator import fit_calibrator
        from strikecast.calibration.reliability import make_reliability_diagram
        from strikecast.eval.scoring import expected_calibration_error

        p_raw, outcomes = _data(seed=5)
        cal = fit_calibrator("isotonic", p_raw, outcomes)
        p_cal = np.asarray(cal.apply(p_raw))

        out = tmp_path / "rel.png"
        result = make_reliability_diagram(p_raw, p_cal, outcomes, out)

        assert result.ece_raw == expected_calibration_error(p_raw, outcomes)
        assert result.ece_cal == expected_calibration_error(p_cal, outcomes)

    def test_calibrated_ece_not_worse(self, tmp_path) -> None:  # noqa: ANN001
        from strikecast.calibration.calibrator import fit_calibrator
        from strikecast.calibration.reliability import make_reliability_diagram

        p_raw, outcomes = _data(seed=7)
        half = len(p_raw) // 2
        cal = fit_calibrator("isotonic", p_raw[:half], outcomes[:half])
        p_cal = np.asarray(cal.apply(p_raw))

        result = make_reliability_diagram(
            p_raw, p_cal, outcomes, tmp_path / "r.png"
        )
        assert result.ece_cal <= result.ece_raw

    def test_respects_bin_count(self, tmp_path) -> None:  # noqa: ANN001
        from strikecast.calibration.reliability import make_reliability_diagram
        from strikecast.eval.scoring import expected_calibration_error

        p_raw, outcomes = _data(seed=9)
        result = make_reliability_diagram(
            p_raw, p_raw, outcomes, tmp_path / "r.png", n_bins=10
        )
        assert result.ece_raw == expected_calibration_error(p_raw, outcomes, n_bins=10)
