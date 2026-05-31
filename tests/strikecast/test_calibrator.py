"""Unit + property tests for the post-hoc calibration layer (FR-020..023).

The calibrator is fit on a held-out validation split and applied to test
probabilities. Tests verify monotonicity (isotonic invariant), the Platt
fallback, fit-on-val / apply-on-test separation, the save/load round-trip
(FR-023), that calibration reduces ECE on a miscalibrated fixture, and the
hypothesis property that calibrated ``p`` stays in [0, 1] and monotone in raw p.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def _miscalibrated_data(
    n: int = 6000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic, systematically over-confident raw probabilities.

    ``true_p`` drives the Bernoulli outcomes, while the raw probability is a
    monotone distortion ``true_p ** 2`` (pushed toward 0). A good calibrator
    fit on (p_raw, outcomes) recovers ``true_p`` and lowers ECE.
    """
    rng = np.random.RandomState(seed)
    true_p = rng.uniform(0.0, 1.0, n)
    outcomes = (rng.uniform(0.0, 1.0, n) < true_p).astype(float)
    p_raw = true_p**2
    return p_raw, outcomes


class TestIsotonicCalibrator:
    def test_apply_is_monotonic_non_decreasing(self) -> None:
        from strikecast.calibration.calibrator import IsotonicCalibrator

        p_raw, outcomes = _miscalibrated_data()
        cal = IsotonicCalibrator().fit(p_raw, outcomes)

        grid = np.linspace(0.0, 1.0, 200)
        mapped = np.asarray(cal.apply(grid))
        assert np.all(np.diff(mapped) >= -1e-9)

    def test_apply_clamped_to_unit_interval(self) -> None:
        from strikecast.calibration.calibrator import IsotonicCalibrator

        p_raw, outcomes = _miscalibrated_data()
        cal = IsotonicCalibrator().fit(p_raw, outcomes)

        mapped = np.asarray(cal.apply(np.array([-0.5, 0.0, 0.5, 1.0, 1.5])))
        assert np.all(mapped >= 0.0)
        assert np.all(mapped <= 1.0)

    def test_scalar_input_returns_float(self) -> None:
        from strikecast.calibration.calibrator import IsotonicCalibrator

        p_raw, outcomes = _miscalibrated_data()
        cal = IsotonicCalibrator().fit(p_raw, outcomes)

        out = cal.apply(0.5)
        assert isinstance(out, float)
        assert 0.0 <= out <= 1.0

    def test_satisfies_calibrator_protocol(self) -> None:
        from strikecast.calibration.calibrator import IsotonicCalibrator
        from strikecast.estimators.base import Calibrator

        p_raw, outcomes = _miscalibrated_data()
        cal = IsotonicCalibrator().fit(p_raw, outcomes)
        assert isinstance(cal, Calibrator)


class TestPlattCalibrator:
    def test_apply_monotonic_and_bounded(self) -> None:
        from strikecast.calibration.calibrator import PlattCalibrator

        p_raw, outcomes = _miscalibrated_data()
        cal = PlattCalibrator().fit(p_raw, outcomes)

        grid = np.linspace(0.0, 1.0, 200)
        mapped = np.asarray(cal.apply(grid))
        assert np.all(np.diff(mapped) >= -1e-9)
        assert np.all((mapped >= 0.0) & (mapped <= 1.0))

    def test_satisfies_calibrator_protocol(self) -> None:
        from strikecast.calibration.calibrator import PlattCalibrator
        from strikecast.estimators.base import Calibrator

        p_raw, outcomes = _miscalibrated_data()
        cal = PlattCalibrator().fit(p_raw, outcomes)
        assert isinstance(cal, Calibrator)


class TestFitCalibratorFactory:
    def test_isotonic_method_returns_isotonic(self) -> None:
        from strikecast.calibration.calibrator import (
            IsotonicCalibrator,
            fit_calibrator,
        )

        p_raw, outcomes = _miscalibrated_data()
        cal = fit_calibrator("isotonic", p_raw, outcomes)
        assert isinstance(cal, IsotonicCalibrator)

    def test_platt_method_returns_platt(self) -> None:
        from strikecast.calibration.calibrator import (
            PlattCalibrator,
            fit_calibrator,
        )

        p_raw, outcomes = _miscalibrated_data()
        cal = fit_calibrator("platt", p_raw, outcomes)
        assert isinstance(cal, PlattCalibrator)

    def test_unknown_method_raises(self) -> None:
        from strikecast.calibration.calibrator import fit_calibrator

        p_raw, outcomes = _miscalibrated_data()
        with pytest.raises(ValueError, match="method"):
            fit_calibrator("bogus", p_raw, outcomes)  # type: ignore[arg-type]


class TestFitOnValApplyOnTest:
    def test_calibration_reduces_ece_isotonic(self) -> None:
        from strikecast.calibration.calibrator import fit_calibrator
        from strikecast.eval.scoring import expected_calibration_error

        p_raw, outcomes = _miscalibrated_data(n=8000, seed=1)
        half = len(p_raw) // 2
        val_p, val_y = p_raw[:half], outcomes[:half]
        test_p, test_y = p_raw[half:], outcomes[half:]

        cal = fit_calibrator("isotonic", val_p, val_y)
        test_p_cal = np.asarray(cal.apply(test_p))

        ece_raw = expected_calibration_error(test_p, test_y)
        ece_cal = expected_calibration_error(test_p_cal, test_y)

        assert ece_cal <= ece_raw
        assert ece_cal < 0.05

    def test_calibration_reduces_ece_platt(self) -> None:
        from strikecast.calibration.calibrator import fit_calibrator
        from strikecast.eval.scoring import expected_calibration_error

        p_raw, outcomes = _miscalibrated_data(n=8000, seed=2)
        half = len(p_raw) // 2
        cal = fit_calibrator("platt", p_raw[:half], outcomes[:half])
        test_p, test_y = p_raw[half:], outcomes[half:]
        test_p_cal = np.asarray(cal.apply(test_p))

        ece_raw = expected_calibration_error(test_p, test_y)
        ece_cal = expected_calibration_error(test_p_cal, test_y)
        assert ece_cal <= ece_raw


class TestSaveLoadRoundTrip:
    def test_isotonic_round_trip(self, tmp_path) -> None:  # noqa: ANN001
        from strikecast.calibration.calibrator import (
            IsotonicCalibrator,
            load_calibrator,
        )

        p_raw, outcomes = _miscalibrated_data()
        cal = IsotonicCalibrator(
            model_checkpoint="NeoQuasar/Kronos-small",
            data_window=("2025-12-01", "2026-05-30"),
        ).fit(p_raw, outcomes)

        path = tmp_path / "cal.pkl"
        cal.save(path)
        loaded = load_calibrator(path)

        grid = np.linspace(0.0, 1.0, 50)
        np.testing.assert_allclose(
            np.asarray(cal.apply(grid)), np.asarray(loaded.apply(grid))
        )
        assert loaded.model_checkpoint == "NeoQuasar/Kronos-small"
        assert loaded.data_window == ("2025-12-01", "2026-05-30")

    def test_platt_round_trip(self, tmp_path) -> None:  # noqa: ANN001
        from strikecast.calibration.calibrator import (
            PlattCalibrator,
            load_calibrator,
        )

        p_raw, outcomes = _miscalibrated_data()
        cal = PlattCalibrator().fit(p_raw, outcomes)
        path = tmp_path / "platt.pkl"
        cal.save(path)
        loaded = load_calibrator(path)

        grid = np.linspace(0.0, 1.0, 50)
        np.testing.assert_allclose(
            np.asarray(cal.apply(grid)), np.asarray(loaded.apply(grid))
        )

    def test_apply_before_fit_raises(self) -> None:
        from strikecast.calibration.calibrator import IsotonicCalibrator

        with pytest.raises(RuntimeError, match="fit"):
            IsotonicCalibrator().apply(0.5)


class TestCalibratorProperties:
    @given(raw=st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=50, deadline=None)
    def test_calibrated_p_in_unit_interval(self, raw: float) -> None:
        from strikecast.calibration.calibrator import fit_calibrator

        p_raw, outcomes = _miscalibrated_data(seed=3)
        cal = fit_calibrator("isotonic", p_raw, outcomes)
        out = float(np.asarray(cal.apply(raw)).reshape(-1)[0])
        assert 0.0 <= out <= 1.0

    @given(
        a=st.floats(min_value=0.0, max_value=1.0),
        b=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_monotone_in_raw_p(self, a: float, b: float) -> None:
        from strikecast.calibration.calibrator import fit_calibrator

        p_raw, outcomes = _miscalibrated_data(seed=4)
        cal = fit_calibrator("isotonic", p_raw, outcomes)
        lo, hi = (a, b) if a <= b else (b, a)
        out_lo = float(np.asarray(cal.apply(lo)).reshape(-1)[0])
        out_hi = float(np.asarray(cal.apply(hi)).reshape(-1)[0])
        assert out_lo <= out_hi + 1e-9
