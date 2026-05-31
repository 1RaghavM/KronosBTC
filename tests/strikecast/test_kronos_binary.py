"""Unit + property tests for the Kronos zero-shot binary estimator (FR-010..015).

These tests inject a deterministic fake ``PathSampler`` so the estimator's
Monte Carlo aggregation, bootstrap CI, calibration hook, leakage safety, and
context clamping can be verified without loading the heavy Kronos model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from strikecast.constants import WINDOW_SECONDS


def _make_lookback(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    log_returns = rng.normal(0, 0.001, n)
    log_prices = np.log(35000.0) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
    base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
    return pd.DataFrame(
        {
            "window_open_ts": [base_ts + i * WINDOW_SECONDS for i in range(n)],
            "open": prices,
            "high": prices * 1.0005,
            "low": prices * 0.9995,
            "close": prices,
            "volume": 1.0,
            "amount": 0.0,
        }
    )


class _GaussianFakeSampler:
    """Deterministic sampler: lognormal closes centered on the last close.

    Depends ONLY on the lookback's last close + a fixed seed, so it is
    leakage-safe by construction and reproducible across calls.
    """

    def __init__(self, sigma: float = 0.002, seed: int = 7) -> None:
        self._sigma = sigma
        self._seed = seed

    def sample_closes(
        self,
        lookback_df: pd.DataFrame,
        x_timestamp: pd.DatetimeIndex,
        y_timestamp: pd.DatetimeIndex,
        sample_count: int,
        temperature: float,
        top_p: float,
    ) -> np.ndarray:
        current_price = float(lookback_df["close"].values[-1])
        rng = np.random.RandomState(self._seed)
        log_rets = rng.normal(0.0, self._sigma, sample_count)
        return current_price * np.exp(log_rets)


class _ConstantSampler:
    """Returns all closes at a fixed value (for deterministic edge cases)."""

    def __init__(self, value: float) -> None:
        self._value = value

    def sample_closes(self, lookback_df, x_timestamp, y_timestamp, sample_count, temperature, top_p):  # noqa: ANN001
        return np.full(sample_count, self._value, dtype=float)


class _DoublingCalibrator:
    """Monotonic toy calibrator: p -> clip(2*p, 0, 1)."""

    def apply(self, p_raw):  # noqa: ANN001
        return np.clip(np.asarray(p_raw, dtype=float) * 2.0, 0.0, 1.0)


class TestKronosBinaryEstimator:
    def test_atm_near_half(self) -> None:
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        lookback = _make_lookback()
        strike = float(lookback["close"].iloc[-1])

        est = KronosBinaryEstimator(_GaussianFakeSampler(), sample_count=20000, seed=42)
        result = est.estimate(lookback, strike)

        assert abs(result.p - 0.5) < 0.02

    def test_deep_itm_high_probability(self) -> None:
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        lookback = _make_lookback()
        current = float(lookback["close"].iloc[-1])

        est = KronosBinaryEstimator(_GaussianFakeSampler(sigma=0.001), sample_count=10000, seed=42)
        result = est.estimate(lookback, current * 0.99)

        assert result.p > 0.99

    def test_deep_otm_low_probability(self) -> None:
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        lookback = _make_lookback()
        current = float(lookback["close"].iloc[-1])

        est = KronosBinaryEstimator(_GaussianFakeSampler(sigma=0.001), sample_count=10000, seed=42)
        result = est.estimate(lookback, current * 1.01)

        assert result.p < 0.01

    def test_n_samples_recorded(self) -> None:
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        lookback = _make_lookback()
        est = KronosBinaryEstimator(_GaussianFakeSampler(), sample_count=3000, seed=42)
        result = est.estimate(lookback, 35000.0)

        assert result.n_samples == 3000

    def test_bootstrap_ci_contains_point_estimate(self) -> None:
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        lookback = _make_lookback()
        current = float(lookback["close"].iloc[-1])

        est = KronosBinaryEstimator(
            _GaussianFakeSampler(), sample_count=5000, seed=42, n_bootstrap=1000
        )
        result = est.estimate(lookback, current * 1.0003)

        assert result.ci_low <= result.p <= result.ci_high

    def test_no_calibrator_means_p_equals_p_raw(self) -> None:
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        lookback = _make_lookback()
        est = KronosBinaryEstimator(_GaussianFakeSampler(), sample_count=2000, seed=42)
        result = est.estimate(lookback, 35010.0)

        assert result.p == result.p_raw

    def test_calibrator_is_applied(self) -> None:
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        lookback = _make_lookback()
        strike = float(lookback["close"].iloc[-1])

        raw_est = KronosBinaryEstimator(_GaussianFakeSampler(), sample_count=20000, seed=42)
        raw = raw_est.estimate(lookback, strike)

        cal_est = KronosBinaryEstimator(
            _GaussianFakeSampler(),
            calibrator=_DoublingCalibrator(),
            sample_count=20000,
            seed=42,
        )
        cal = cal_est.estimate(lookback, strike)

        assert cal.p_raw == raw.p_raw
        assert abs(cal.p - min(1.0, 2.0 * raw.p_raw)) < 1e-9
        assert cal.p != cal.p_raw

    def test_deterministic_with_same_seed(self) -> None:
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        lookback = _make_lookback()
        est1 = KronosBinaryEstimator(_GaussianFakeSampler(seed=11), sample_count=5000, seed=99)
        est2 = KronosBinaryEstimator(_GaussianFakeSampler(seed=11), sample_count=5000, seed=99)

        r1 = est1.estimate(lookback, 35010.0)
        r2 = est2.estimate(lookback, 35010.0)

        assert r1.p == r2.p
        assert r1.ci_low == r2.ci_low
        assert r1.ci_high == r2.ci_high

    def test_minimum_lookback_enforced(self) -> None:
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        tiny = _make_lookback(n=10)
        est = KronosBinaryEstimator(_GaussianFakeSampler(), sample_count=1000, min_lookback=50)

        with pytest.raises(ValueError, match="lookback"):
            est.estimate(tiny, 35000.0)

    def test_context_clamped_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        class _RecordingSampler:
            def __init__(self) -> None:
                self.seen_rows = -1

            def sample_closes(self, lookback_df, x_timestamp, y_timestamp, sample_count, temperature, top_p):  # noqa: ANN001
                self.seen_rows = len(lookback_df)
                return np.full(sample_count, float(lookback_df["close"].values[-1]))

        sampler = _RecordingSampler()
        lookback = _make_lookback(n=900)
        est = KronosBinaryEstimator(sampler, sample_count=500, max_context=512)

        import logging

        with caplog.at_level(logging.WARNING):
            est.estimate(lookback, 35000.0)

        assert sampler.seen_rows == 512
        assert any("context" in rec.message.lower() for rec in caplog.records)

    def test_target_timestamp_is_next_grid_window(self) -> None:
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        captured: dict[str, pd.DatetimeIndex] = {}

        class _CapturingSampler:
            def sample_closes(self, lookback_df, x_timestamp, y_timestamp, sample_count, temperature, top_p):  # noqa: ANN001
                captured["x"] = x_timestamp
                captured["y"] = y_timestamp
                return np.full(sample_count, 35000.0)

        lookback = _make_lookback(n=100)
        est = KronosBinaryEstimator(_CapturingSampler(), sample_count=100)
        est.estimate(lookback, 35000.0)

        last_ts = int(lookback["window_open_ts"].iloc[-1])
        expected = pd.to_datetime([last_ts + WINDOW_SECONDS], unit="s", utc=True)
        assert list(captured["y"]) == list(expected)
        assert len(captured["x"]) == 100
        # Kronos' calc_time_stamps requires the ``.dt`` accessor -> must be Series,
        # not a DatetimeIndex (regression guard).
        assert isinstance(captured["x"], pd.Series)
        assert isinstance(captured["y"], pd.Series)
        assert hasattr(captured["x"], "dt") and hasattr(captured["y"], "dt")

    @given(strike_pct=st.floats(min_value=0.95, max_value=1.05))
    @settings(max_examples=50, deadline=None)
    def test_probability_always_in_zero_one(self, strike_pct: float) -> None:
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        lookback = _make_lookback()
        current = float(lookback["close"].iloc[-1])
        est = KronosBinaryEstimator(_GaussianFakeSampler(), sample_count=500, seed=42)
        result = est.estimate(lookback, current * strike_pct)

        assert 0.0 <= result.p <= 1.0
        assert 0.0 <= result.ci_low <= result.ci_high <= 1.0

    def test_monotonic_in_strike(self) -> None:
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        lookback = _make_lookback()
        current = float(lookback["close"].iloc[-1])
        est = KronosBinaryEstimator(_GaussianFakeSampler(), sample_count=20000, seed=42)

        strikes = [current * m for m in [0.995, 0.998, 1.0, 1.002, 1.005]]
        probs = [est.estimate(lookback, s).p for s in strikes]

        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1]

    def test_satisfies_estimator_protocol(self) -> None:
        from strikecast.estimators.base import Estimator
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        est = KronosBinaryEstimator(_GaussianFakeSampler())
        assert isinstance(est, Estimator)


class TestKronosNoLeakage:
    """NFR-002: shuffling future bars must not change the past probability."""

    def test_no_leakage(self) -> None:
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        series = _make_lookback(n=500)
        split_idx = 250
        lookback = series.iloc[:split_idx].copy()
        strike = float(lookback["close"].iloc[-1])

        est = KronosBinaryEstimator(_GaussianFakeSampler(), sample_count=4000, seed=42)
        p_original = est.estimate(lookback, strike)

        shuffled = series.copy()
        future = shuffled.iloc[split_idx:].copy()
        future_shuffled = future.sample(frac=1, random_state=99).reset_index(drop=True)
        future_shuffled["window_open_ts"] = future["window_open_ts"].values
        shuffled.iloc[split_idx:] = future_shuffled.values

        lookback_after = shuffled.iloc[:split_idx].copy()
        p_shuffled = est.estimate(lookback_after, strike)

        assert p_original.p == p_shuffled.p
