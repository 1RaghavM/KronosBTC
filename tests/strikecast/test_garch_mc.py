from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from strikecast.constants import WINDOW_SECONDS


def _make_lookback(n: int = 300, seed: int = 42) -> pd.DataFrame:
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
        }
    )


class TestSimulateProbability:
    """Test the MC simulation step in isolation (no GARCH fitting)."""

    def test_atm_near_half(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        est = GarchMonteCarloEstimator(n_samples=50000, seed=42)
        result = est._simulate_probability(current_price=35000.0, sigma=0.001, strike=35000.0)

        assert abs(result.p - 0.5) < 0.02

    def test_deep_itm_high_probability(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        est = GarchMonteCarloEstimator(n_samples=10000, seed=42)
        result = est._simulate_probability(current_price=35000.0, sigma=0.001, strike=34900.0)

        assert result.p > 0.99

    def test_deep_otm_low_probability(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        est = GarchMonteCarloEstimator(n_samples=10000, seed=42)
        result = est._simulate_probability(current_price=35000.0, sigma=0.001, strike=35100.0)

        assert result.p < 0.01

    def test_bootstrap_ci_contains_point_estimate(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        est = GarchMonteCarloEstimator(n_samples=5000, seed=42, n_bootstrap=1000)
        result = est._simulate_probability(current_price=35000.0, sigma=0.001, strike=35010.0)

        assert result.ci_low <= result.p <= result.ci_high

    def test_n_samples_recorded(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        est = GarchMonteCarloEstimator(n_samples=2000, seed=42)
        result = est._simulate_probability(current_price=35000.0, sigma=0.001, strike=35000.0)

        assert result.n_samples == 2000

    def test_deterministic_with_same_seed(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        est1 = GarchMonteCarloEstimator(n_samples=5000, seed=123)
        est2 = GarchMonteCarloEstimator(n_samples=5000, seed=123)

        r1 = est1._simulate_probability(35000.0, 0.001, 35010.0)
        r2 = est2._simulate_probability(35000.0, 0.001, 35010.0)

        assert r1.p == r2.p
        assert r1.ci_low == r2.ci_low
        assert r1.ci_high == r2.ci_high


class TestGarchFit:
    def test_estimate_runs_on_real_data(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        lookback = _make_lookback(n=300)
        current_price = float(lookback["close"].iloc[-1])

        est = GarchMonteCarloEstimator(n_samples=1000, seed=42)
        result = est.estimate(lookback, strike=current_price)

        assert 0.0 <= result.p <= 1.0
        assert result.n_samples == 1000

    def test_minimum_lookback_enforced(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        tiny = _make_lookback(n=20)
        est = GarchMonteCarloEstimator(n_samples=1000, seed=42, min_lookback=50)

        with pytest.raises(ValueError, match="lookback"):
            est.estimate(tiny, 35000.0)

    def test_garch_failure_falls_back_to_realized_vol(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        lookback = _make_lookback(n=300)
        est = GarchMonteCarloEstimator(n_samples=1000, seed=42)

        with patch("strikecast.estimators.garch_mc.arch_model") as mock_am:
            mock_am.side_effect = Exception("convergence failed")
            result = est.estimate(lookback, strike=35000.0)

        assert 0.0 <= result.p <= 1.0

    @given(
        strike_pct=st.floats(min_value=0.95, max_value=1.05),
    )
    @settings(max_examples=20)
    def test_probability_always_in_zero_one(self, strike_pct: float) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        lookback = _make_lookback(n=300)
        current_price = float(lookback["close"].iloc[-1])

        est = GarchMonteCarloEstimator(n_samples=500, seed=42)
        result = est.estimate(lookback, strike=current_price * strike_pct)

        assert 0.0 <= result.p <= 1.0
