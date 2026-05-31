import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from strikecast.constants import WINDOW_SECONDS


def _make_lookback(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic lookback with known realized vol."""
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


class TestRandomWalkEstimator:
    def test_atm_returns_half(self) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        lookback = _make_lookback()
        current_price = float(lookback["close"].iloc[-1])
        strike = current_price

        est = RandomWalkEstimator()
        result = est.estimate(lookback, strike)

        assert abs(result.p - 0.5) < 1e-10

    def test_deep_itm_near_one(self) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        lookback = _make_lookback()
        current_price = float(lookback["close"].iloc[-1])
        strike = current_price * 0.99

        est = RandomWalkEstimator()
        result = est.estimate(lookback, strike)

        assert result.p > 0.99

    def test_deep_otm_near_zero(self) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        lookback = _make_lookback()
        current_price = float(lookback["close"].iloc[-1])
        strike = current_price * 1.01

        est = RandomWalkEstimator()
        result = est.estimate(lookback, strike)

        assert result.p < 0.01

    def test_analytic_no_samples(self) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        lookback = _make_lookback()
        est = RandomWalkEstimator()
        result = est.estimate(lookback, 35000.0)

        assert result.n_samples == 0
        assert result.p == result.p_raw
        assert result.ci_low == result.p
        assert result.ci_high == result.p

    def test_matches_scipy_cdf(self) -> None:
        from scipy.stats import norm

        from strikecast.estimators.random_walk import RandomWalkEstimator

        lookback = _make_lookback()
        closes = lookback["close"].values
        log_rets = np.diff(np.log(closes))
        sigma = float(np.std(log_rets, ddof=1))
        current_price = float(closes[-1])
        strike = current_price * 1.002

        expected = 1.0 - norm.cdf(np.log(strike / current_price) / sigma)

        est = RandomWalkEstimator()
        result = est.estimate(lookback, strike)

        assert abs(result.p - expected) < 1e-12

    def test_minimum_lookback_enforced(self) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        tiny = _make_lookback(n=2)
        est = RandomWalkEstimator(min_lookback=10)

        with pytest.raises(ValueError, match="lookback"):
            est.estimate(tiny, 35000.0)

    @given(
        strike_pct=st.floats(min_value=0.95, max_value=1.05),
    )
    @settings(max_examples=50)
    def test_probability_always_in_zero_one(self, strike_pct: float) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        lookback = _make_lookback()
        current_price = float(lookback["close"].iloc[-1])
        strike = current_price * strike_pct

        est = RandomWalkEstimator()
        result = est.estimate(lookback, strike)

        assert 0.0 <= result.p <= 1.0

    def test_monotonic_in_strike(self) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        lookback = _make_lookback()
        current_price = float(lookback["close"].iloc[-1])
        est = RandomWalkEstimator()

        strikes = [current_price * m for m in [0.995, 0.998, 1.0, 1.002, 1.005]]
        probs = [est.estimate(lookback, s).p for s in strikes]

        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1], (
                f"P should decrease as strike increases: "
                f"P({strikes[i]:.2f})={probs[i]:.6f} < P({strikes[i+1]:.2f})={probs[i+1]:.6f}"
            )
