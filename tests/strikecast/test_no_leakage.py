"""NFR-002: No look-ahead leakage.

Permuting bars strictly after a window must leave that window's
probability bit-identical. A failure here invalidates every backtest
number — treat as a release blocker, not a flaky test.
"""

import numpy as np
import pandas as pd
import pytest

from strikecast.constants import WINDOW_SECONDS


def _make_series(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
    log_returns = rng.normal(0, 0.001, n)
    log_prices = np.log(35000.0) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
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


class TestNoLeakage:
    def test_random_walk_no_leakage(self) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        series = _make_series()
        split_idx = 250
        lookback = series.iloc[:split_idx].copy()
        strike = float(lookback["close"].iloc[-1])

        est = RandomWalkEstimator()
        p_original = est.estimate(lookback, strike)

        shuffled = series.copy()
        future = shuffled.iloc[split_idx:].copy()
        future_shuffled = future.sample(frac=1, random_state=99).reset_index(drop=True)
        future_shuffled["window_open_ts"] = future["window_open_ts"].values
        shuffled.iloc[split_idx:] = future_shuffled.values

        lookback_after = shuffled.iloc[:split_idx].copy()
        p_shuffled = est.estimate(lookback_after, strike)

        assert p_original.p == p_shuffled.p, (
            f"Random walk probability changed after shuffling future data: "
            f"{p_original.p} != {p_shuffled.p}"
        )

    def test_garch_mc_no_leakage(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        series = _make_series()
        split_idx = 250
        lookback = series.iloc[:split_idx].copy()
        strike = float(lookback["close"].iloc[-1])

        est = GarchMonteCarloEstimator(n_samples=1000, seed=42)
        p_original = est.estimate(lookback, strike)

        shuffled = series.copy()
        future = shuffled.iloc[split_idx:].copy()
        future_shuffled = future.sample(frac=1, random_state=99).reset_index(drop=True)
        future_shuffled["window_open_ts"] = future["window_open_ts"].values
        shuffled.iloc[split_idx:] = future_shuffled.values

        lookback_after = shuffled.iloc[:split_idx].copy()
        p_shuffled = est.estimate(lookback_after, strike)

        assert p_original.p == p_shuffled.p, (
            f"GARCH-MC probability changed after shuffling future data: "
            f"{p_original.p} != {p_shuffled.p}"
        )
