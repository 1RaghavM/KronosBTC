"""NFR-003: Reproducibility.

Two runs with the same seed + config produce identical Brier scores
to 1e-9.
"""

import numpy as np
import pandas as pd
import pytest

from strikecast.constants import WINDOW_SECONDS


def _build_predictions_and_labels(seed: int = 42):
    """Run both estimators on synthetic data and return (predictions_df, labels_df)."""
    from strikecast.estimators.garch_mc import GarchMonteCarloEstimator
    from strikecast.estimators.random_walk import RandomWalkEstimator

    rng = np.random.RandomState(seed)
    base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
    n_lookback = 300
    n_test = 20

    log_returns = rng.normal(0, 0.001, n_lookback + n_test)
    log_prices = np.log(35000.0) + np.cumsum(log_returns)
    prices = np.exp(log_prices)

    candles = pd.DataFrame(
        {
            "window_open_ts": [base_ts + i * WINDOW_SECONDS for i in range(n_lookback + n_test)],
            "open": prices,
            "high": prices * 1.0005,
            "low": prices * 0.9995,
            "close": prices * (1 + rng.normal(0, 0.0003, n_lookback + n_test)),
            "volume": 1.0,
        }
    )

    rw = RandomWalkEstimator()
    garch = GarchMonteCarloEstimator(n_samples=500, seed=seed)

    predictions = []
    for i in range(n_lookback, n_lookback + n_test):
        lookback = candles.iloc[:i]
        target_open = float(candles.iloc[i]["open"])
        strike = target_open
        target_ts = int(candles.iloc[i]["window_open_ts"])

        for name, est in [("randomwalk", rw), ("garch_mc", garch)]:
            result = est.estimate(lookback, strike)
            predictions.append(
                {
                    "window_open_ts": target_ts,
                    "estimator": name,
                    "strike": strike,
                    "p": result.p,
                    "moneyness": 0.0,
                }
            )

    pred_df = pd.DataFrame(predictions)

    labels = pd.DataFrame(
        {
            "window_open_ts": candles.iloc[n_lookback:]["window_open_ts"].values,
            "outcome_up": candles.iloc[n_lookback:]["close"].values
            > candles.iloc[n_lookback:]["open"].values,
        }
    )

    return pred_df, labels


class TestReproducibility:
    def test_identical_brier_across_runs(self) -> None:
        from strikecast.eval.scoring import score_predictions

        pred1, labels1 = _build_predictions_and_labels(seed=42)
        pred2, labels2 = _build_predictions_and_labels(seed=42)

        scores1 = score_predictions(pred1, labels1, n_bootstrap=100, seed=42)
        scores2 = score_predictions(pred2, labels2, n_bootstrap=100, seed=42)

        for s1, s2 in zip(scores1, scores2):
            assert abs(s1.brier - s2.brier) < 1e-9, (
                f"Brier mismatch for {s1.estimator}/{s1.moneyness_bucket}: "
                f"{s1.brier} != {s2.brier}"
            )
            assert abs(s1.logloss - s2.logloss) < 1e-9, (
                f"Log loss mismatch for {s1.estimator}/{s1.moneyness_bucket}: "
                f"{s1.logloss} != {s2.logloss}"
            )

    def test_random_walk_deterministic(self) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        rng = np.random.RandomState(42)
        prices = 35000.0 + np.cumsum(rng.randn(200)) * 0.5
        lookback = pd.DataFrame({"close": prices})

        est = RandomWalkEstimator()
        r1 = est.estimate(lookback, 35000.0)
        r2 = est.estimate(lookback, 35000.0)

        assert r1.p == r2.p

    def test_garch_mc_deterministic_with_seed(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        rng = np.random.RandomState(42)
        n = 300
        log_returns = rng.normal(0, 0.001, n)
        log_prices = np.log(35000.0) + np.cumsum(log_returns)
        prices = np.exp(log_prices)
        lookback = pd.DataFrame({"close": prices})

        est1 = GarchMonteCarloEstimator(n_samples=500, seed=99)
        est2 = GarchMonteCarloEstimator(n_samples=500, seed=99)

        r1 = est1.estimate(lookback, 35000.0)
        r2 = est2.estimate(lookback, 35000.0)

        assert abs(r1.p - r2.p) < 1e-9
