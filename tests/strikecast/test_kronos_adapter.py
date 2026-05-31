"""Tests for the Kronos path-sampler adapter (NFR-009 fork isolation).

The adapter is exercised with a stub predictor so no model weights are loaded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strikecast.constants import WINDOW_SECONDS


def _lookback(n: int = 64) -> pd.DataFrame:
    base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
    rng = np.random.RandomState(0)
    prices = 35000.0 + rng.randn(n).cumsum()
    return pd.DataFrame(
        {
            "window_open_ts": [base_ts + i * WINDOW_SECONDS for i in range(n)],
            "open": prices,
            "high": prices + 5,
            "low": prices - 5,
            "close": prices,
            "volume": 1.0,
            "amount": 0.0,
        }
    )


class _StubPredictor:
    """Records predict_batch calls and returns deterministic close paths."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._counter = 0.0

    def predict_batch(
        self,
        df_list,
        x_timestamp_list,
        y_timestamp_list,
        pred_len,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    ):
        self.calls.append(
            {
                "batch": len(df_list),
                "pred_len": pred_len,
                "T": T,
                "top_p": top_p,
                "sample_count": sample_count,
                "verbose": verbose,
            }
        )
        out = []
        for _ in df_list:
            self._counter += 1.0
            y = y_timestamp_list[0]
            out.append(
                pd.DataFrame(
                    {
                        "open": [35000.0] * len(y),
                        "high": [35000.0] * len(y),
                        "low": [35000.0] * len(y),
                        "close": [35000.0 + self._counter] * len(y),
                        "volume": [1.0] * len(y),
                        "amount": [0.0] * len(y),
                    },
                    index=y,
                )
            )
        return out


class TestKronosPathSampler:
    def test_returns_one_close_per_sample(self) -> None:
        from strikecast.estimators.kronos_adapter import KronosPathSampler

        predictor = _StubPredictor()
        sampler = KronosPathSampler(predictor, max_batch=256)

        lb = _lookback()
        x_ts = pd.to_datetime(lb["window_open_ts"], unit="s", utc=True)
        y_ts = pd.to_datetime(
            [int(lb["window_open_ts"].iloc[-1]) + WINDOW_SECONDS], unit="s", utc=True
        )

        closes = sampler.sample_closes(lb, x_ts, y_ts, sample_count=10, temperature=1.0, top_p=0.9)

        assert isinstance(closes, np.ndarray)
        assert closes.shape == (10,)
        assert len(np.unique(closes)) == 10  # each draw independent

    def test_uses_sample_count_one_per_path(self) -> None:
        from strikecast.estimators.kronos_adapter import KronosPathSampler

        predictor = _StubPredictor()
        sampler = KronosPathSampler(predictor, max_batch=4)

        lb = _lookback()
        x_ts = pd.to_datetime(lb["window_open_ts"], unit="s", utc=True)
        y_ts = pd.to_datetime(
            [int(lb["window_open_ts"].iloc[-1]) + WINDOW_SECONDS], unit="s", utc=True
        )

        sampler.sample_closes(lb, x_ts, y_ts, sample_count=10, temperature=1.3, top_p=0.8)

        assert all(c["sample_count"] == 1 for c in predictor.calls)
        assert all(c["verbose"] is False for c in predictor.calls)
        assert all(c["T"] == 1.3 and c["top_p"] == 0.8 for c in predictor.calls)
        assert sum(c["batch"] for c in predictor.calls) == 10  # chunked but total == sample_count
        assert max(c["batch"] for c in predictor.calls) <= 4

    def test_integrates_with_estimator(self) -> None:
        from strikecast.estimators.kronos_adapter import KronosPathSampler
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        predictor = _StubPredictor()
        sampler = KronosPathSampler(predictor)
        est = KronosBinaryEstimator(sampler, sample_count=20, min_lookback=10)

        result = est.estimate(_lookback(), strike=35000.0)
        assert 0.0 <= result.p <= 1.0
        assert result.n_samples == 20
