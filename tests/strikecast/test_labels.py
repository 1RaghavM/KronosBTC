"""Tests for Coinbase-close resolution labels (FR-042, model-internal eval)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strikecast.constants import LABEL_COLUMNS, WINDOW_SECONDS


def _candles(n: int = 30, seed: int = 1) -> pd.DataFrame:
    base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
    rng = np.random.RandomState(seed)
    opens = 35000.0 + rng.randn(n).cumsum()
    closes = opens * (1 + rng.normal(0, 0.001, n))
    return pd.DataFrame(
        {
            "symbol": "BTC/USD",
            "granularity": WINDOW_SECONDS,
            "window_open_ts": [base_ts + i * WINDOW_SECONDS for i in range(n)],
            "open": opens,
            "high": np.maximum(opens, closes) + 1,
            "low": np.minimum(opens, closes) - 1,
            "close": closes,
            "volume": 1.0,
            "amount": 0.0,
            "source": "coinbase",
        }
    )


class TestBuildCoinbaseLabels:
    def test_has_label_schema(self) -> None:
        from strikecast.data.labels import build_coinbase_labels

        labels = build_coinbase_labels(_candles())
        assert set(LABEL_COLUMNS).issubset(labels.columns)

    def test_outcome_is_close_gt_open(self) -> None:
        from strikecast.data.labels import build_coinbase_labels

        candles = _candles()
        labels = build_coinbase_labels(candles)
        expected = (candles["close"].to_numpy() > candles["open"].to_numpy())
        np.testing.assert_array_equal(labels["outcome_up"].to_numpy(), expected)

    def test_coinbase_close_recorded(self) -> None:
        from strikecast.data.labels import build_coinbase_labels

        candles = _candles()
        labels = build_coinbase_labels(candles)
        np.testing.assert_allclose(
            labels["coinbase_close"].to_numpy(), candles["close"].to_numpy()
        )

    def test_grid_aligned_and_writable(self, tmp_path) -> None:
        from strikecast.data.labels import build_coinbase_labels
        from strikecast.data.store import DataStore

        for sub in ["candles", "pm_markets", "resolution_labels", "reports"]:
            (tmp_path / sub).mkdir()
        labels = build_coinbase_labels(_candles())
        assert (labels["window_open_ts"] % WINDOW_SECONDS == 0).all()
        DataStore(tmp_path).append_labels(labels)  # must not raise

    def test_empty_candles_returns_empty(self) -> None:
        from strikecast.data.labels import build_coinbase_labels

        labels = build_coinbase_labels(pd.DataFrame(columns=["window_open_ts", "open", "close"]))
        assert labels.empty
        assert set(LABEL_COLUMNS).issubset(labels.columns)
