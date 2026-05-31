from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from strikecast.constants import WINDOW_SECONDS


def _make_ohlcv_response(base_ts: int, n: int = 5) -> list[list]:
    """Create a fake ccxt ohlcv response (list of [ts_ms, o, h, l, c, v])."""
    return [
        [
            (base_ts + i * WINDOW_SECONDS) * 1000,
            35000.0 + i,
            35010.0 + i,
            34990.0 + i,
            35005.0 + i,
            1.5 + i * 0.1,
        ]
        for i in range(n)
    ]


class TestCoinbaseSource:
    def test_fetch_returns_correct_schema(self) -> None:
        from strikecast.data.candle_source import CoinbaseSource

        base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = _make_ohlcv_response(base_ts, 5)

        source = CoinbaseSource(exchange=mock_exchange)
        df = source.fetch("BTC/USD", WINDOW_SECONDS, base_ts, base_ts + 5 * WINDOW_SECONDS)

        assert len(df) == 5
        assert list(df.columns) == [
            "symbol", "granularity", "window_open_ts",
            "open", "high", "low", "close", "volume", "amount", "source",
        ]
        assert (df["symbol"] == "BTC/USD").all()
        assert (df["source"] == "coinbase").all()
        assert (df["window_open_ts"] % WINDOW_SECONDS == 0).all()

    def test_fetch_filters_to_requested_range(self) -> None:
        from strikecast.data.candle_source import CoinbaseSource

        base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = _make_ohlcv_response(base_ts, 10)

        source = CoinbaseSource(exchange=mock_exchange)
        df = source.fetch(
            "BTC/USD", WINDOW_SECONDS, base_ts, base_ts + 5 * WINDOW_SECONDS
        )

        assert len(df) == 5
        assert df["window_open_ts"].max() < base_ts + 5 * WINDOW_SECONDS

    def test_fetch_empty_response(self) -> None:
        from strikecast.data.candle_source import CoinbaseSource

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = []

        source = CoinbaseSource(exchange=mock_exchange)
        df = source.fetch("BTC/USD", WINDOW_SECONDS, 1_700_000_000, 1_700_001_000)

        assert len(df) == 0
