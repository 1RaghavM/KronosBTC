from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from strikecast.constants import CANDLE_COLUMNS, WINDOW_SECONDS


def _make_source_fetch(base_ts: int, total: int, batch: int = 300) -> MagicMock:
    """Create a mock CandleSource that returns batches of candles."""
    call_count = 0

    def fetch(symbol: str, granularity: int, start: int, end: int) -> pd.DataFrame:
        nonlocal call_count
        batch_start = base_ts + call_count * batch * granularity
        n = min(batch, (total - call_count * batch))
        if n <= 0:
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        call_count += 1
        return pd.DataFrame(
            {
                "symbol": symbol,
                "granularity": granularity,
                "window_open_ts": [batch_start + i * granularity for i in range(n)],
                "open": 35000.0,
                "high": 35010.0,
                "low": 34990.0,
                "close": 35005.0,
                "volume": 1.0,
                "amount": 0.0,
                "source": "coinbase",
            }
        )

    mock = MagicMock()
    mock.fetch = MagicMock(side_effect=fetch)
    return mock


class TestPaginator:
    def test_single_batch(self) -> None:
        from strikecast.data.paginator import fetch_all_candles

        base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
        source = _make_source_fetch(base_ts, total=50)

        result = fetch_all_candles(
            source=source,
            symbol="BTC/USD",
            granularity=WINDOW_SECONDS,
            start_ts=base_ts,
            end_ts=base_ts + 50 * WINDOW_SECONDS,
            rate_limit=1000.0,
        )
        assert len(result) == 50

    def test_multiple_batches(self) -> None:
        from strikecast.data.paginator import fetch_all_candles

        base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
        source = _make_source_fetch(base_ts, total=500, batch=300)

        result = fetch_all_candles(
            source=source,
            symbol="BTC/USD",
            granularity=WINDOW_SECONDS,
            start_ts=base_ts,
            end_ts=base_ts + 500 * WINDOW_SECONDS,
            rate_limit=1000.0,
        )
        assert len(result) == 500
        assert source.fetch.call_count == 2

    def test_empty_source(self) -> None:
        from strikecast.data.paginator import fetch_all_candles

        source = MagicMock()
        source.fetch.return_value = pd.DataFrame(columns=CANDLE_COLUMNS)

        result = fetch_all_candles(
            source=source,
            symbol="BTC/USD",
            granularity=WINDOW_SECONDS,
            start_ts=1_700_000_000,
            end_ts=1_700_100_000,
            rate_limit=1000.0,
        )
        assert len(result) == 0
