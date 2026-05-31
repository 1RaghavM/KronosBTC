from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from strikecast.constants import CANDLE_COLUMNS, WINDOW_SECONDS


@runtime_checkable
class CandleSource(Protocol):
    def fetch(
        self, symbol: str, granularity: int, start: int, end: int
    ) -> pd.DataFrame: ...


_TIMEFRAME_MAP: dict[int, str] = {
    60: "1m",
    300: "5m",
    900: "15m",
    3600: "1h",
    86400: "1d",
}


class CoinbaseSource:
    def __init__(self, exchange: object | None = None) -> None:
        if exchange is None:
            import ccxt

            exchange = ccxt.coinbase()
        self._exchange = exchange

    def fetch(
        self, symbol: str, granularity: int, start: int, end: int
    ) -> pd.DataFrame:
        timeframe = _TIMEFRAME_MAP.get(granularity)
        if timeframe is None:
            raise ValueError(
                f"Unsupported granularity {granularity}. "
                f"Supported: {list(_TIMEFRAME_MAP.keys())}"
            )

        ohlcv = self._exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=start * 1000, limit=300
        )

        if not ohlcv:
            return pd.DataFrame(columns=CANDLE_COLUMNS)

        df = pd.DataFrame(
            ohlcv, columns=["timestamp_ms", "open", "high", "low", "close", "volume"]
        )
        df["window_open_ts"] = (df["timestamp_ms"] // 1000).astype(int)
        df["symbol"] = symbol
        df["granularity"] = granularity
        df["amount"] = 0.0
        df["source"] = "coinbase"

        df = df[(df["window_open_ts"] >= start) & (df["window_open_ts"] < end)]
        return df[CANDLE_COLUMNS].reset_index(drop=True)
