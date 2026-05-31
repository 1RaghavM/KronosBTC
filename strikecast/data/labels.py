from __future__ import annotations

import pandas as pd

from strikecast.constants import LABEL_COLUMNS


def build_coinbase_labels(candles: pd.DataFrame) -> pd.DataFrame:
    """Derive model-internal resolution labels from Coinbase candles (FR-042).

    The "Up or Down" outcome for a window is ``close > open`` using the
    Coinbase close, which is consistent with the model's training data and
    lets us backtest without any Polymarket dependency.

    Note:
        ``oracle_close`` is set to the Coinbase close as a placeholder so the
        label table satisfies its schema. It is NOT the Chainlink oracle price
        and must not be used for Polymarket paper-PnL (which requires the real
        Chainlink feed; the Coinbase-Chainlink basis is a Phase 4 concern).

    Args:
        candles: Candle frame with ``window_open_ts``, ``open``, ``close``.

    Returns:
        A resolution-label frame with columns
        ``[window_open_ts, oracle_close, coinbase_close, outcome_up]``,
        one row per unique window, sorted by ``window_open_ts``.
    """
    if candles.empty:
        return pd.DataFrame(columns=LABEL_COLUMNS)

    df = (
        candles[["window_open_ts", "open", "close"]]
        .drop_duplicates("window_open_ts")
        .sort_values("window_open_ts")
    )
    close = df["close"].astype(float).to_numpy()
    open_ = df["open"].astype(float).to_numpy()

    return pd.DataFrame(
        {
            "window_open_ts": df["window_open_ts"].astype("int64").to_numpy(),
            "oracle_close": close,
            "coinbase_close": close,
            "outcome_up": close > open_,
        }
    )
