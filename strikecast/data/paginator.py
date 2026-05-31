from __future__ import annotations

import logging
import time

import pandas as pd

from strikecast.constants import CANDLE_COLUMNS
from strikecast.data.candle_source import CandleSource

logger = logging.getLogger(__name__)


def fetch_all_candles(
    source: CandleSource,
    symbol: str,
    granularity: int,
    start_ts: int,
    end_ts: int,
    rate_limit: float = 10.0,
    batch_size: int = 300,
) -> pd.DataFrame:
    all_batches: list[pd.DataFrame] = []
    current_start = start_ts
    request_count = 0

    while current_start < end_ts:
        batch = source.fetch(symbol, granularity, current_start, end_ts)

        if batch.empty:
            break

        all_batches.append(batch)
        request_count += 1

        last_ts = int(batch["window_open_ts"].max())
        next_start = last_ts + granularity

        if next_start <= current_start:
            break
        current_start = next_start

        logger.info(
            "Fetched batch %d: %d candles (up to ts=%d)",
            request_count,
            len(batch),
            last_ts,
        )

        if current_start < end_ts and rate_limit > 0:
            time.sleep(1.0 / rate_limit)

    if not all_batches:
        return pd.DataFrame(columns=CANDLE_COLUMNS)

    result = pd.concat(all_batches, ignore_index=True)
    result = result[
        (result["window_open_ts"] >= start_ts) & (result["window_open_ts"] < end_ts)
    ]
    return result.drop_duplicates(subset=["window_open_ts"]).reset_index(drop=True)
