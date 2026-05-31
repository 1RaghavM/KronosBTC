"""FR-007: Every stored window_open_ts must be aligned to the 300s grid.

Loads all three Parquet stores and verifies that every timestamp
satisfies ts % 300 == 0.
"""
from pathlib import Path

import pandas as pd
import pytest

from strikecast.constants import WINDOW_SECONDS


class TestGridAlignment:
    def test_candle_timestamps_aligned(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_candles(sample_candles)
        candles = store.read_candles()

        misaligned = candles[candles["window_open_ts"] % WINDOW_SECONDS != 0]
        assert len(misaligned) == 0, (
            f"Found {len(misaligned)} candle timestamps not aligned to "
            f"{WINDOW_SECONDS}s grid: {misaligned['window_open_ts'].tolist()[:5]}"
        )

    def test_market_timestamps_aligned(
        self, tmp_data_dir: Path, sample_markets: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_markets(sample_markets)
        markets = store.read_markets()

        misaligned = markets[markets["window_open_ts"] % WINDOW_SECONDS != 0]
        assert len(misaligned) == 0, (
            f"Found {len(misaligned)} market timestamps not aligned to "
            f"{WINDOW_SECONDS}s grid"
        )

    def test_label_timestamps_aligned(
        self, tmp_data_dir: Path, sample_labels: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_labels(sample_labels)
        labels = store.read_labels()

        misaligned = labels[labels["window_open_ts"] % WINDOW_SECONDS != 0]
        assert len(misaligned) == 0, (
            f"Found {len(misaligned)} label timestamps not aligned to "
            f"{WINDOW_SECONDS}s grid"
        )
