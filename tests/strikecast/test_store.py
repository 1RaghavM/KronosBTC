from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strikecast.constants import WINDOW_SECONDS


class TestCandleStore:
    def test_append_and_read_round_trip(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_candles(sample_candles)
        result = store.read_candles()

        assert len(result) == len(sample_candles)
        assert list(result.columns) == list(sample_candles.columns)
        pd.testing.assert_frame_equal(
            result.sort_values("window_open_ts").reset_index(drop=True),
            sample_candles.sort_values("window_open_ts").reset_index(drop=True),
        )

    def test_rejects_off_grid_timestamps(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore, GridAlignmentError

        bad = sample_candles.copy()
        bad.loc[0, "window_open_ts"] = bad.loc[0, "window_open_ts"] + 1

        store = DataStore(tmp_data_dir)
        with pytest.raises(GridAlignmentError):
            store.append_candles(bad)

    def test_deduplicates_on_rewrite(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_candles(sample_candles)
        store.append_candles(sample_candles)
        result = store.read_candles()

        assert len(result) == len(sample_candles)

    def test_append_extends_existing_data(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        first_half = sample_candles.iloc[:10].copy()
        second_half = sample_candles.iloc[10:].copy()

        store = DataStore(tmp_data_dir)
        store.append_candles(first_half)
        store.append_candles(second_half)
        result = store.read_candles()

        assert len(result) == len(sample_candles)

    def test_rejects_missing_columns(self, tmp_data_dir: Path) -> None:
        from strikecast.data.store import DataStore

        bad = pd.DataFrame({"window_open_ts": [1_700_000_000], "open": [100.0]})
        store = DataStore(tmp_data_dir)
        with pytest.raises(ValueError, match="Missing columns"):
            store.append_candles(bad)

    def test_read_empty_returns_empty_dataframe(self, tmp_data_dir: Path) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        result = store.read_candles()
        assert len(result) == 0
