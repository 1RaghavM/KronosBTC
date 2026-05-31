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


class TestGapDetection:
    def test_no_gaps_returns_empty(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_candles(sample_candles)
        gaps = store.detect_gaps()
        assert len(gaps) == 0

    def test_detects_single_gap(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        gapped = sample_candles.drop(index=[5, 6, 7]).reset_index(drop=True)
        store = DataStore(tmp_data_dir)
        store.append_candles(gapped)
        gaps = store.detect_gaps()

        assert len(gaps) == 1
        assert gaps.iloc[0]["missing_count"] == 3

    def test_detects_multiple_gaps(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        gapped = sample_candles.drop(index=[3, 10, 11]).reset_index(drop=True)
        store = DataStore(tmp_data_dir)
        store.append_candles(gapped)
        gaps = store.detect_gaps()

        assert len(gaps) == 2
        assert gaps.iloc[0]["missing_count"] == 1
        assert gaps.iloc[1]["missing_count"] == 2

    def test_empty_store_returns_empty_gaps(self, tmp_data_dir: Path) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        gaps = store.detect_gaps()
        assert len(gaps) == 0


class TestMarketStore:
    def test_append_and_read_round_trip(
        self, tmp_data_dir: Path, sample_markets: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_markets(sample_markets)
        result = store.read_markets()

        assert len(result) == len(sample_markets)
        pd.testing.assert_frame_equal(
            result.sort_values("window_open_ts").reset_index(drop=True),
            sample_markets.sort_values("window_open_ts").reset_index(drop=True),
        )

    def test_deduplicates_on_rewrite(
        self, tmp_data_dir: Path, sample_markets: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_markets(sample_markets)
        store.append_markets(sample_markets)
        result = store.read_markets()
        assert len(result) == len(sample_markets)


class TestLabelStore:
    def test_append_and_read_round_trip(
        self, tmp_data_dir: Path, sample_labels: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_labels(sample_labels)
        result = store.read_labels()

        assert len(result) == len(sample_labels)
        pd.testing.assert_frame_equal(
            result.sort_values("window_open_ts").reset_index(drop=True),
            sample_labels.sort_values("window_open_ts").reset_index(drop=True),
        )

    def test_rejects_off_grid_label(
        self, tmp_data_dir: Path, sample_labels: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore, GridAlignmentError

        bad = sample_labels.copy()
        bad.loc[0, "window_open_ts"] = bad.loc[0, "window_open_ts"] + 7

        store = DataStore(tmp_data_dir)
        with pytest.raises(GridAlignmentError):
            store.append_labels(bad)


class TestDuckDBQuery:
    def test_query_candles(
        self, tmp_data_dir: Path, sample_candles: pd.DataFrame
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_candles(sample_candles)
        result = store.query("SELECT count(*) AS n FROM candles")
        assert result.iloc[0]["n"] == len(sample_candles)

    def test_query_join_candles_and_labels(
        self,
        tmp_data_dir: Path,
        sample_candles: pd.DataFrame,
        sample_labels: pd.DataFrame,
    ) -> None:
        from strikecast.data.store import DataStore

        store = DataStore(tmp_data_dir)
        store.append_candles(sample_candles)
        store.append_labels(sample_labels)

        result = store.query(
            """
            SELECT c.window_open_ts, c.close, l.oracle_close
            FROM candles c
            JOIN resolution_labels l ON c.window_open_ts = l.window_open_ts
            """
        )
        assert len(result) == len(sample_labels)
