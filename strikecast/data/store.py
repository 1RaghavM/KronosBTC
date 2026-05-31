from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from strikecast.constants import (
    CANDLE_COLUMNS,
    LABEL_COLUMNS,
    MARKET_COLUMNS,
    WINDOW_SECONDS,
)


class GridAlignmentError(Exception):
    pass


TableName = Literal["candles", "pm_markets", "resolution_labels"]

_SCHEMAS: dict[TableName, list[str]] = {
    "candles": CANDLE_COLUMNS,
    "pm_markets": MARKET_COLUMNS,
    "resolution_labels": LABEL_COLUMNS,
}

_DEDUP_KEYS: dict[TableName, list[str]] = {
    "candles": ["symbol", "granularity", "window_open_ts"],
    "pm_markets": ["window_open_ts"],
    "resolution_labels": ["window_open_ts"],
}


class DataStore:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    def _table_path(self, table: TableName) -> Path:
        return self.data_dir / table / f"{table}.parquet"

    def _validate_grid(self, df: pd.DataFrame) -> None:
        offgrid = df["window_open_ts"] % WINDOW_SECONDS != 0
        if offgrid.any():
            bad_ts = df.loc[offgrid, "window_open_ts"].tolist()
            raise GridAlignmentError(
                f"Timestamps not aligned to {WINDOW_SECONDS}s grid: {bad_ts[:5]}"
            )

    def _validate_columns(self, df: pd.DataFrame, table: TableName) -> None:
        expected = set(_SCHEMAS[table])
        actual = set(df.columns)
        missing = expected - actual
        if missing:
            raise ValueError(f"Missing columns for {table}: {missing}")

    def _write(self, df: pd.DataFrame, table: TableName) -> None:
        self._validate_columns(df, table)
        self._validate_grid(df)

        path = self._table_path(table)
        df = df[_SCHEMAS[table]].copy()

        if path.exists():
            existing = pq.read_table(path).to_pandas()
            df = pd.concat([existing, df], ignore_index=True)

        dedup_keys = _DEDUP_KEYS[table]
        df = df.drop_duplicates(subset=dedup_keys, keep="last")
        df = df.sort_values("window_open_ts").reset_index(drop=True)

        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

    def _read(self, table: TableName) -> pd.DataFrame:
        path = self._table_path(table)
        if not path.exists():
            return pd.DataFrame(columns=_SCHEMAS[table])
        return pq.read_table(path).to_pandas()

    def append_candles(self, df: pd.DataFrame) -> None:
        self._write(df, "candles")

    def read_candles(self) -> pd.DataFrame:
        return self._read("candles")

    def append_markets(self, df: pd.DataFrame) -> None:
        self._write(df, "pm_markets")

    def read_markets(self) -> pd.DataFrame:
        return self._read("pm_markets")

    def append_labels(self, df: pd.DataFrame) -> None:
        self._write(df, "resolution_labels")

    def read_labels(self) -> pd.DataFrame:
        return self._read("resolution_labels")

    def query(self, sql: str) -> pd.DataFrame:
        import duckdb

        conn = duckdb.connect()
        for table in _SCHEMAS:
            path = self._table_path(table)
            if path.exists():
                conn.execute(
                    f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path}')"
                )
        return conn.execute(sql).fetchdf()

    def detect_gaps(
        self, symbol: str = "BTC/USD", granularity: int = WINDOW_SECONDS
    ) -> pd.DataFrame:
        candles = self.read_candles()
        if candles.empty:
            return pd.DataFrame(columns=["gap_start_ts", "gap_end_ts", "missing_count"])

        subset = candles[
            (candles["symbol"] == symbol) & (candles["granularity"] == granularity)
        ]
        if subset.empty:
            return pd.DataFrame(columns=["gap_start_ts", "gap_end_ts", "missing_count"])

        ts = subset["window_open_ts"].sort_values().values
        ts_min, ts_max = int(ts[0]), int(ts[-1])

        expected = set(range(ts_min, ts_max + granularity, granularity))
        actual = set(int(t) for t in ts)
        missing = sorted(expected - actual)

        if not missing:
            return pd.DataFrame(columns=["gap_start_ts", "gap_end_ts", "missing_count"])

        gaps: list[dict[str, int]] = []
        gap_start = missing[0]
        prev = missing[0]
        count = 1

        for m in missing[1:]:
            if m == prev + granularity:
                prev = m
                count += 1
            else:
                gaps.append(
                    {"gap_start_ts": gap_start, "gap_end_ts": prev, "missing_count": count}
                )
                gap_start = m
                prev = m
                count = 1
        gaps.append(
            {"gap_start_ts": gap_start, "gap_end_ts": prev, "missing_count": count}
        )

        return pd.DataFrame(gaps)
