import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestCLIDataPhase:
    def test_run_data_phase_creates_candle_store(self, tmp_path: Path) -> None:
        from strikecast.cli import run_data_phase
        from strikecast.config import StrikecastConfig
        from strikecast.data.store import DataStore

        config = StrikecastConfig(
            data={"data_dir": str(tmp_path), "start": "2026-01-01", "end": "2026-01-02"},
            polymarket={"enabled": False},
        )
        for subdir in ["candles", "pm_markets", "resolution_labels", "reports"]:
            (tmp_path / subdir).mkdir(exist_ok=True)

        candles_df = _make_minimal_candles(start_ts=1735689600, n=10)

        with patch(
            "strikecast.cli.CoinbaseSource"
        ), patch(
            "strikecast.cli.fetch_all_candles", return_value=candles_df
        ):
            run_data_phase(config)

        store = DataStore(tmp_path)
        candles = store.read_candles()
        assert len(candles) > 0

    def test_cli_main_parses_phase_flag(self) -> None:
        from strikecast.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "--config", "config/default.yaml", "--phase", "data"])
        assert args.phase == "data"
        assert args.config == "config/default.yaml"

    def test_cli_main_default_phase_is_all(self) -> None:
        from strikecast.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "--config", "config/default.yaml"])
        assert args.phase == "all"


def _make_minimal_candles(start_ts: int, n: int = 10):
    import pandas as pd
    from strikecast.constants import WINDOW_SECONDS

    start_ts = start_ts - (start_ts % WINDOW_SECONDS)
    return pd.DataFrame(
        {
            "symbol": "BTC/USD",
            "granularity": WINDOW_SECONDS,
            "window_open_ts": [start_ts + i * WINDOW_SECONDS for i in range(n)],
            "open": 42000.0,
            "high": 42010.0,
            "low": 41990.0,
            "close": 42005.0,
            "volume": 1.0,
            "amount": 0.0,
            "source": "coinbase",
        }
    )
