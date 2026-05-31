from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np

from strikecast.config import StrikecastConfig, load_config
from strikecast.data.candle_source import CoinbaseSource
from strikecast.data.paginator import fetch_all_candles
from strikecast.data.store import DataStore

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strikecast",
        description="Strikecast: calibrated probability engine for BTC binary outcomes",
    )
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run the Strikecast pipeline")
    run_parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config YAML file",
    )
    run_parser.add_argument(
        "--phase",
        type=str,
        choices=["all", "data", "baseline", "kronos", "finetune", "decision"],
        default="all",
        help="Which phase to run (default: all)",
    )

    return parser


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _ensure_data_dirs(data_dir: Path) -> None:
    for subdir in ["candles", "pm_markets", "resolution_labels", "reports"]:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)


def _ts_from_date(date_str: str) -> int:
    from datetime import datetime, timezone

    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def run_data_phase(config: StrikecastConfig) -> None:
    data_dir = Path(config.data.data_dir)
    _ensure_data_dirs(data_dir)
    store = DataStore(data_dir)

    start_ts = _ts_from_date(config.data.start)
    end_ts = _ts_from_date(config.data.end)

    logger.info("Fetching candles from %s to %s", config.data.start, config.data.end)
    source = CoinbaseSource()
    candles = fetch_all_candles(
        source=source,
        symbol=config.symbol,
        granularity=config.granularity,
        start_ts=start_ts,
        end_ts=end_ts,
        rate_limit=float(config.data.rate_limit_req_per_sec),
    )

    if not candles.empty:
        store.append_candles(candles)
        logger.info("Stored %d candles", len(candles))
    else:
        logger.warning("No candles fetched")

    gaps = store.detect_gaps(symbol=config.symbol, granularity=config.granularity)
    if not gaps.empty:
        total_missing = int(gaps["missing_count"].sum())
        total_expected = (end_ts - start_ts) // config.granularity
        pct = 100.0 * total_missing / total_expected if total_expected > 0 else 0
        logger.warning(
            "Data quality: %d gaps (%d missing windows, %.2f%%)",
            len(gaps),
            total_missing,
            pct,
        )
    else:
        logger.info("Data quality: no gaps detected")

    if config.polymarket.enabled:
        logger.info("Polymarket ingestion: historical mode (Phase 0)")
        from strikecast.data.polymarket_read import (
            fetch_market_metadata,
            fetch_resolution_labels,
        )

        try:
            markets = fetch_market_metadata(start_ts=start_ts, end_ts=end_ts)
            if not markets.empty:
                store.append_markets(markets)
                logger.info("Stored %d Polymarket markets", len(markets))

                candle_data = store.read_candles()
                labels = fetch_resolution_labels(candle_data, markets)
                if not labels.empty:
                    store.append_labels(labels)
                    logger.info("Stored %d resolution labels", len(labels))
            else:
                logger.info("No Polymarket markets found for this window")
        except Exception:
            logger.exception("Polymarket ingestion failed (non-fatal for Phase 0)")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)
    _set_seed(config.seed)
    logger.info("Loaded config from %s (seed=%d)", args.config, config.seed)

    if args.phase in ("all", "data"):
        run_data_phase(config)

    if args.phase == "data":
        logger.info("Phase 0 (data) complete.")
        return

    if args.phase in ("all", "baseline"):
        logger.info("Phase 1 (baseline): not yet implemented")

    if args.phase in ("all", "kronos"):
        logger.info("Phase 2 (kronos): not yet implemented")

    if args.phase in ("all", "finetune"):
        logger.info("Phase 3 (finetune): not yet implemented")

    if args.phase in ("all", "decision"):
        logger.info("Phase 4 (decision): not yet implemented")
