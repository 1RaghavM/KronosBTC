from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from strikecast.config import StrikecastConfig, load_config
from strikecast.data.candle_source import CoinbaseSource
from strikecast.data.paginator import fetch_all_candles
from strikecast.data.store import DataStore
from strikecast.estimators.garch_mc import GarchMonteCarloEstimator
from strikecast.estimators.random_walk import RandomWalkEstimator
from strikecast.eval.report import RunReport, get_git_commit, make_run_id, write_report
from strikecast.eval.scoring import score_predictions
from strikecast.eval.splits import walk_forward_split

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


def run_baseline_phase(config: StrikecastConfig) -> RunReport:
    """Run Phase 1: baseline estimators on test windows, score, and report."""
    data_dir = Path(config.data.data_dir)
    _ensure_data_dirs(data_dir)
    store = DataStore(data_dir)

    candles = store.read_candles()
    labels = store.read_labels()

    if candles.empty:
        raise RuntimeError("No candles in store. Run Phase 0 (data) first.")

    timestamps = sorted(int(t) for t in candles["window_open_ts"].unique())
    splits = walk_forward_split(
        timestamps=timestamps,
        train_frac=config.eval.train_frac,
        val_frac=config.eval.val_frac,
        purge_windows=config.eval.purge_windows,
        embargo_windows=config.eval.embargo_windows,
    )
    logger.info(
        "Split: train=%d, val=%d, test=%d windows",
        len(splits.train),
        len(splits.val),
        len(splits.test),
    )

    train_set = set(splits.train)
    train_candles = candles[candles["window_open_ts"].isin(train_set)].sort_values(
        "window_open_ts"
    )
    train_ts = train_candles["window_open_ts"].values

    rw = RandomWalkEstimator()
    garch = GarchMonteCarloEstimator(
        n_samples=config.estimators.sample_count,
        seed=config.seed,
    )
    estimators: list[tuple[str, object]] = [("randomwalk", rw), ("garch_mc", garch)]

    predictions: list[dict] = []
    test_timestamps = sorted(splits.test)

    for i, target_ts in enumerate(test_timestamps):
        window = candles[candles["window_open_ts"] == target_ts]
        if window.empty:
            continue

        open_price = float(window.iloc[0]["open"])
        strike = open_price

        idx = int(np.searchsorted(train_ts, target_ts, side="left"))
        start_idx = max(0, idx - config.estimators.garch_lookback)
        lookback = train_candles.iloc[start_idx:idx]

        if len(lookback) < 50:
            continue

        for name, est in estimators:
            try:
                result = est.estimate(lookback, strike)
                predictions.append(
                    {
                        "window_open_ts": target_ts,
                        "estimator": name,
                        "strike": strike,
                        "p": result.p,
                        "p_raw": result.p_raw,
                        "p_ci_low": result.ci_low,
                        "p_ci_high": result.ci_high,
                        "n_samples": result.n_samples,
                        "moneyness": 0.0,
                    }
                )
            except Exception:
                logger.exception("Estimator %s failed for ts=%d", name, target_ts)

        if (i + 1) % 50 == 0:
            logger.info("Evaluated %d / %d test windows", i + 1, len(test_timestamps))

    pred_df = pd.DataFrame(predictions)
    logger.info("Generated %d predictions across %d estimators", len(pred_df), len(estimators))

    scores = score_predictions(
        pred_df,
        labels,
        reference_estimator="randomwalk",
        moneyness_near=config.eval.moneyness_near_threshold,
        moneyness_far=config.eval.moneyness_far_threshold,
        n_bootstrap=config.eval.bootstrap_samples,
        seed=config.seed,
    )

    git_commit = get_git_commit()
    run_id = make_run_id(git_commit)

    report = RunReport(
        run_id=run_id,
        data_window=(config.data.start, config.data.end),
        model_checkpoint=None,
        git_commit=git_commit,
        seed=config.seed,
        scores=scores,
        kill_criterion_passed=None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    reports_dir = data_dir / "reports"
    write_report(report, reports_dir)
    logger.info("Report written to %s", reports_dir)

    for s in scores:
        if s.moneyness_bucket == "all":
            logger.info(
                "  %s: Brier=%.4f [%.4f, %.4f], BSS=%s",
                s.estimator,
                s.brier,
                s.ci_brier[0],
                s.ci_brier[1],
                f"{s.brier_skill_score:+.4f}" if s.brier_skill_score is not None else "n/a",
            )

    return report


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
        run_baseline_phase(config)

    if args.phase in ("all", "kronos"):
        logger.info("Phase 2 (kronos): not yet implemented")

    if args.phase in ("all", "finetune"):
        logger.info("Phase 3 (finetune): not yet implemented")

    if args.phase in ("all", "decision"):
        logger.info("Phase 4 (decision): not yet implemented")
