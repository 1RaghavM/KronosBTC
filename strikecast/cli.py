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
from strikecast.data.labels import build_coinbase_labels
from strikecast.data.paginator import fetch_all_candles
from strikecast.data.store import DataStore
from strikecast.estimators.base import Estimator
from strikecast.estimators.garch_mc import GarchMonteCarloEstimator
from strikecast.estimators.kronos_binary import KronosBinaryEstimator
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
    run_parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Override Monte Carlo sample_count (lower for faster MPS/CPU runs)",
    )
    run_parser.add_argument(
        "--max-test-windows",
        type=int,
        default=None,
        help="Cap number of test windows evaluated (for quick laptop backtests)",
    )
    run_parser.add_argument(
        "--label-source",
        type=str,
        choices=["coinbase", "chainlink"],
        default=None,
        help="Override label source (default from config; coinbase = self-contained)",
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


def _evaluate_estimators(
    config: StrikecastConfig,
    estimators: list[tuple[str, Estimator]],
    *,
    model_checkpoint: str | None = None,
    kill_criterion_passed: bool | None = None,
) -> RunReport:
    """Run the walk-forward evaluation for a set of estimators and report.

    Splits the candle store, generates a probability per estimator for every
    test window using a trailing lookback, scores all estimators on the
    identical test windows/labels, then writes JSON + Markdown reports.

    Args:
        config: Loaded Strikecast config.
        estimators: ``(name, estimator)`` pairs scored side by side. The first
            entry is used as the Brier-skill-score reference baseline.
        model_checkpoint: Recorded in the report (e.g. the Kronos HF id).
        kill_criterion_passed: Optional kill-criterion flag for the report.

    Returns:
        The persisted :class:`RunReport`.
    """
    data_dir = Path(config.data.data_dir)
    _ensure_data_dirs(data_dir)
    store = DataStore(data_dir)

    candles = store.read_candles()

    if candles.empty:
        raise RuntimeError("No candles in store. Run Phase 0 (data) first.")

    if config.eval.label_source == "coinbase":
        labels = build_coinbase_labels(candles)
        logger.info("Using Coinbase-close labels (model-internal, %d windows)", len(labels))
    else:
        labels = store.read_labels()
        logger.info("Using Chainlink resolution labels (%d windows)", len(labels))

    if labels.empty:
        raise RuntimeError(
            f"No labels available for label_source='{config.eval.label_source}'. "
            "For Chainlink labels, run Phase 0 Polymarket ingestion first."
        )

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

    predictions: list[dict] = []
    test_timestamps = sorted(splits.test)
    if config.eval.max_test_windows is not None:
        test_timestamps = test_timestamps[: config.eval.max_test_windows]
        logger.info(
            "Capping evaluation to first %d test windows (max_test_windows)",
            len(test_timestamps),
        )

    for i, target_ts in enumerate(test_timestamps):
        window = candles[candles["window_open_ts"] == target_ts]
        if window.empty:
            continue

        strike = float(window.iloc[0]["open"])

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

    reference = estimators[0][0]
    scores = score_predictions(
        pred_df,
        labels,
        reference_estimator=reference,
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
        model_checkpoint=model_checkpoint,
        git_commit=git_commit,
        seed=config.seed,
        scores=scores,
        kill_criterion_passed=kill_criterion_passed,
        timestamp=datetime.now(timezone.utc).isoformat(),
        label_source=config.eval.label_source,
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


def run_baseline_phase(config: StrikecastConfig) -> RunReport:
    """Run Phase 1: baseline estimators on test windows, score, and report."""
    estimators: list[tuple[str, Estimator]] = [
        ("randomwalk", RandomWalkEstimator()),
        (
            "garch_mc",
            GarchMonteCarloEstimator(
                n_samples=config.estimators.sample_count,
                seed=config.seed,
            ),
        ),
    ]
    return _evaluate_estimators(config, estimators)


def _build_kronos_estimator(config: StrikecastConfig) -> KronosBinaryEstimator:
    """Load Kronos weights and build the zero-shot binary estimator."""
    from strikecast.estimators.kronos_adapter import build_kronos_path_sampler

    sampler = build_kronos_path_sampler(
        checkpoint=config.model.checkpoint,
        tokenizer_name=config.model.tokenizer,
        device=config.model.device,
        max_context=config.model.max_context,
        max_batch=config.model.max_batch,
    )
    return KronosBinaryEstimator(
        sampler,
        sample_count=config.estimators.sample_count,
        temperature=config.estimators.temperature,
        top_p=config.estimators.top_p,
        seed=config.seed,
        max_context=config.model.max_context,
    )


def run_kronos_phase(
    config: StrikecastConfig,
    *,
    kronos_estimator: Estimator | None = None,
) -> RunReport:
    """Run Phase 2: Kronos zero-shot, scored next to the baselines.

    Args:
        config: Loaded Strikecast config.
        kronos_estimator: Optional pre-built estimator (used in tests to avoid
            loading model weights). When ``None``, the real Kronos checkpoint
            from the config is loaded.

    Returns:
        The persisted :class:`RunReport` covering randomwalk, garch_mc, and
        kronos on the identical test set.
    """
    if kronos_estimator is None:
        kronos_estimator = _build_kronos_estimator(config)

    estimators: list[tuple[str, Estimator]] = [
        ("randomwalk", RandomWalkEstimator()),
        (
            "garch_mc",
            GarchMonteCarloEstimator(
                n_samples=config.estimators.sample_count,
                seed=config.seed,
            ),
        ),
        ("kronos", kronos_estimator),
    ]
    return _evaluate_estimators(
        config,
        estimators,
        model_checkpoint=config.model.checkpoint,
    )


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

    if args.sample_count is not None:
        config.estimators.sample_count = args.sample_count
    if args.max_test_windows is not None:
        config.eval.max_test_windows = args.max_test_windows
    if args.label_source is not None:
        config.eval.label_source = args.label_source

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
        run_kronos_phase(config)

    if args.phase in ("all", "finetune"):
        logger.info("Phase 3 (finetune): not yet implemented")

    if args.phase in ("all", "decision"):
        logger.info("Phase 4 (decision): not yet implemented")
