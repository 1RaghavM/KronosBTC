from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from strikecast.calibration.calibrator import fit_calibrator
from strikecast.calibration.reliability import make_reliability_diagram
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

_PredRow = dict[str, object]


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
        "--max-val-windows",
        type=int,
        default=None,
        help="Cap number of validation windows used to fit the calibrator",
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


def _predict_over_windows(
    candles: pd.DataFrame,
    lookback_pool: pd.DataFrame,
    lookback_ts: np.ndarray,
    target_timestamps: list[int],
    estimators: list[tuple[str, Estimator]],
    garch_lookback: int,
    min_lookback: int = 50,
) -> list[_PredRow]:
    """Generate one prediction row per (window, estimator) using a trailing lookback.

    The lookback for a target window is the trailing ``garch_lookback`` candles
    **strictly before** that window, drawn from the full contiguous candle
    series (``lookback_pool``). Only past bars are reachable and the target
    window's own close (the label) is never included, so this is leakage-safe
    (NFR-002). Using the full series — rather than the train split — is critical:
    test/val windows fall after the train range, so a train-only pool would feed
    every estimator a stale, wrong-priced context and collapse every probability
    to 0/1. Used for both the test windows and the disjoint validation windows
    that fit the calibrator (FR-020).

    Args:
        candles: Full candle frame (for reading each target window's open).
        lookback_pool: Full candle series (deduped, sorted by time) used as the
            lookback pool for every target window.
        lookback_ts: ``window_open_ts`` array of ``lookback_pool`` (sorted).
        target_timestamps: Windows to predict (Unix epoch seconds).
        estimators: ``(name, estimator)`` pairs to run on every window.
        garch_lookback: Max number of trailing candles in the lookback.
        min_lookback: Skip windows whose lookback has fewer than this many rows.

    Returns:
        List of prediction dicts (one per window/estimator), each holding the
        raw and (estimator-)calibrated probability plus the bootstrap CI.
    """
    predictions: list[_PredRow] = []
    for i, target_ts in enumerate(target_timestamps):
        window = candles[candles["window_open_ts"] == target_ts]
        if window.empty:
            continue

        strike = float(window.iloc[0]["open"])

        # side="left" => index of target_ts itself, so the slice ends strictly
        # before it (the target window's close is excluded -> no leakage).
        idx = int(np.searchsorted(lookback_ts, target_ts, side="left"))
        start_idx = max(0, idx - garch_lookback)
        lookback = lookback_pool.iloc[start_idx:idx]

        if len(lookback) < min_lookback:
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
            logger.info("Evaluated %d / %d windows", i + 1, len(target_timestamps))

    return predictions


@dataclass
class _KronosCalibrationOutput:
    """Bundle returned by :func:`_calibrate_and_score_kronos`."""

    prediction_rows: list[_PredRow]
    reliability_path: str | None
    ece_uncalibrated: float | None
    ece_calibrated: float | None
    calibration_method: str | None


def _calibrate_and_score_kronos(
    *,
    config: StrikecastConfig,
    kronos_estimator: Estimator,
    candles: pd.DataFrame,
    labels: pd.DataFrame,
    lookback_pool: pd.DataFrame,
    lookback_ts: np.ndarray,
    val_timestamps: list[int],
    test_timestamps: list[int],
    model_checkpoint: str | None,
    reports_dir: Path,
    run_id: str,
) -> _KronosCalibrationOutput:
    """Fit the calibrator on the val split and score raw + calibrated Kronos.

    The calibrator is fit ONLY on the validation split (disjoint from train and
    test, FR-020) using Kronos raw probabilities and their outcomes. It is then
    applied to the test-split raw probabilities (FR-021). A CORP reliability
    diagram and uncalibrated/calibrated ECE are written for audit (FR-022), and
    the fitted map is persisted as a versioned artifact (FR-023).

    Args:
        config: Loaded Strikecast config.
        kronos_estimator: Raw (uncalibrated) Kronos estimator.
        candles: Full candle frame.
        labels: Outcome labels keyed by ``window_open_ts``.
        lookback_pool: Full contiguous candle series used for trailing lookbacks.
        lookback_ts: ``window_open_ts`` array of ``lookback_pool``.
        val_timestamps: Validation window timestamps (calibration set).
        test_timestamps: Test window timestamps (scored set).
        model_checkpoint: Checkpoint id stored on the calibrator artifact.
        reports_dir: Directory for the reliability PNG and calibrator artifact.
        run_id: Run identifier used to name artifacts.

    Returns:
        A :class:`_KronosCalibrationOutput` with ``kronos_raw`` + ``kronos_cal``
        prediction rows and calibration diagnostics.
    """
    data_window = (config.data.start, config.data.end)
    method = config.calibration.method

    val_rows = _predict_over_windows(
        candles,
        lookback_pool,
        lookback_ts,
        val_timestamps,
        [("kronos_val", kronos_estimator)],
        config.estimators.garch_lookback,
    )
    test_rows = _predict_over_windows(
        candles,
        lookback_pool,
        lookback_ts,
        test_timestamps,
        [("kronos_raw", kronos_estimator)],
        config.estimators.garch_lookback,
    )

    label_outcomes = labels.set_index("window_open_ts")["outcome_up"].astype(float)

    val_df = pd.DataFrame(val_rows)
    if val_df.empty:
        logger.warning(
            "No Kronos validation predictions; skipping calibration (kronos_cal == kronos_raw)"
        )
        fallback_cal_rows = [dict(r, estimator="kronos_cal") for r in test_rows]
        return _KronosCalibrationOutput(
            prediction_rows=test_rows + fallback_cal_rows,
            reliability_path=None,
            ece_uncalibrated=None,
            ece_calibrated=None,
            calibration_method=None,
        )

    val_df = val_df.join(label_outcomes, on="window_open_ts").dropna(subset=["outcome_up"])
    val_p_raw = val_df["p_raw"].to_numpy(dtype=float)
    val_y = val_df["outcome_up"].to_numpy(dtype=float)

    calibrator = fit_calibrator(
        method,
        val_p_raw,
        val_y,
        model_checkpoint=model_checkpoint,
        data_window=data_window,
    )
    logger.info(
        "Fit %s calibrator on %d disjoint validation windows", method, len(val_p_raw)
    )

    artifact_path = reports_dir / f"{run_id}_calibrator.pkl"
    calibrator.save(artifact_path)

    cal_rows: list[_PredRow] = []
    for row in test_rows:
        p_raw_val = cast(float, row["p_raw"])
        p_cal = float(np.asarray(calibrator.apply(p_raw_val)).reshape(-1)[0])
        cal_row = dict(row)
        cal_row["estimator"] = "kronos_cal"
        cal_row["p"] = p_cal
        cal_rows.append(cal_row)

    reliability_path: str | None = None
    ece_uncalibrated: float | None = None
    ece_calibrated: float | None = None

    test_df = pd.DataFrame(test_rows).join(label_outcomes, on="window_open_ts")
    test_df = test_df.dropna(subset=["outcome_up"])
    if not test_df.empty:
        p_raw_arr = test_df["p_raw"].to_numpy(dtype=float)
        y_arr = test_df["outcome_up"].to_numpy(dtype=float)
        p_cal_arr = np.asarray(calibrator.apply(p_raw_arr), dtype=float)
        png_path = reports_dir / f"{run_id}_reliability.png"
        rel = make_reliability_diagram(p_raw_arr, p_cal_arr, y_arr, png_path)
        reliability_path = str(rel.png_path)
        ece_uncalibrated = rel.ece_raw
        ece_calibrated = rel.ece_cal
        logger.info(
            "Kronos ECE: raw=%.4f, calibrated=%.4f (reliability diagram: %s)",
            ece_uncalibrated,
            ece_calibrated,
            reliability_path,
        )

    return _KronosCalibrationOutput(
        prediction_rows=test_rows + cal_rows,
        reliability_path=reliability_path,
        ece_uncalibrated=ece_uncalibrated,
        ece_calibrated=ece_calibrated,
        calibration_method=method,
    )


def _evaluate_estimators(
    config: StrikecastConfig,
    estimators: list[tuple[str, Estimator]],
    *,
    kronos_estimator: Estimator | None = None,
    model_checkpoint: str | None = None,
    kill_criterion_passed: bool | None = None,
) -> RunReport:
    """Run the walk-forward evaluation for a set of estimators and report.

    Splits the candle store, generates a probability per estimator for every
    test window using a trailing lookback, scores all estimators on the
    identical test windows/labels, then writes JSON + Markdown reports.

    When ``kronos_estimator`` is provided (Phase 3), a calibrator is fit on the
    **disjoint validation split** (FR-020) from the Kronos raw probabilities and
    their outcomes, then both ``kronos_raw`` and ``kronos_cal`` are scored on the
    test set next to the baselines (FR-021), and a CORP reliability diagram plus
    uncalibrated/calibrated ECE are attached to the report (FR-022).

    Args:
        config: Loaded Strikecast config.
        estimators: ``(name, estimator)`` pairs scored side by side. The first
            entry is used as the Brier-skill-score reference baseline.
        kronos_estimator: Optional zero-shot/fine-tuned Kronos estimator. When
            set, its raw probabilities are calibrated on the val split.
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

    # Lookback pool = the full contiguous candle series (strictly-past slices are
    # taken per window inside _predict_over_windows). The train/val/test split
    # only decides which windows are scored / used to fit the calibrator; it must
    # NOT restrict the lookback context, or test windows get a stale train-tail
    # context and every probability collapses to 0/1.
    lookback_pool = (
        candles.drop_duplicates("window_open_ts")
        .sort_values("window_open_ts")
        .reset_index(drop=True)
    )
    lookback_ts = lookback_pool["window_open_ts"].to_numpy()

    test_timestamps = sorted(splits.test)
    if config.eval.max_test_windows is not None:
        test_timestamps = test_timestamps[: config.eval.max_test_windows]
        logger.info(
            "Capping evaluation to first %d test windows (max_test_windows)",
            len(test_timestamps),
        )

    val_timestamps = sorted(splits.val)
    if config.eval.max_val_windows is not None:
        # Keep the most recent val windows (closest to the test period) for calibration.
        val_timestamps = val_timestamps[-config.eval.max_val_windows :]
        logger.info(
            "Capping calibration to last %d validation windows (max_val_windows)",
            len(val_timestamps),
        )

    predictions = _predict_over_windows(
        candles,
        lookback_pool,
        lookback_ts,
        test_timestamps,
        estimators,
        config.estimators.garch_lookback,
    )
    logger.info(
        "Generated %d baseline predictions across %d estimators",
        len(predictions),
        len(estimators),
    )

    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    git_commit = get_git_commit()
    run_id = make_run_id(git_commit)

    reliability_path: str | None = None
    ece_uncalibrated: float | None = None
    ece_calibrated: float | None = None
    calibration_method: str | None = None

    if kronos_estimator is not None:
        kronos_rows = _calibrate_and_score_kronos(
            config=config,
            kronos_estimator=kronos_estimator,
            candles=candles,
            labels=labels,
            lookback_pool=lookback_pool,
            lookback_ts=lookback_ts,
            val_timestamps=val_timestamps,
            test_timestamps=test_timestamps,
            model_checkpoint=model_checkpoint,
            reports_dir=reports_dir,
            run_id=run_id,
        )
        predictions.extend(kronos_rows.prediction_rows)
        reliability_path = kronos_rows.reliability_path
        ece_uncalibrated = kronos_rows.ece_uncalibrated
        ece_calibrated = kronos_rows.ece_calibrated
        calibration_method = kronos_rows.calibration_method

    pred_df = pd.DataFrame(predictions)
    logger.info("Total predictions: %d rows", len(pred_df))

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
        reliability_diagram_path=reliability_path,
        ece_uncalibrated=ece_uncalibrated,
        ece_calibrated=ece_calibrated,
        calibration_method=calibration_method,
    )

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
        The persisted :class:`RunReport` covering randomwalk, garch_mc,
        kronos_raw, and kronos_cal on the identical test set, with the
        calibrator fit on the disjoint validation split.
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
    ]
    return _evaluate_estimators(
        config,
        estimators,
        kronos_estimator=kronos_estimator,
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
    if args.max_val_windows is not None:
        config.eval.max_val_windows = args.max_val_windows
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
