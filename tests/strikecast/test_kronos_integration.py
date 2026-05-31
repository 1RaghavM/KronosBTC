"""Integration test for the Phase 2 Kronos CLI phase.

A fake estimator is injected so the end-to-end wiring (split -> evaluate ->
score -> report) is exercised without loading Kronos weights. The key
contract: Kronos is scored *next to* the baselines on the identical test set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from strikecast.constants import WINDOW_SECONDS


def _make_store_with_data(tmp_path: Path, n: int = 500, seed: int = 42):
    from strikecast.data.store import DataStore

    for subdir in ["candles", "pm_markets", "resolution_labels", "reports"]:
        (tmp_path / subdir).mkdir(exist_ok=True)

    rng = np.random.RandomState(seed)
    base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)

    log_returns = rng.normal(0, 0.001, n)
    log_prices = np.log(35000.0) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
    closes = prices * (1 + rng.normal(0, 0.0005, n))

    candles = pd.DataFrame(
        {
            "symbol": "BTC/USD",
            "granularity": WINDOW_SECONDS,
            "window_open_ts": [base_ts + i * WINDOW_SECONDS for i in range(n)],
            "open": prices,
            "high": prices * (1 + rng.uniform(0.0001, 0.001, n)),
            "low": prices * (1 - rng.uniform(0.0001, 0.001, n)),
            "close": closes,
            "volume": rng.uniform(0.1, 10.0, n),
            "amount": 0.0,
            "source": "coinbase",
        }
    )
    labels = pd.DataFrame(
        {
            "window_open_ts": candles["window_open_ts"],
            "oracle_close": candles["close"],
            "coinbase_close": candles["close"],
            "outcome_up": candles["close"] > candles["open"],
        }
    )

    store = DataStore(tmp_path)
    store.append_candles(candles)
    store.append_labels(labels)
    return store


class _FakeSampler:
    def sample_closes(self, lookback_df, x_timestamp, y_timestamp, sample_count, temperature, top_p):  # noqa: ANN001
        current = float(lookback_df["close"].to_numpy()[-1])
        rng = np.random.RandomState(123)
        return current * np.exp(rng.normal(0.0, 0.002, sample_count))


def _config(tmp_path: Path):
    from strikecast.config import StrikecastConfig

    return StrikecastConfig(
        data={"data_dir": str(tmp_path)},
        estimators={"sample_count": 500, "garch_lookback": 200},
        eval={"train_frac": 0.6, "val_frac": 0.2, "bootstrap_samples": 100},
    )


class TestRunKronosPhase:
    def test_kronos_scored_next_to_baselines(self, tmp_path: Path) -> None:
        from strikecast.cli import run_kronos_phase
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        _make_store_with_data(tmp_path)
        kronos = KronosBinaryEstimator(_FakeSampler(), sample_count=500, seed=42)

        report = run_kronos_phase(_config(tmp_path), kronos_estimator=kronos)

        estimators = {s.estimator for s in report.scores}
        assert "randomwalk" in estimators
        assert "garch_mc" in estimators
        assert "kronos_raw" in estimators
        assert "kronos_cal" in estimators

    def test_report_has_uncal_and_cal_ece(self, tmp_path: Path) -> None:
        from strikecast.cli import run_kronos_phase
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        _make_store_with_data(tmp_path)
        kronos = KronosBinaryEstimator(_FakeSampler(), sample_count=500, seed=42)

        report = run_kronos_phase(_config(tmp_path), kronos_estimator=kronos)

        assert report.ece_uncalibrated is not None
        assert report.ece_calibrated is not None
        assert 0.0 <= report.ece_uncalibrated <= 1.0
        assert 0.0 <= report.ece_calibrated <= 1.0
        assert report.calibration_method == "isotonic"

    def test_reliability_diagram_written(self, tmp_path: Path) -> None:
        from strikecast.cli import run_kronos_phase
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        _make_store_with_data(tmp_path)
        kronos = KronosBinaryEstimator(_FakeSampler(), sample_count=500, seed=42)

        report = run_kronos_phase(_config(tmp_path), kronos_estimator=kronos)

        assert report.reliability_diagram_path is not None
        assert Path(report.reliability_diagram_path).exists()
        assert len(list((tmp_path / "reports").glob("*_calibrator.pkl"))) == 1

    def test_kronos_cal_scores_have_valid_ece(self, tmp_path: Path) -> None:
        from strikecast.cli import run_kronos_phase
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        _make_store_with_data(tmp_path)
        kronos = KronosBinaryEstimator(_FakeSampler(), sample_count=500, seed=42)

        report = run_kronos_phase(_config(tmp_path), kronos_estimator=kronos)
        cal_scores = [s for s in report.scores if s.estimator == "kronos_cal"]
        assert cal_scores
        for s in cal_scores:
            assert 0.0 <= s.ece <= 1.0

    def test_report_records_model_checkpoint(self, tmp_path: Path) -> None:
        from strikecast.cli import run_kronos_phase
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        _make_store_with_data(tmp_path)
        kronos = KronosBinaryEstimator(_FakeSampler(), sample_count=500, seed=42)

        report = run_kronos_phase(_config(tmp_path), kronos_estimator=kronos)
        assert report.model_checkpoint == "NeoQuasar/Kronos-small"

    def test_writes_report_files(self, tmp_path: Path) -> None:
        from strikecast.cli import run_kronos_phase
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        _make_store_with_data(tmp_path)
        kronos = KronosBinaryEstimator(_FakeSampler(), sample_count=500, seed=42)

        run_kronos_phase(_config(tmp_path), kronos_estimator=kronos)

        reports_dir = tmp_path / "reports"
        assert len(list(reports_dir.glob("*.json"))) == 1
        assert len(list(reports_dir.glob("*.md"))) == 1

    def test_all_scores_valid(self, tmp_path: Path) -> None:
        from strikecast.cli import run_kronos_phase
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        _make_store_with_data(tmp_path)
        kronos = KronosBinaryEstimator(_FakeSampler(), sample_count=500, seed=42)

        report = run_kronos_phase(_config(tmp_path), kronos_estimator=kronos)
        for s in report.scores:
            assert 0.0 <= s.brier <= 1.0
            assert s.logloss >= 0.0
            assert s.n_windows > 0

    def test_self_contained_backtest_without_stored_labels(self, tmp_path: Path) -> None:
        """Coinbase label source: a real backtest needs only candles, no Polymarket."""
        from strikecast.cli import run_kronos_phase
        from strikecast.data.store import DataStore
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        for sub in ["candles", "pm_markets", "resolution_labels", "reports"]:
            (tmp_path / sub).mkdir(exist_ok=True)

        rng = np.random.RandomState(7)
        base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
        n = 500
        prices = np.exp(np.log(35000.0) + np.cumsum(rng.normal(0, 0.001, n)))
        candles = pd.DataFrame(
            {
                "symbol": "BTC/USD",
                "granularity": WINDOW_SECONDS,
                "window_open_ts": [base_ts + i * WINDOW_SECONDS for i in range(n)],
                "open": prices,
                "high": prices * 1.001,
                "low": prices * 0.999,
                "close": prices * (1 + rng.normal(0, 0.0005, n)),
                "volume": 1.0,
                "amount": 0.0,
                "source": "coinbase",
            }
        )
        DataStore(tmp_path).append_candles(candles)  # NOTE: no labels written

        kronos = KronosBinaryEstimator(_FakeSampler(), sample_count=300, seed=42)
        report = run_kronos_phase(_config(tmp_path), kronos_estimator=kronos)

        assert report.label_source == "coinbase"
        assert len(report.scores) > 0
        assert {"randomwalk", "garch_mc", "kronos_raw", "kronos_cal"} <= {
            s.estimator for s in report.scores
        }

    def test_max_test_windows_cap(self, tmp_path: Path) -> None:
        from strikecast.cli import run_kronos_phase
        from strikecast.config import StrikecastConfig
        from strikecast.estimators.kronos_binary import KronosBinaryEstimator

        _make_store_with_data(tmp_path)
        config = StrikecastConfig(
            data={"data_dir": str(tmp_path)},
            estimators={"sample_count": 200, "garch_lookback": 200},
            eval={
                "train_frac": 0.6,
                "val_frac": 0.2,
                "bootstrap_samples": 100,
                "max_test_windows": 10,
            },
        )
        kronos = KronosBinaryEstimator(_FakeSampler(), sample_count=200, seed=42)
        report = run_kronos_phase(config, kronos_estimator=kronos)

        for s in report.scores:
            if s.moneyness_bucket == "all":
                assert s.n_windows <= 10
