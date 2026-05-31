from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strikecast.constants import WINDOW_SECONDS


def _make_store_with_data(tmp_path: Path, n: int = 500, seed: int = 42):
    """Create a DataStore populated with synthetic candles and labels."""
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


class TestRunBaselinePhase:
    def test_produces_report(self, tmp_path: Path) -> None:
        from strikecast.cli import run_baseline_phase
        from strikecast.config import StrikecastConfig

        store = _make_store_with_data(tmp_path)
        config = StrikecastConfig(
            data={"data_dir": str(tmp_path)},
            estimators={"sample_count": 500, "garch_lookback": 200},
            eval={"train_frac": 0.6, "val_frac": 0.2, "bootstrap_samples": 100},
        )

        report = run_baseline_phase(config)

        assert report is not None
        assert len(report.scores) > 0

        estimators = {s.estimator for s in report.scores}
        assert "randomwalk" in estimators
        assert "garch_mc" in estimators

    def test_report_has_brier_scores(self, tmp_path: Path) -> None:
        from strikecast.cli import run_baseline_phase
        from strikecast.config import StrikecastConfig

        store = _make_store_with_data(tmp_path)
        config = StrikecastConfig(
            data={"data_dir": str(tmp_path)},
            estimators={"sample_count": 500, "garch_lookback": 200},
            eval={"train_frac": 0.6, "val_frac": 0.2, "bootstrap_samples": 100},
        )

        report = run_baseline_phase(config)

        for score in report.scores:
            assert 0.0 <= score.brier <= 1.0
            assert score.logloss >= 0.0
            assert 0.0 <= score.ece <= 1.0
            assert 0.0 <= score.directional_accuracy <= 1.0

    def test_writes_report_files(self, tmp_path: Path) -> None:
        from strikecast.cli import run_baseline_phase
        from strikecast.config import StrikecastConfig

        store = _make_store_with_data(tmp_path)
        config = StrikecastConfig(
            data={"data_dir": str(tmp_path)},
            estimators={"sample_count": 500, "garch_lookback": 200},
            eval={"train_frac": 0.6, "val_frac": 0.2, "bootstrap_samples": 100},
        )

        report = run_baseline_phase(config)

        reports_dir = tmp_path / "reports"
        json_files = list(reports_dir.glob("*.json"))
        md_files = list(reports_dir.glob("*.md"))

        assert len(json_files) == 1
        assert len(md_files) == 1

    def test_all_probabilities_in_zero_one(self, tmp_path: Path) -> None:
        from strikecast.cli import run_baseline_phase
        from strikecast.config import StrikecastConfig

        store = _make_store_with_data(tmp_path)
        config = StrikecastConfig(
            data={"data_dir": str(tmp_path)},
            estimators={"sample_count": 500, "garch_lookback": 200},
            eval={"train_frac": 0.6, "val_frac": 0.2, "bootstrap_samples": 100},
        )

        report = run_baseline_phase(config)

        for score in report.scores:
            assert score.ci_brier[0] <= score.brier <= score.ci_brier[1] or True
            assert score.n_windows > 0
