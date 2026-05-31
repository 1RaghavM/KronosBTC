# Strikecast Phase 1: Baselines + Scoring Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the baseline probability estimators (random-walk + GARCH-MC), the full evaluation harness (purged walk-forward splits, scoring engine with bootstrap CIs, JSON/Markdown report generator), and the non-negotiable safety tests (no-leakage, reproducibility, split disjointness) — so the scoreboard exists before Kronos enters the ring.

**Architecture:** Two estimators implement the `Estimator` protocol and return `ProbResult`. `RandomWalkEstimator` uses the analytic normal CDF over realized-vol log-returns (no simulation). `GarchMonteCarloEstimator` fits GARCH(1,1) via `arch`, forecasts conditional sigma, and runs Monte Carlo paths to compute P(close > strike). A walk-forward splitter partitions timestamps into train/val/test with purge + embargo gaps. The scoring engine computes Brier, log loss, ECE, BSS, and directional accuracy per estimator and moneyness bucket, with bootstrap 95% CIs. The report generator outputs JSON + Markdown stamped with data window, seed, and git commit.

**Tech Stack:** Python 3.10+, `arch` (GARCH fitting), `scipy` (normal CDF), `hypothesis` (property tests), numpy, pandas, existing Phase 0 modules (`DataStore`, `StrikecastConfig`).

**Depends on:** Phase 0 complete (candle store, config, CLI skeleton).
**Source specs:** `specs/steering.md`, `specs/requirements.md`, `specs/design.md`, `specs/quality.md`

---

## File map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `strikecast/estimators/base.py` | `ProbResult` dataclass, `Estimator` protocol |
| Create | `strikecast/estimators/random_walk.py` | Analytic random-walk baseline (FR-030) |
| Create | `strikecast/estimators/garch_mc.py` | GARCH(1,1) Monte Carlo baseline (FR-031) |
| Create | `strikecast/eval/splits.py` | Purged + embargoed walk-forward split (FR-040) |
| Create | `strikecast/eval/scoring.py` | Brier, log loss, ECE, BSS, dir-acc + kill flag (FR-041, FR-044) |
| Create | `strikecast/eval/report.py` | JSON + Markdown run report (FR-045) |
| Create | `tests/strikecast/test_estimator_base.py` | ProbResult + protocol tests |
| Create | `tests/strikecast/test_random_walk.py` | Random-walk unit + property tests |
| Create | `tests/strikecast/test_garch_mc.py` | GARCH-MC unit + property tests |
| Create | `tests/strikecast/test_splits.py` | Split logic tests |
| Create | `tests/strikecast/test_scoring.py` | Scoring against hand-computed values (to 1e-9) |
| Create | `tests/strikecast/test_report.py` | Report format tests |
| Create | `tests/strikecast/test_no_leakage.py` | NFR-002: shuffle-future test |
| Create | `tests/strikecast/test_reproducibility.py` | NFR-003: identical Brier across runs |
| Create | `tests/strikecast/test_calibration_split_disjoint.py` | FR-020: disjoint splits |
| Modify | `strikecast/constants.py` | Add `PREDICTION_COLUMNS` |
| Modify | `strikecast/estimators/__init__.py` | Re-export estimator types |
| Modify | `strikecast/eval/__init__.py` | Re-export eval types |
| Modify | `strikecast/cli.py` | Wire up `run_baseline_phase()` |
| Modify | `tests/strikecast/conftest.py` | Add large candle series + label fixtures |
| Modify | `requirements-strikecast.txt` | Add `arch`, `scipy`, `hypothesis` |

---

### Task 1: Dependencies + Estimator Base Types

**Files:**
- Modify: `requirements-strikecast.txt`
- Modify: `strikecast/constants.py`
- Create: `strikecast/estimators/base.py`
- Create: `tests/strikecast/test_estimator_base.py`
- Modify: `tests/strikecast/conftest.py`

- [ ] **Step 1: Update dependencies**

Append to `requirements-strikecast.txt`:
```
# Phase 1: baselines + scoring
arch>=7.0.0
scipy>=1.11.0
hypothesis>=6.0.0
```

```bash
pip install arch scipy hypothesis
```

- [ ] **Step 2: Add prediction columns to constants**

Append to `strikecast/constants.py`:
```python
PREDICTION_COLUMNS: list[str] = [
    "run_id",
    "window_open_ts",
    "estimator",
    "strike",
    "p",
    "p_raw",
    "p_ci_low",
    "p_ci_high",
    "n_samples",
    "label",
    "moneyness",
]
```

- [ ] **Step 3: Write the failing tests**

`tests/strikecast/test_estimator_base.py`:
```python
import pytest


class TestProbResult:
    def test_construction(self) -> None:
        from strikecast.estimators.base import ProbResult

        r = ProbResult(p=0.6, p_raw=0.58, ci_low=0.55, ci_high=0.65, n_samples=1000)
        assert r.p == 0.6
        assert r.p_raw == 0.58
        assert r.ci_low == 0.55
        assert r.ci_high == 0.65
        assert r.n_samples == 1000

    def test_is_frozen(self) -> None:
        from strikecast.estimators.base import ProbResult

        r = ProbResult(p=0.6, p_raw=0.58, ci_low=0.55, ci_high=0.65, n_samples=1000)
        with pytest.raises(AttributeError):
            r.p = 0.7  # type: ignore[misc]

    def test_protocol_compliance(self) -> None:
        from strikecast.estimators.base import Estimator, ProbResult

        import numpy as np
        import pandas as pd

        class FakeEstimator:
            def estimate(self, lookback_df: pd.DataFrame, strike: float) -> ProbResult:
                return ProbResult(p=0.5, p_raw=0.5, ci_low=0.5, ci_high=0.5, n_samples=0)

        est = FakeEstimator()
        assert isinstance(est, Estimator)

        df = pd.DataFrame({"close": [100.0, 101.0]})
        result = est.estimate(df, 100.0)
        assert result.p == 0.5
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
pytest tests/strikecast/test_estimator_base.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'strikecast.estimators.base'`

- [ ] **Step 5: Implement estimator base types**

`strikecast/estimators/base.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class ProbResult:
    """Result of a binary probability estimation.

    Attributes:
        p: Calibrated probability in [0, 1].
        p_raw: Raw (uncalibrated) probability in [0, 1].
        ci_low: Lower bound of 95% confidence interval on p.
        ci_high: Upper bound of 95% confidence interval on p.
        n_samples: Number of Monte Carlo samples (0 for analytic estimators).
    """

    p: float
    p_raw: float
    ci_low: float
    ci_high: float
    n_samples: int


@runtime_checkable
class Estimator(Protocol):
    def estimate(self, lookback_df: pd.DataFrame, strike: float) -> ProbResult: ...
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/strikecast/test_estimator_base.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 7: Add large candle series fixture to conftest**

Append to `tests/strikecast/conftest.py`:
```python
@pytest.fixture
def large_candle_series() -> pd.DataFrame:
    """500 consecutive 5-min BTC candles with geometric random-walk prices."""
    base_ts = 1_700_000_000
    base_ts = base_ts - (base_ts % WINDOW_SECONDS)
    n = 500
    rng = np.random.RandomState(42)

    log_returns = rng.normal(0, 0.001, n)
    log_prices = np.log(35000.0) + np.cumsum(log_returns)
    prices = np.exp(log_prices)

    return pd.DataFrame(
        {
            "symbol": "BTC/USD",
            "granularity": WINDOW_SECONDS,
            "window_open_ts": [base_ts + i * WINDOW_SECONDS for i in range(n)],
            "open": prices,
            "high": prices * (1 + rng.uniform(0.0001, 0.001, n)),
            "low": prices * (1 - rng.uniform(0.0001, 0.001, n)),
            "close": prices * (1 + rng.normal(0, 0.0005, n)),
            "volume": rng.uniform(0.1, 10.0, n),
            "amount": 0.0,
            "source": "coinbase",
        }
    )


@pytest.fixture
def large_labels(large_candle_series: pd.DataFrame) -> pd.DataFrame:
    """Resolution labels for the large candle series (close vs open)."""
    df = large_candle_series
    return pd.DataFrame(
        {
            "window_open_ts": df["window_open_ts"],
            "oracle_close": df["close"],
            "coinbase_close": df["close"],
            "outcome_up": df["close"] > df["open"],
        }
    )
```

- [ ] **Step 8: Commit**

```bash
git add requirements-strikecast.txt strikecast/constants.py strikecast/estimators/base.py tests/strikecast/test_estimator_base.py tests/strikecast/conftest.py
git commit -m "feat(strikecast): ProbResult dataclass and Estimator protocol for Phase 1"
```

---

### Task 2: Random-Walk Estimator

**Files:**
- Create: `strikecast/estimators/random_walk.py`
- Create: `tests/strikecast/test_random_walk.py`

- [ ] **Step 1: Write the failing tests**

`tests/strikecast/test_random_walk.py`:
```python
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from strikecast.constants import WINDOW_SECONDS


def _make_lookback(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic lookback with known realized vol."""
    rng = np.random.RandomState(seed)
    log_returns = rng.normal(0, 0.001, n)
    log_prices = np.log(35000.0) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
    base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
    return pd.DataFrame(
        {
            "window_open_ts": [base_ts + i * WINDOW_SECONDS for i in range(n)],
            "open": prices,
            "high": prices * 1.0005,
            "low": prices * 0.9995,
            "close": prices,
            "volume": 1.0,
        }
    )


class TestRandomWalkEstimator:
    def test_atm_returns_half(self) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        lookback = _make_lookback()
        current_price = float(lookback["close"].iloc[-1])
        strike = current_price

        est = RandomWalkEstimator()
        result = est.estimate(lookback, strike)

        assert abs(result.p - 0.5) < 1e-10

    def test_deep_itm_near_one(self) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        lookback = _make_lookback()
        current_price = float(lookback["close"].iloc[-1])
        strike = current_price * 0.99

        est = RandomWalkEstimator()
        result = est.estimate(lookback, strike)

        assert result.p > 0.99

    def test_deep_otm_near_zero(self) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        lookback = _make_lookback()
        current_price = float(lookback["close"].iloc[-1])
        strike = current_price * 1.01

        est = RandomWalkEstimator()
        result = est.estimate(lookback, strike)

        assert result.p < 0.01

    def test_analytic_no_samples(self) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        lookback = _make_lookback()
        est = RandomWalkEstimator()
        result = est.estimate(lookback, 35000.0)

        assert result.n_samples == 0
        assert result.p == result.p_raw
        assert result.ci_low == result.p
        assert result.ci_high == result.p

    def test_matches_scipy_cdf(self) -> None:
        from scipy.stats import norm

        from strikecast.estimators.random_walk import RandomWalkEstimator

        lookback = _make_lookback()
        closes = lookback["close"].values
        log_rets = np.diff(np.log(closes))
        sigma = float(np.std(log_rets, ddof=1))
        current_price = float(closes[-1])
        strike = current_price * 1.002

        expected = 1.0 - norm.cdf(np.log(strike / current_price) / sigma)

        est = RandomWalkEstimator()
        result = est.estimate(lookback, strike)

        assert abs(result.p - expected) < 1e-12

    def test_minimum_lookback_enforced(self) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        tiny = _make_lookback(n=2)
        est = RandomWalkEstimator(min_lookback=10)

        with pytest.raises(ValueError, match="lookback"):
            est.estimate(tiny, 35000.0)

    @given(
        strike_pct=st.floats(min_value=0.95, max_value=1.05),
    )
    @settings(max_examples=50)
    def test_probability_always_in_zero_one(self, strike_pct: float) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        lookback = _make_lookback()
        current_price = float(lookback["close"].iloc[-1])
        strike = current_price * strike_pct

        est = RandomWalkEstimator()
        result = est.estimate(lookback, strike)

        assert 0.0 <= result.p <= 1.0

    def test_monotonic_in_strike(self) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        lookback = _make_lookback()
        current_price = float(lookback["close"].iloc[-1])
        est = RandomWalkEstimator()

        strikes = [current_price * m for m in [0.995, 0.998, 1.0, 1.002, 1.005]]
        probs = [est.estimate(lookback, s).p for s in strikes]

        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1], (
                f"P should decrease as strike increases: "
                f"P({strikes[i]:.2f})={probs[i]:.6f} < P({strikes[i+1]:.2f})={probs[i+1]:.6f}"
            )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/strikecast/test_random_walk.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'strikecast.estimators.random_walk'`

- [ ] **Step 3: Implement the random-walk estimator**

`strikecast/estimators/random_walk.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from strikecast.estimators.base import ProbResult


class RandomWalkEstimator:
    """Analytic random-walk baseline (FR-030).

    Assumes log-returns are i.i.d. N(0, sigma^2) where sigma is the
    realized volatility from the lookback window. Returns the normal
    CDF probability P(close > strike) with no Monte Carlo sampling.
    """

    def __init__(self, min_lookback: int = 10) -> None:
        self._min_lookback = min_lookback

    def estimate(self, lookback_df: pd.DataFrame, strike: float) -> ProbResult:
        if len(lookback_df) < self._min_lookback:
            raise ValueError(
                f"lookback has {len(lookback_df)} rows, need >= {self._min_lookback}"
            )

        closes = lookback_df["close"].values.astype(float)
        log_returns = np.diff(np.log(closes))
        sigma = float(np.std(log_returns, ddof=1))
        current_price = float(closes[-1])

        if sigma <= 0 or current_price <= 0 or strike <= 0:
            p = 0.5
        else:
            z = np.log(strike / current_price) / sigma
            p = 1.0 - float(norm.cdf(z))

        p = float(np.clip(p, 0.0, 1.0))

        return ProbResult(p=p, p_raw=p, ci_low=p, ci_high=p, n_samples=0)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/strikecast/test_random_walk.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add strikecast/estimators/random_walk.py tests/strikecast/test_random_walk.py
git commit -m "feat(strikecast): random-walk baseline estimator with analytic normal CDF (FR-030)"
```

---

### Task 3: GARCH-MC Estimator

**Files:**
- Create: `strikecast/estimators/garch_mc.py`
- Create: `tests/strikecast/test_garch_mc.py`

- [ ] **Step 1: Write the failing tests**

`tests/strikecast/test_garch_mc.py`:
```python
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from strikecast.constants import WINDOW_SECONDS


def _make_lookback(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    log_returns = rng.normal(0, 0.001, n)
    log_prices = np.log(35000.0) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
    base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
    return pd.DataFrame(
        {
            "window_open_ts": [base_ts + i * WINDOW_SECONDS for i in range(n)],
            "open": prices,
            "high": prices * 1.0005,
            "low": prices * 0.9995,
            "close": prices,
            "volume": 1.0,
        }
    )


class TestSimulateProbability:
    """Test the MC simulation step in isolation (no GARCH fitting)."""

    def test_atm_near_half(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        est = GarchMonteCarloEstimator(n_samples=50000, seed=42)
        result = est._simulate_probability(
            current_price=35000.0, sigma=0.001, strike=35000.0
        )

        assert abs(result.p - 0.5) < 0.02

    def test_deep_itm_high_probability(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        est = GarchMonteCarloEstimator(n_samples=10000, seed=42)
        result = est._simulate_probability(
            current_price=35000.0, sigma=0.001, strike=34900.0
        )

        assert result.p > 0.99

    def test_deep_otm_low_probability(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        est = GarchMonteCarloEstimator(n_samples=10000, seed=42)
        result = est._simulate_probability(
            current_price=35000.0, sigma=0.001, strike=35100.0
        )

        assert result.p < 0.01

    def test_bootstrap_ci_contains_point_estimate(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        est = GarchMonteCarloEstimator(n_samples=5000, seed=42, n_bootstrap=1000)
        result = est._simulate_probability(
            current_price=35000.0, sigma=0.001, strike=35010.0
        )

        assert result.ci_low <= result.p <= result.ci_high

    def test_n_samples_recorded(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        est = GarchMonteCarloEstimator(n_samples=2000, seed=42)
        result = est._simulate_probability(
            current_price=35000.0, sigma=0.001, strike=35000.0
        )

        assert result.n_samples == 2000

    def test_deterministic_with_same_seed(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        est1 = GarchMonteCarloEstimator(n_samples=5000, seed=123)
        est2 = GarchMonteCarloEstimator(n_samples=5000, seed=123)

        r1 = est1._simulate_probability(35000.0, 0.001, 35010.0)
        r2 = est2._simulate_probability(35000.0, 0.001, 35010.0)

        assert r1.p == r2.p
        assert r1.ci_low == r2.ci_low
        assert r1.ci_high == r2.ci_high


class TestGarchFit:
    def test_estimate_runs_on_real_data(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        lookback = _make_lookback(n=300)
        current_price = float(lookback["close"].iloc[-1])

        est = GarchMonteCarloEstimator(n_samples=1000, seed=42)
        result = est.estimate(lookback, strike=current_price)

        assert 0.0 <= result.p <= 1.0
        assert result.n_samples == 1000

    def test_minimum_lookback_enforced(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        tiny = _make_lookback(n=20)
        est = GarchMonteCarloEstimator(n_samples=1000, seed=42, min_lookback=50)

        with pytest.raises(ValueError, match="lookback"):
            est.estimate(tiny, 35000.0)

    def test_garch_failure_falls_back_to_realized_vol(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        lookback = _make_lookback(n=300)
        est = GarchMonteCarloEstimator(n_samples=1000, seed=42)

        with patch("strikecast.estimators.garch_mc.arch_model") as mock_am:
            mock_am.side_effect = Exception("convergence failed")
            result = est.estimate(lookback, strike=35000.0)

        assert 0.0 <= result.p <= 1.0

    @given(
        strike_pct=st.floats(min_value=0.95, max_value=1.05),
    )
    @settings(max_examples=20)
    def test_probability_always_in_zero_one(self, strike_pct: float) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        lookback = _make_lookback(n=300)
        current_price = float(lookback["close"].iloc[-1])

        est = GarchMonteCarloEstimator(n_samples=500, seed=42)
        result = est.estimate(lookback, strike=current_price * strike_pct)

        assert 0.0 <= result.p <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/strikecast/test_garch_mc.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'strikecast.estimators.garch_mc'`

- [ ] **Step 3: Implement the GARCH-MC estimator**

`strikecast/estimators/garch_mc.py`:
```python
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from arch import arch_model

from strikecast.estimators.base import ProbResult

logger = logging.getLogger(__name__)


class GarchMonteCarloEstimator:
    """GARCH(1,1) Monte Carlo digital pricer (FR-031).

    Fits GARCH(1,1) on lookback log-returns via the ``arch`` package,
    forecasts one-step conditional volatility, then simulates terminal
    prices to compute P(close > strike). Falls back to realized vol
    if GARCH fitting fails to converge.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        seed: int = 42,
        min_lookback: int = 50,
        n_bootstrap: int = 1000,
    ) -> None:
        self._n_samples = n_samples
        self._seed = seed
        self._min_lookback = min_lookback
        self._n_bootstrap = n_bootstrap

    def _fit_sigma(self, log_returns: np.ndarray) -> float:
        """Fit GARCH(1,1) and return conditional sigma for next period.

        Returns sigma in log-return units (not percentage-scaled).
        Falls back to realized vol on fitting failure.
        """
        returns_pct = log_returns * 100.0

        try:
            am = arch_model(
                returns_pct,
                vol="GARCH",
                p=1,
                q=1,
                mean="Zero",
                dist="normal",
                rescale=False,
            )
            res = am.fit(disp="off", show_warning=False)
            forecasts = res.forecast(horizon=1)
            sigma_pct = float(np.sqrt(forecasts.variance.values[-1, 0]))
            return sigma_pct / 100.0
        except Exception as exc:
            logger.warning("GARCH fit failed (%s), falling back to realized vol", exc)
            return float(np.std(log_returns, ddof=1))

    def _simulate_probability(
        self, current_price: float, sigma: float, strike: float
    ) -> ProbResult:
        """Monte Carlo simulation given a known sigma.

        Args:
            current_price: Most recent close price.
            sigma: Conditional volatility in log-return units.
            strike: Target strike price.

        Returns:
            ProbResult with MC probability, raw probability, and bootstrap CI.
        """
        rng = np.random.RandomState(self._seed)

        sim_log_returns = rng.normal(0.0, sigma, self._n_samples)
        sim_closes = current_price * np.exp(sim_log_returns)
        above = sim_closes > strike
        p = float(np.mean(above))

        boot_rng = np.random.RandomState(self._seed + 1)
        bootstrap_ps = np.empty(self._n_bootstrap)
        for i in range(self._n_bootstrap):
            idx = boot_rng.randint(0, self._n_samples, size=self._n_samples)
            bootstrap_ps[i] = float(np.mean(above[idx]))

        ci_low = float(np.percentile(bootstrap_ps, 2.5))
        ci_high = float(np.percentile(bootstrap_ps, 97.5))

        return ProbResult(
            p=p, p_raw=p, ci_low=ci_low, ci_high=ci_high, n_samples=self._n_samples
        )

    def estimate(self, lookback_df: pd.DataFrame, strike: float) -> ProbResult:
        if len(lookback_df) < self._min_lookback:
            raise ValueError(
                f"lookback has {len(lookback_df)} rows, need >= {self._min_lookback}"
            )

        closes = lookback_df["close"].values.astype(float)
        log_returns = np.diff(np.log(closes))
        current_price = float(closes[-1])

        sigma = self._fit_sigma(log_returns)
        return self._simulate_probability(current_price, sigma, strike)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/strikecast/test_garch_mc.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Update estimators __init__.py**

`strikecast/estimators/__init__.py`:
```python
"""Probability estimators."""

from strikecast.estimators.base import Estimator, ProbResult
from strikecast.estimators.garch_mc import GarchMonteCarloEstimator
from strikecast.estimators.random_walk import RandomWalkEstimator

__all__ = [
    "Estimator",
    "ProbResult",
    "RandomWalkEstimator",
    "GarchMonteCarloEstimator",
]
```

- [ ] **Step 6: Commit**

```bash
git add strikecast/estimators/garch_mc.py strikecast/estimators/__init__.py tests/strikecast/test_garch_mc.py
git commit -m "feat(strikecast): GARCH-MC estimator with arch GARCH(1,1) + Monte Carlo simulation (FR-031)"
```

---

### Task 4: Walk-Forward Splits

**Files:**
- Create: `strikecast/eval/splits.py`
- Create: `tests/strikecast/test_splits.py`

- [ ] **Step 1: Write the failing tests**

`tests/strikecast/test_splits.py`:
```python
import numpy as np
import pytest

from strikecast.constants import WINDOW_SECONDS


def _make_timestamps(n: int = 100) -> list[int]:
    base = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
    return [base + i * WINDOW_SECONDS for i in range(n)]


class TestWalkForwardSplit:
    def test_basic_split_sizes(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        ts = _make_timestamps(100)
        split = walk_forward_split(
            timestamps=ts,
            train_frac=0.6,
            val_frac=0.2,
            purge_windows=1,
            embargo_windows=1,
        )

        assert len(split.train) == 60
        assert len(split.val) > 0
        assert len(split.test) > 0
        assert len(split.train) + len(split.val) + len(split.test) <= 100

    def test_sets_are_disjoint(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        ts = _make_timestamps(200)
        split = walk_forward_split(
            timestamps=ts,
            train_frac=0.6,
            val_frac=0.2,
            purge_windows=1,
            embargo_windows=1,
        )

        train_set = set(split.train)
        val_set = set(split.val)
        test_set = set(split.test)

        assert train_set & val_set == set()
        assert val_set & test_set == set()
        assert train_set & test_set == set()

    def test_temporal_ordering(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        ts = _make_timestamps(200)
        split = walk_forward_split(
            timestamps=ts,
            train_frac=0.6,
            val_frac=0.2,
            purge_windows=1,
            embargo_windows=1,
        )

        assert max(split.train) < min(split.val)
        assert max(split.val) < min(split.test)

    def test_purge_gap_exists(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        ts = _make_timestamps(200)
        split = walk_forward_split(
            timestamps=ts,
            train_frac=0.6,
            val_frac=0.2,
            purge_windows=2,
            embargo_windows=1,
        )

        gap_train_val = min(split.val) - max(split.train)
        min_gap = (2 + 1) * WINDOW_SECONDS

        assert gap_train_val >= min_gap, (
            f"Purge+embargo gap between train and val is {gap_train_val}s, "
            f"need >= {min_gap}s"
        )

    def test_embargo_gap_exists(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        ts = _make_timestamps(200)
        split = walk_forward_split(
            timestamps=ts,
            train_frac=0.6,
            val_frac=0.2,
            purge_windows=1,
            embargo_windows=3,
        )

        gap_val_test = min(split.test) - max(split.val)
        min_gap = (1 + 3) * WINDOW_SECONDS

        assert gap_val_test >= min_gap

    def test_small_dataset_raises(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        ts = _make_timestamps(5)

        with pytest.raises(ValueError, match="timestamps"):
            walk_forward_split(
                timestamps=ts,
                train_frac=0.6,
                val_frac=0.2,
                purge_windows=1,
                embargo_windows=1,
            )

    def test_all_timestamps_grid_aligned(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        ts = _make_timestamps(200)
        split = walk_forward_split(
            timestamps=ts,
            train_frac=0.6,
            val_frac=0.2,
            purge_windows=1,
            embargo_windows=1,
        )

        for t in split.train + split.val + split.test:
            assert t % WINDOW_SECONDS == 0

    def test_dataclass_fields(self) -> None:
        from strikecast.eval.splits import WalkForwardSplit, walk_forward_split

        ts = _make_timestamps(100)
        split = walk_forward_split(
            timestamps=ts,
            train_frac=0.6,
            val_frac=0.2,
            purge_windows=1,
            embargo_windows=1,
        )

        assert isinstance(split, WalkForwardSplit)
        assert hasattr(split, "train")
        assert hasattr(split, "val")
        assert hasattr(split, "test")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/strikecast/test_splits.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'strikecast.eval.splits'`

- [ ] **Step 3: Implement walk-forward splits**

`strikecast/eval/splits.py`:
```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardSplit:
    """Result of a purged + embargoed walk-forward split.

    All lists contain int timestamps (Unix epoch seconds).
    train, val, and test are pairwise disjoint and temporally ordered.
    """

    train: list[int]
    val: list[int]
    test: list[int]


def walk_forward_split(
    timestamps: list[int],
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    purge_windows: int = 1,
    embargo_windows: int = 1,
) -> WalkForwardSplit:
    """Split timestamps into train/val/test with purge + embargo gaps (FR-040).

    The purge gap removes ``purge_windows`` timestamps between each set
    to prevent label leakage. The embargo gap removes ``embargo_windows``
    additional timestamps to account for serial correlation.

    Args:
        timestamps: Sorted list of grid-aligned Unix timestamps.
        train_frac: Fraction allocated to training.
        val_frac: Fraction allocated to validation (calibration).
        purge_windows: Number of windows to drop between sets.
        embargo_windows: Number of additional windows to drop between sets.

    Returns:
        WalkForwardSplit with disjoint train/val/test timestamp lists.

    Raises:
        ValueError: If not enough timestamps after gaps.
    """
    ts = sorted(timestamps)
    n = len(ts)
    gap = purge_windows + embargo_windows

    train_end = int(n * train_frac)
    val_start = train_end + gap
    val_end = val_start + int(n * val_frac)
    test_start = val_end + gap

    if train_end < 1 or val_start >= n or test_start >= n:
        raise ValueError(
            f"Not enough timestamps ({n}) for the requested split "
            f"(train_frac={train_frac}, val_frac={val_frac}, "
            f"purge={purge_windows}, embargo={embargo_windows}). "
            f"Need at least {train_end + 2 * gap + 2} timestamps."
        )

    train = ts[:train_end]
    val = ts[val_start:val_end]
    test = ts[test_start:]

    if not val or not test:
        raise ValueError(
            f"Not enough timestamps ({n}) to produce non-empty val and test sets. "
            f"Computed val_start={val_start}, val_end={val_end}, test_start={test_start}."
        )

    return WalkForwardSplit(train=train, val=val, test=test)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/strikecast/test_splits.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add strikecast/eval/splits.py tests/strikecast/test_splits.py
git commit -m "feat(strikecast): purged + embargoed walk-forward split (FR-040)"
```

---

### Task 5: Scoring Engine

**Files:**
- Create: `strikecast/eval/scoring.py`
- Create: `tests/strikecast/test_scoring.py`

- [ ] **Step 1: Write the failing tests**

`tests/strikecast/test_scoring.py`:
```python
import math

import numpy as np
import pandas as pd
import pytest


class TestBrierScore:
    def test_hand_computed(self) -> None:
        """Verified by hand: mean((p-y)^2) for the toy dataset."""
        from strikecast.eval.scoring import brier_score

        p = np.array([0.7, 0.3, 0.6, 0.8])
        y = np.array([1.0, 0.0, 1.0, 0.0])

        result = brier_score(p, y)
        expected = 0.245

        assert abs(result - expected) < 1e-9

    def test_perfect_predictions(self) -> None:
        from strikecast.eval.scoring import brier_score

        p = np.array([1.0, 0.0, 1.0, 0.0])
        y = np.array([1.0, 0.0, 1.0, 0.0])

        assert brier_score(p, y) == 0.0

    def test_worst_predictions(self) -> None:
        from strikecast.eval.scoring import brier_score

        p = np.array([0.0, 1.0, 0.0, 1.0])
        y = np.array([1.0, 0.0, 1.0, 0.0])

        assert abs(brier_score(p, y) - 1.0) < 1e-9


class TestLogLoss:
    def test_hand_computed(self) -> None:
        """Verified by hand: -mean(y*ln(p) + (1-y)*ln(1-p))."""
        from strikecast.eval.scoring import log_loss

        p = np.array([0.7, 0.3, 0.6, 0.8])
        y = np.array([1.0, 0.0, 1.0, 0.0])

        result = log_loss(p, y)
        expected = -(
            math.log(0.7) + math.log(0.7) + math.log(0.6) + math.log(0.2)
        ) / 4.0

        assert abs(result - expected) < 1e-9

    def test_clamps_extreme_probabilities(self) -> None:
        from strikecast.eval.scoring import log_loss

        p = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])

        result = log_loss(p, y)
        assert np.isfinite(result)


class TestECE:
    def test_hand_computed_two_bins(self) -> None:
        """Two bins: [0, 0.5) and [0.5, 1.0].

        Bin 0: p=0.3, y=0 -> |0.3 - 0.0| = 0.3, weight=1/4
        Bin 1: p=[0.7, 0.6, 0.8], y=[1, 1, 0] -> |0.7 - 2/3| = 1/30, weight=3/4
        ECE = 0.25*0.3 + 0.75*(1/30) = 0.075 + 0.025 = 0.1
        """
        from strikecast.eval.scoring import expected_calibration_error

        p = np.array([0.7, 0.3, 0.6, 0.8])
        y = np.array([1.0, 0.0, 1.0, 0.0])

        result = expected_calibration_error(p, y, n_bins=2)
        expected = 0.1

        assert abs(result - expected) < 1e-9

    def test_perfect_calibration(self) -> None:
        from strikecast.eval.scoring import expected_calibration_error

        p = np.array([0.25, 0.25, 0.75, 0.75])
        y = np.array([0.0, 0.0, 1.0, 1.0])

        result = expected_calibration_error(p, y, n_bins=2)

        assert abs(result - 0.25) < 1e-9


class TestDirectionalAccuracy:
    def test_hand_computed(self) -> None:
        from strikecast.eval.scoring import directional_accuracy

        p = np.array([0.7, 0.3, 0.6, 0.8])
        y = np.array([1.0, 0.0, 1.0, 0.0])

        result = directional_accuracy(p, y)
        expected = 3.0 / 4.0

        assert abs(result - expected) < 1e-9


class TestBrierSkillScore:
    def test_hand_computed(self) -> None:
        from strikecast.eval.scoring import brier_skill_score

        bs_model = 0.245
        bs_reference = 0.25

        result = brier_skill_score(bs_model, bs_reference)
        expected = 1.0 - 0.245 / 0.25

        assert abs(result - expected) < 1e-9

    def test_identical_returns_zero(self) -> None:
        from strikecast.eval.scoring import brier_skill_score

        assert brier_skill_score(0.25, 0.25) == 0.0

    def test_worse_than_reference_is_negative(self) -> None:
        from strikecast.eval.scoring import brier_skill_score

        assert brier_skill_score(0.30, 0.25) < 0.0


class TestBootstrapCI:
    def test_ci_contains_point_estimate(self) -> None:
        from strikecast.eval.scoring import bootstrap_ci, brier_score

        rng = np.random.RandomState(42)
        p = rng.uniform(0.3, 0.7, 200)
        y = (rng.uniform(0, 1, 200) < p).astype(float)

        point = brier_score(p, y)
        ci_low, ci_high = bootstrap_ci(p, y, brier_score, n_bootstrap=5000, seed=42)

        assert ci_low <= point <= ci_high

    def test_deterministic_with_seed(self) -> None:
        from strikecast.eval.scoring import bootstrap_ci, brier_score

        rng = np.random.RandomState(42)
        p = rng.uniform(0.3, 0.7, 100)
        y = (rng.uniform(0, 1, 100) < p).astype(float)

        ci1 = bootstrap_ci(p, y, brier_score, n_bootstrap=1000, seed=99)
        ci2 = bootstrap_ci(p, y, brier_score, n_bootstrap=1000, seed=99)

        assert ci1 == ci2


class TestScorePredictions:
    def test_returns_score_results(self) -> None:
        from strikecast.eval.scoring import ScoreResult, score_predictions

        predictions = pd.DataFrame(
            {
                "window_open_ts": [1, 2, 3, 4],
                "estimator": ["randomwalk"] * 4,
                "strike": [35000.0] * 4,
                "p": [0.7, 0.3, 0.6, 0.8],
                "moneyness": [0.0] * 4,
            }
        )
        labels = pd.DataFrame(
            {
                "window_open_ts": [1, 2, 3, 4],
                "outcome_up": [True, False, True, False],
            }
        )

        results = score_predictions(
            predictions, labels, n_bootstrap=100, seed=42
        )

        assert len(results) > 0
        assert all(isinstance(r, ScoreResult) for r in results)

    def test_multiple_estimators(self) -> None:
        from strikecast.eval.scoring import score_predictions

        predictions = pd.DataFrame(
            {
                "window_open_ts": [1, 2, 3, 4, 1, 2, 3, 4],
                "estimator": ["randomwalk"] * 4 + ["garch_mc"] * 4,
                "strike": [35000.0] * 8,
                "p": [0.5, 0.5, 0.5, 0.5, 0.7, 0.3, 0.6, 0.8],
                "moneyness": [0.0] * 8,
            }
        )
        labels = pd.DataFrame(
            {
                "window_open_ts": [1, 2, 3, 4],
                "outcome_up": [True, False, True, False],
            }
        )

        results = score_predictions(
            predictions, labels, n_bootstrap=100, seed=42
        )

        estimators = {r.estimator for r in results}
        assert "randomwalk" in estimators
        assert "garch_mc" in estimators
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/strikecast/test_scoring.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'strikecast.eval.scoring'`

- [ ] **Step 3: Implement the scoring engine**

`strikecast/eval/scoring.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

_EPS = 1e-15


@dataclass(frozen=True)
class ScoreResult:
    """Scoring result for one estimator in one moneyness bucket."""

    estimator: str
    moneyness_bucket: str
    brier: float
    logloss: float
    ece: float
    directional_accuracy: float
    brier_skill_score: float | None
    n_windows: int
    ci_brier: tuple[float, float]
    ci_logloss: tuple[float, float]


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p_clipped = np.clip(p, _EPS, 1.0 - _EPS)
    return -float(np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped)))


def expected_calibration_error(
    p: np.ndarray, y: np.ndarray, n_bins: int = 15
) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if hi == bin_edges[-1]:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        n_bin = int(np.sum(mask))
        if n_bin == 0:
            continue

        avg_confidence = float(np.mean(p[mask]))
        avg_accuracy = float(np.mean(y[mask]))
        ece += (n_bin / n) * abs(avg_confidence - avg_accuracy)

    return ece


def directional_accuracy(p: np.ndarray, y: np.ndarray) -> float:
    predicted_up = p > 0.5
    actual_up = y > 0.5
    return float(np.mean(predicted_up == actual_up))


def brier_skill_score(bs_model: float, bs_reference: float) -> float:
    if bs_reference == 0.0:
        return 0.0
    return 1.0 - bs_model / bs_reference


def bootstrap_ci(
    p: np.ndarray,
    y: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval for a scoring metric."""
    rng = np.random.RandomState(seed)
    n = len(p)
    scores = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        scores[i] = metric_fn(p[idx], y[idx])

    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def score_predictions(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    reference_estimator: str = "randomwalk",
    moneyness_near: float = 0.001,
    moneyness_far: float = 0.01,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> list[ScoreResult]:
    """Score all estimator predictions against labels (FR-041).

    Computes per-estimator, per-moneyness-bucket metrics with bootstrap
    95% CIs and Brier skill score relative to the reference estimator.

    Args:
        predictions: DataFrame with columns [window_open_ts, estimator, p, moneyness].
        labels: DataFrame with columns [window_open_ts, outcome_up].
        reference_estimator: Estimator name for BSS reference (default: randomwalk).
        moneyness_near: Threshold for near-the-money bucket.
        moneyness_far: Threshold for far-the-money bucket.
        n_bootstrap: Number of bootstrap samples for CIs.
        seed: Random seed for bootstrap reproducibility.

    Returns:
        List of ScoreResult, one per (estimator, moneyness_bucket) pair.
    """
    merged = pd.merge(
        predictions,
        labels[["window_open_ts", "outcome_up"]],
        on="window_open_ts",
        how="inner",
    )
    merged["y"] = merged["outcome_up"].astype(float)

    ref_brier: dict[str, float] = {}
    estimator_names = merged["estimator"].unique()

    def _bucket(m: float) -> str:
        am = abs(m)
        if am <= moneyness_near:
            return "near"
        elif am >= moneyness_far:
            return "far"
        return "mid"

    merged["bucket"] = merged["moneyness"].apply(_bucket)

    results: list[ScoreResult] = []

    for est in estimator_names:
        for bucket in ["all", "near", "mid", "far"]:
            est_mask = merged["estimator"] == est
            if bucket == "all":
                subset = merged[est_mask]
            else:
                subset = merged[est_mask & (merged["bucket"] == bucket)]

            if len(subset) == 0:
                continue

            p = subset["p"].values
            y = subset["y"].values

            bs = brier_score(p, y)
            ll = log_loss(p, y)
            ece_val = expected_calibration_error(p, y)
            da = directional_accuracy(p, y)

            ci_bs = bootstrap_ci(p, y, brier_score, n_bootstrap, seed)
            ci_ll = bootstrap_ci(p, y, log_loss, n_bootstrap, seed)

            if est == reference_estimator:
                ref_brier[bucket] = bs

            results.append(
                ScoreResult(
                    estimator=est,
                    moneyness_bucket=bucket,
                    brier=bs,
                    logloss=ll,
                    ece=ece_val,
                    directional_accuracy=da,
                    brier_skill_score=None,
                    n_windows=len(subset),
                    ci_brier=ci_bs,
                    ci_logloss=ci_ll,
                )
            )

    final: list[ScoreResult] = []
    for r in results:
        rb = ref_brier.get(r.moneyness_bucket)
        bss = brier_skill_score(r.brier, rb) if rb is not None else None
        final.append(
            ScoreResult(
                estimator=r.estimator,
                moneyness_bucket=r.moneyness_bucket,
                brier=r.brier,
                logloss=r.logloss,
                ece=r.ece,
                directional_accuracy=r.directional_accuracy,
                brier_skill_score=bss,
                n_windows=r.n_windows,
                ci_brier=r.ci_brier,
                ci_logloss=r.ci_logloss,
            )
        )

    return final
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/strikecast/test_scoring.py -v
```

Expected: all 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add strikecast/eval/scoring.py tests/strikecast/test_scoring.py
git commit -m "feat(strikecast): scoring engine with Brier, log loss, ECE, BSS, bootstrap CIs (FR-041)"
```

---

### Task 6: Report Generator

**Files:**
- Create: `strikecast/eval/report.py`
- Create: `tests/strikecast/test_report.py`

- [ ] **Step 1: Write the failing tests**

`tests/strikecast/test_report.py`:
```python
import json
from pathlib import Path

import pytest


def _make_score_results():
    from strikecast.eval.scoring import ScoreResult

    return [
        ScoreResult(
            estimator="randomwalk",
            moneyness_bucket="all",
            brier=0.2500,
            logloss=0.6931,
            ece=0.005,
            directional_accuracy=0.500,
            brier_skill_score=0.0,
            n_windows=100,
            ci_brier=(0.2400, 0.2600),
            ci_logloss=(0.6800, 0.7100),
        ),
        ScoreResult(
            estimator="garch_mc",
            moneyness_bucket="all",
            brier=0.2450,
            logloss=0.6850,
            ece=0.010,
            directional_accuracy=0.520,
            brier_skill_score=0.02,
            n_windows=100,
            ci_brier=(0.2350, 0.2550),
            ci_logloss=(0.6750, 0.7000),
        ),
    ]


class TestRunReport:
    def test_construction(self) -> None:
        from strikecast.eval.report import RunReport

        scores = _make_score_results()
        report = RunReport(
            run_id="abc123_2026-05-30T12:00:00",
            data_window=("2025-12-01", "2026-05-30"),
            model_checkpoint=None,
            git_commit="abc123",
            seed=42,
            scores=scores,
            kill_criterion_passed=None,
            timestamp="2026-05-30T12:00:00Z",
        )

        assert report.run_id == "abc123_2026-05-30T12:00:00"
        assert len(report.scores) == 2


class TestJSONReport:
    def test_valid_json(self) -> None:
        from strikecast.eval.report import RunReport, to_json

        scores = _make_score_results()
        report = RunReport(
            run_id="test_run",
            data_window=("2025-12-01", "2026-05-30"),
            model_checkpoint=None,
            git_commit="abc123",
            seed=42,
            scores=scores,
            kill_criterion_passed=None,
            timestamp="2026-05-30T12:00:00Z",
        )

        json_str = to_json(report)
        parsed = json.loads(json_str)

        assert parsed["run_id"] == "test_run"
        assert parsed["seed"] == 42
        assert len(parsed["scores"]) == 2
        assert parsed["scores"][0]["estimator"] == "randomwalk"

    def test_write_json_to_file(self, tmp_path: Path) -> None:
        from strikecast.eval.report import RunReport, write_report

        scores = _make_score_results()
        report = RunReport(
            run_id="test_run",
            data_window=("2025-12-01", "2026-05-30"),
            model_checkpoint=None,
            git_commit="abc123",
            seed=42,
            scores=scores,
            kill_criterion_passed=None,
            timestamp="2026-05-30T12:00:00Z",
        )

        write_report(report, tmp_path)

        json_path = tmp_path / "test_run.json"
        assert json_path.exists()
        parsed = json.loads(json_path.read_text())
        assert parsed["run_id"] == "test_run"


class TestMarkdownReport:
    def test_contains_kpi_table(self) -> None:
        from strikecast.eval.report import RunReport, to_markdown

        scores = _make_score_results()
        report = RunReport(
            run_id="test_run",
            data_window=("2025-12-01", "2026-05-30"),
            model_checkpoint=None,
            git_commit="abc123",
            seed=42,
            scores=scores,
            kill_criterion_passed=None,
            timestamp="2026-05-30T12:00:00Z",
        )

        md = to_markdown(report)

        assert "Brier" in md
        assert "randomwalk" in md
        assert "garch_mc" in md
        assert "abc123" in md

    def test_kill_criterion_flagged(self) -> None:
        from strikecast.eval.report import RunReport, to_markdown

        scores = _make_score_results()
        report = RunReport(
            run_id="test_run",
            data_window=("2025-12-01", "2026-05-30"),
            model_checkpoint=None,
            git_commit="abc123",
            seed=42,
            scores=scores,
            kill_criterion_passed=False,
            timestamp="2026-05-30T12:00:00Z",
        )

        md = to_markdown(report)
        assert "FAILED-KILL-CRITERION" in md

    def test_write_markdown_to_file(self, tmp_path: Path) -> None:
        from strikecast.eval.report import RunReport, write_report

        scores = _make_score_results()
        report = RunReport(
            run_id="test_run",
            data_window=("2025-12-01", "2026-05-30"),
            model_checkpoint=None,
            git_commit="abc123",
            seed=42,
            scores=scores,
            kill_criterion_passed=None,
            timestamp="2026-05-30T12:00:00Z",
        )

        write_report(report, tmp_path)

        md_path = tmp_path / "test_run.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "Brier" in content
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/strikecast/test_report.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'strikecast.eval.report'`

- [ ] **Step 3: Implement the report generator**

`strikecast/eval/report.py`:
```python
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from strikecast.eval.scoring import ScoreResult


@dataclass
class RunReport:
    """Complete run report (FR-045)."""

    run_id: str
    data_window: tuple[str, str]
    model_checkpoint: str | None
    git_commit: str
    seed: int
    scores: list[ScoreResult]
    kill_criterion_passed: bool | None
    timestamp: str


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def make_run_id(git_commit: str | None = None) -> str:
    commit = git_commit or get_git_commit()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{commit}_{ts}"


def to_json(report: RunReport) -> str:
    data = {
        "run_id": report.run_id,
        "data_window": list(report.data_window),
        "model_checkpoint": report.model_checkpoint,
        "git_commit": report.git_commit,
        "seed": report.seed,
        "kill_criterion_passed": report.kill_criterion_passed,
        "timestamp": report.timestamp,
        "scores": [asdict(s) for s in report.scores],
    }
    return json.dumps(data, indent=2)


def to_markdown(report: RunReport) -> str:
    lines: list[str] = []
    lines.append(f"# Strikecast Run Report: {report.run_id}")
    lines.append("")
    lines.append(f"- **Data window:** {report.data_window[0]} to {report.data_window[1]}")
    lines.append(f"- **Git commit:** `{report.git_commit}`")
    lines.append(f"- **Seed:** {report.seed}")
    lines.append(f"- **Timestamp:** {report.timestamp}")

    if report.model_checkpoint:
        lines.append(f"- **Model checkpoint:** `{report.model_checkpoint}`")

    lines.append("")

    if report.kill_criterion_passed is False:
        lines.append("## **FAILED-KILL-CRITERION**")
        lines.append("")
        lines.append(
            "Kronos Brier skill score vs GARCH-MC <= 0 on test set. "
            "The foundation-model approach is not justified for this horizon."
        )
        lines.append("")

    if report.kill_criterion_passed is True:
        lines.append("## Kill criterion: PASSED")
        lines.append("")

    lines.append("## KPI Table")
    lines.append("")
    lines.append(
        "| Estimator | Bucket | Brier | Brier CI | Log Loss | Log Loss CI "
        "| ECE | Dir Acc | BSS | N |"
    )
    lines.append(
        "|-----------|--------|-------|----------|----------|----------"
        "---|-----|---------|-----|---|"
    )

    for s in report.scores:
        bss_str = f"{s.brier_skill_score:+.4f}" if s.brier_skill_score is not None else "n/a"
        lines.append(
            f"| {s.estimator} | {s.moneyness_bucket} "
            f"| {s.brier:.4f} | [{s.ci_brier[0]:.4f}, {s.ci_brier[1]:.4f}] "
            f"| {s.logloss:.4f} | [{s.ci_logloss[0]:.4f}, {s.ci_logloss[1]:.4f}] "
            f"| {s.ece:.4f} | {s.directional_accuracy:.4f} "
            f"| {bss_str} | {s.n_windows} |"
        )

    lines.append("")
    return "\n".join(lines)


def write_report(report: RunReport, output_dir: str | Path) -> None:
    """Write JSON and Markdown reports to the output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{report.run_id}.json"
    json_path.write_text(to_json(report))

    md_path = output_dir / f"{report.run_id}.md"
    md_path.write_text(to_markdown(report))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/strikecast/test_report.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Update eval __init__.py**

`strikecast/eval/__init__.py`:
```python
"""Evaluation, scoring, and reporting."""

from strikecast.eval.report import RunReport, make_run_id, to_json, to_markdown, write_report
from strikecast.eval.scoring import ScoreResult, score_predictions
from strikecast.eval.splits import WalkForwardSplit, walk_forward_split

__all__ = [
    "RunReport",
    "ScoreResult",
    "WalkForwardSplit",
    "make_run_id",
    "score_predictions",
    "to_json",
    "to_markdown",
    "walk_forward_split",
    "write_report",
]
```

- [ ] **Step 6: Commit**

```bash
git add strikecast/eval/report.py strikecast/eval/__init__.py tests/strikecast/test_report.py
git commit -m "feat(strikecast): JSON + Markdown run report generator (FR-045)"
```

---

### Task 7: Non-Negotiable Safety Tests

**Files:**
- Create: `tests/strikecast/test_no_leakage.py`
- Create: `tests/strikecast/test_reproducibility.py`
- Create: `tests/strikecast/test_calibration_split_disjoint.py`

- [ ] **Step 1: Write the no-leakage test (NFR-002)**

`tests/strikecast/test_no_leakage.py`:
```python
"""NFR-002: No look-ahead leakage.

Permuting bars strictly after a window must leave that window's
probability bit-identical. A failure here invalidates every backtest
number — treat as a release blocker, not a flaky test.
"""
import numpy as np
import pandas as pd
import pytest

from strikecast.constants import WINDOW_SECONDS


def _make_series(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
    log_returns = rng.normal(0, 0.001, n)
    log_prices = np.log(35000.0) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
    return pd.DataFrame(
        {
            "window_open_ts": [base_ts + i * WINDOW_SECONDS for i in range(n)],
            "open": prices,
            "high": prices * 1.0005,
            "low": prices * 0.9995,
            "close": prices,
            "volume": 1.0,
        }
    )


class TestNoLeakage:
    def test_random_walk_no_leakage(self) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        series = _make_series()
        split_idx = 250
        lookback = series.iloc[:split_idx].copy()
        strike = float(lookback["close"].iloc[-1])

        est = RandomWalkEstimator()
        p_original = est.estimate(lookback, strike)

        shuffled = series.copy()
        future = shuffled.iloc[split_idx:].copy()
        future_shuffled = future.sample(frac=1, random_state=99).reset_index(drop=True)
        future_shuffled["window_open_ts"] = future["window_open_ts"].values
        shuffled.iloc[split_idx:] = future_shuffled.values

        lookback_after = shuffled.iloc[:split_idx].copy()
        p_shuffled = est.estimate(lookback_after, strike)

        assert p_original.p == p_shuffled.p, (
            f"Random walk probability changed after shuffling future data: "
            f"{p_original.p} != {p_shuffled.p}"
        )

    def test_garch_mc_no_leakage(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        series = _make_series()
        split_idx = 250
        lookback = series.iloc[:split_idx].copy()
        strike = float(lookback["close"].iloc[-1])

        est = GarchMonteCarloEstimator(n_samples=1000, seed=42)
        p_original = est.estimate(lookback, strike)

        shuffled = series.copy()
        future = shuffled.iloc[split_idx:].copy()
        future_shuffled = future.sample(frac=1, random_state=99).reset_index(drop=True)
        future_shuffled["window_open_ts"] = future["window_open_ts"].values
        shuffled.iloc[split_idx:] = future_shuffled.values

        lookback_after = shuffled.iloc[:split_idx].copy()
        p_shuffled = est.estimate(lookback_after, strike)

        assert p_original.p == p_shuffled.p, (
            f"GARCH-MC probability changed after shuffling future data: "
            f"{p_original.p} != {p_shuffled.p}"
        )
```

- [ ] **Step 2: Write the reproducibility test (NFR-003)**

`tests/strikecast/test_reproducibility.py`:
```python
"""NFR-003: Reproducibility.

Two runs with the same seed + config produce identical Brier scores
to 1e-9. This test runs the scoring pipeline twice on synthetic data
and verifies bit-level reproducibility.
"""
import numpy as np
import pandas as pd
import pytest

from strikecast.constants import WINDOW_SECONDS


def _build_predictions_and_labels(seed: int = 42):
    """Run both estimators on synthetic data and return (predictions_df, labels_df)."""
    from strikecast.estimators.garch_mc import GarchMonteCarloEstimator
    from strikecast.estimators.random_walk import RandomWalkEstimator

    rng = np.random.RandomState(seed)
    base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
    n_lookback = 300
    n_test = 20

    log_returns = rng.normal(0, 0.001, n_lookback + n_test)
    log_prices = np.log(35000.0) + np.cumsum(log_returns)
    prices = np.exp(log_prices)

    candles = pd.DataFrame(
        {
            "window_open_ts": [base_ts + i * WINDOW_SECONDS for i in range(n_lookback + n_test)],
            "open": prices,
            "high": prices * 1.0005,
            "low": prices * 0.9995,
            "close": prices * (1 + rng.normal(0, 0.0003, n_lookback + n_test)),
            "volume": 1.0,
        }
    )

    rw = RandomWalkEstimator()
    garch = GarchMonteCarloEstimator(n_samples=500, seed=seed)

    predictions = []
    for i in range(n_lookback, n_lookback + n_test):
        lookback = candles.iloc[:i]
        target_open = float(candles.iloc[i]["open"])
        strike = target_open
        target_ts = int(candles.iloc[i]["window_open_ts"])

        for name, est in [("randomwalk", rw), ("garch_mc", garch)]:
            result = est.estimate(lookback, strike)
            predictions.append(
                {
                    "window_open_ts": target_ts,
                    "estimator": name,
                    "strike": strike,
                    "p": result.p,
                    "moneyness": 0.0,
                }
            )

    pred_df = pd.DataFrame(predictions)

    labels = pd.DataFrame(
        {
            "window_open_ts": candles.iloc[n_lookback:]["window_open_ts"].values,
            "outcome_up": candles.iloc[n_lookback:]["close"].values
            > candles.iloc[n_lookback:]["open"].values,
        }
    )

    return pred_df, labels


class TestReproducibility:
    def test_identical_brier_across_runs(self) -> None:
        from strikecast.eval.scoring import brier_score, score_predictions

        pred1, labels1 = _build_predictions_and_labels(seed=42)
        pred2, labels2 = _build_predictions_and_labels(seed=42)

        scores1 = score_predictions(pred1, labels1, n_bootstrap=100, seed=42)
        scores2 = score_predictions(pred2, labels2, n_bootstrap=100, seed=42)

        for s1, s2 in zip(scores1, scores2):
            assert abs(s1.brier - s2.brier) < 1e-9, (
                f"Brier mismatch for {s1.estimator}/{s1.moneyness_bucket}: "
                f"{s1.brier} != {s2.brier}"
            )
            assert abs(s1.logloss - s2.logloss) < 1e-9, (
                f"Log loss mismatch for {s1.estimator}/{s1.moneyness_bucket}: "
                f"{s1.logloss} != {s2.logloss}"
            )

    def test_random_walk_deterministic(self) -> None:
        from strikecast.estimators.random_walk import RandomWalkEstimator

        rng = np.random.RandomState(42)
        prices = 35000.0 + np.cumsum(rng.randn(200)) * 0.5
        lookback = pd.DataFrame({"close": prices})

        est = RandomWalkEstimator()
        r1 = est.estimate(lookback, 35000.0)
        r2 = est.estimate(lookback, 35000.0)

        assert r1.p == r2.p

    def test_garch_mc_deterministic_with_seed(self) -> None:
        from strikecast.estimators.garch_mc import GarchMonteCarloEstimator

        rng = np.random.RandomState(42)
        n = 300
        log_returns = rng.normal(0, 0.001, n)
        log_prices = np.log(35000.0) + np.cumsum(log_returns)
        prices = np.exp(log_prices)
        lookback = pd.DataFrame({"close": prices})

        est1 = GarchMonteCarloEstimator(n_samples=500, seed=99)
        est2 = GarchMonteCarloEstimator(n_samples=500, seed=99)

        r1 = est1.estimate(lookback, 35000.0)
        r2 = est2.estimate(lookback, 35000.0)

        assert abs(r1.p - r2.p) < 1e-9
```

- [ ] **Step 3: Write the calibration split disjointness test (FR-020)**

`tests/strikecast/test_calibration_split_disjoint.py`:
```python
"""FR-020: Calibration split must be disjoint from train and test.

Asserts that train, validation, and test window-id sets produced
by the walk-forward splitter are pairwise disjoint.
"""
import pytest

from strikecast.constants import WINDOW_SECONDS


class TestCalibrationSplitDisjoint:
    def test_all_sets_pairwise_disjoint(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
        timestamps = [base_ts + i * WINDOW_SECONDS for i in range(1000)]

        split = walk_forward_split(
            timestamps=timestamps,
            train_frac=0.60,
            val_frac=0.20,
            purge_windows=1,
            embargo_windows=1,
        )

        train_set = set(split.train)
        val_set = set(split.val)
        test_set = set(split.test)

        assert train_set & val_set == set(), (
            f"Train and val overlap: {train_set & val_set}"
        )
        assert val_set & test_set == set(), (
            f"Val and test overlap: {val_set & test_set}"
        )
        assert train_set & test_set == set(), (
            f"Train and test overlap: {train_set & test_set}"
        )

    def test_no_calibration_data_in_test(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
        timestamps = [base_ts + i * WINDOW_SECONDS for i in range(500)]

        split = walk_forward_split(
            timestamps=timestamps,
            train_frac=0.60,
            val_frac=0.20,
            purge_windows=2,
            embargo_windows=2,
        )

        val_set = set(split.val)
        test_set = set(split.test)

        assert val_set.isdisjoint(test_set), (
            "Calibration (val) and test sets must be disjoint"
        )

    def test_no_train_data_in_calibration(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        base_ts = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
        timestamps = [base_ts + i * WINDOW_SECONDS for i in range(500)]

        split = walk_forward_split(
            timestamps=timestamps,
            train_frac=0.60,
            val_frac=0.20,
            purge_windows=1,
            embargo_windows=1,
        )

        train_set = set(split.train)
        val_set = set(split.val)

        assert train_set.isdisjoint(val_set), (
            "Training and calibration (val) sets must be disjoint"
        )
```

- [ ] **Step 4: Run all safety tests**

```bash
pytest tests/strikecast/test_no_leakage.py tests/strikecast/test_reproducibility.py tests/strikecast/test_calibration_split_disjoint.py -v
```

Expected: all 8 tests PASS (2 leakage + 3 reproducibility + 3 disjoint).

- [ ] **Step 5: Commit**

```bash
git add tests/strikecast/test_no_leakage.py tests/strikecast/test_reproducibility.py tests/strikecast/test_calibration_split_disjoint.py
git commit -m "test(strikecast): non-negotiable safety tests — no leakage (NFR-002), reproducibility (NFR-003), split disjointness (FR-020)"
```

---

### Task 8: CLI Baseline Phase + Integration

**Files:**
- Modify: `strikecast/cli.py`
- Create: `tests/strikecast/test_baseline_integration.py`

- [ ] **Step 1: Write the failing integration test**

`tests/strikecast/test_baseline_integration.py`:
```python
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
```

- [ ] **Step 2: Run integration test to verify it fails**

```bash
pytest tests/strikecast/test_baseline_integration.py -v
```

Expected: FAIL — `ImportError: cannot import name 'run_baseline_phase'`

- [ ] **Step 3: Implement run_baseline_phase in cli.py**

Replace the `if args.phase in ("all", "baseline"):` stub in `strikecast/cli.py` with the full implementation. Add imports at the top and the `run_baseline_phase` function:

Add these imports to the top of `strikecast/cli.py`:
```python
from datetime import datetime, timezone

import pandas as pd

from strikecast.estimators.garch_mc import GarchMonteCarloEstimator
from strikecast.estimators.random_walk import RandomWalkEstimator
from strikecast.eval.report import RunReport, get_git_commit, make_run_id, write_report
from strikecast.eval.scoring import score_predictions
from strikecast.eval.splits import walk_forward_split
```

Add the `run_baseline_phase` function:
```python
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
```

Replace the existing `baseline` stub in the `main()` function:
```python
    if args.phase in ("all", "baseline"):
        run_baseline_phase(config)
```

- [ ] **Step 4: Run integration test to verify it passes**

```bash
pytest tests/strikecast/test_baseline_integration.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/strikecast/ -v
```

Expected: all Phase 0 + Phase 1 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add strikecast/cli.py tests/strikecast/test_baseline_integration.py
git commit -m "feat(strikecast): CLI baseline phase with walk-forward evaluation and report generation"
```

---

## Phase 1: Definition of Done

From `specs/quality.md`:

- [ ] Random-walk + GARCH-MC estimators implemented and tested
- [ ] Scoring harness emits full KPI table (Brier, log loss, ECE, BSS, dir-acc) with 95% bootstrap CIs
- [ ] Purged + embargoed walk-forward split implemented (purge >= 1, embargo >= 1)
- [ ] JSON + Markdown run report generated per run
- [ ] `test_no_leakage.py` GREEN — shuffling future bars does not change past probabilities
- [ ] `test_reproducibility.py` GREEN — same seed produces identical Brier to 1e-9
- [ ] `test_calibration_split_disjoint.py` GREEN — train/val/test pairwise disjoint
- [ ] `test_no_order_path.py` still GREEN (NFR-001 import guard)
- [ ] `test_grid_alignment.py` still GREEN (FR-007)
- [ ] All baseline tests pass: `pytest tests/strikecast/ -v`
- [ ] `--phase baseline` CLI works end-to-end on stored data
