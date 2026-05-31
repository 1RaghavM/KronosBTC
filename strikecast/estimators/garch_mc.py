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
        """Monte Carlo simulation given a known sigma."""
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

        return ProbResult(p=p, p_raw=p, ci_low=ci_low, ci_high=ci_high, n_samples=self._n_samples)

    def estimate(self, lookback_df: pd.DataFrame, strike: float) -> ProbResult:
        if len(lookback_df) < self._min_lookback:
            raise ValueError(f"lookback has {len(lookback_df)} rows, need >= {self._min_lookback}")

        closes = lookback_df["close"].values.astype(float)
        log_returns = np.diff(np.log(closes))
        current_price = float(closes[-1])

        sigma = self._fit_sigma(log_returns)
        return self._simulate_probability(current_price, sigma, strike)
