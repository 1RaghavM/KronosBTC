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
            raise ValueError(f"lookback has {len(lookback_df)} rows, need >= {self._min_lookback}")

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
