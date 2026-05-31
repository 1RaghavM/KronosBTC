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
