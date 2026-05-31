from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
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


@runtime_checkable
class PathSampler(Protocol):
    """Draws Monte Carlo terminal-close samples for the next window.

    Implementations turn a forecast model (e.g. Kronos) into a source of
    independent sampled close prices for the target window, so the binary
    probability P(close > strike) can be computed as a sample fraction.
    This protocol isolates the heavy model behind a thin, mockable seam
    (NFR-009: Kronos fork isolation).
    """

    def sample_closes(
        self,
        lookback_df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        sample_count: int,
        temperature: float,
        top_p: float,
    ) -> np.ndarray:
        """Return ``sample_count`` independent sampled close prices (USD).

        Args:
            lookback_df: Historical OHLCV candles up to (not including) the
                target window. Must contain ``open, high, low, close``.
            x_timestamp: Timestamps for each lookback row (UTC).
            y_timestamp: Timestamps for the forecast window(s) (UTC).
            sample_count: Number of independent Monte Carlo paths to draw.
            temperature: Sampling temperature ``T``.
            top_p: Nucleus-sampling threshold.

        Returns:
            1-D float array of length ``sample_count`` holding the sampled
            close price for the final forecast window of each path.
        """
        ...


@runtime_checkable
class Calibrator(Protocol):
    """Post-hoc probability calibration map (FR-020, FR-021).

    Maps raw (uncalibrated) probabilities to calibrated ones. Fitted on a
    validation split disjoint from train and test. Applying a fitted map
    must be monotonic so confidence-interval ordering is preserved.
    """

    def apply(self, p_raw: float | np.ndarray) -> float | np.ndarray:
        """Map raw probability/probabilities in [0, 1] to calibrated ones."""
        ...
