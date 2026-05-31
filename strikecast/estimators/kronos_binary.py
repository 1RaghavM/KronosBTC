from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from strikecast.constants import (
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    WINDOW_SECONDS,
)
from strikecast.estimators.base import Calibrator, PathSampler, ProbResult

logger = logging.getLogger(__name__)


class KronosBinaryEstimator:
    """Kronos Monte Carlo binary probability estimator (FR-010..015).

    Treats Kronos as a Monte Carlo path simulator: draws ``sample_count``
    independent forecast paths for the next window via an injected
    :class:`~strikecast.estimators.base.PathSampler`, and reports the raw
    probability ``P(close > strike)`` as the fraction of sampled closes that
    exceed the strike. A bootstrap confidence interval is computed over the
    sampled closes (NFR-007: no point estimate without a CI).

    An optional calibrator (FR-020/021) maps the raw probability to a
    calibrated one; when ``calibrator is None`` (zero-shot, Phase 2) the
    calibrated ``p`` equals ``p_raw``.

    The estimator reads only the lookback window, so permuting any future bar
    leaves the probability bit-identical (NFR-002, leakage-safe by
    construction).
    """

    def __init__(
        self,
        path_sampler: PathSampler,
        calibrator: Calibrator | None = None,
        sample_count: int = DEFAULT_SAMPLE_COUNT,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        seed: int = 42,
        n_bootstrap: int = 1000,
        min_lookback: int = 50,
        max_context: int = 512,
    ) -> None:
        if sample_count <= 0:
            raise ValueError(f"sample_count must be positive, got {sample_count}")
        self._sampler = path_sampler
        self._calibrator = calibrator
        self._sample_count = sample_count
        self._temperature = temperature
        self._top_p = top_p
        self._seed = seed
        self._n_bootstrap = n_bootstrap
        self._min_lookback = min_lookback
        self._max_context = max_context

    def _calibrate(self, p_raw: float) -> float:
        if self._calibrator is None:
            return p_raw
        calibrated = float(np.asarray(self._calibrator.apply(p_raw)).reshape(-1)[0])
        return float(np.clip(calibrated, 0.0, 1.0))

    def estimate(self, lookback_df: pd.DataFrame, strike: float) -> ProbResult:
        """Estimate P(next-window close > strike).

        Args:
            lookback_df: Historical OHLCV candles ending at the window before
                the target. Must contain ``window_open_ts`` (int Unix seconds,
                300s-grid-aligned) and ``open, high, low, close``.
            strike: Strike price in USD.

        Returns:
            A :class:`ProbResult` with calibrated ``p`` (== ``p_raw`` when no
            calibrator is set), the raw probability, a 95% bootstrap CI, and
            the Monte Carlo sample count.

        Raises:
            ValueError: If the lookback has fewer than ``min_lookback`` rows.
        """
        if len(lookback_df) < self._min_lookback:
            raise ValueError(
                f"lookback has {len(lookback_df)} rows, need >= {self._min_lookback}"
            )

        if len(lookback_df) > self._max_context:
            logger.warning(
                "lookback (%d rows) exceeds max_context (%d); clamping to the most "
                "recent %d windows",
                len(lookback_df),
                self._max_context,
                self._max_context,
            )
            lookback_df = lookback_df.iloc[-self._max_context :]

        ts = lookback_df["window_open_ts"].to_numpy(dtype=np.int64)
        # Kronos' calc_time_stamps uses the ``.dt`` accessor, so timestamps must
        # be pandas Series (not a DatetimeIndex).
        x_timestamp = pd.Series(pd.to_datetime(ts, unit="s", utc=True))
        y_timestamp = pd.Series(
            pd.to_datetime([int(ts[-1]) + WINDOW_SECONDS], unit="s", utc=True)
        )

        closes = np.asarray(
            self._sampler.sample_closes(
                lookback_df,
                x_timestamp,
                y_timestamp,
                self._sample_count,
                self._temperature,
                self._top_p,
            ),
            dtype=float,
        ).reshape(-1)

        n = len(closes)
        if n == 0:
            raise RuntimeError("path sampler returned no samples")

        above = closes > strike
        p_raw = float(np.mean(above))

        boot_rng = np.random.RandomState(self._seed)
        bootstrap_ps = np.empty(self._n_bootstrap)
        for i in range(self._n_bootstrap):
            idx = boot_rng.randint(0, n, size=n)
            bootstrap_ps[i] = float(np.mean(above[idx]))

        ci_low_raw = float(np.percentile(bootstrap_ps, 2.5))
        ci_high_raw = float(np.percentile(bootstrap_ps, 97.5))

        p = self._calibrate(p_raw)
        ci_low = self._calibrate(ci_low_raw)
        ci_high = self._calibrate(ci_high_raw)
        if ci_low > ci_high:
            ci_low, ci_high = ci_high, ci_low

        return ProbResult(
            p=p,
            p_raw=p_raw,
            ci_low=ci_low,
            ci_high=ci_high,
            n_samples=n,
        )
