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


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
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
    """Score all estimator predictions against labels (FR-041)."""
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

            p_arr = subset["p"].values
            y_arr = subset["y"].values

            bs = brier_score(p_arr, y_arr)
            ll = log_loss(p_arr, y_arr)
            ece_val = expected_calibration_error(p_arr, y_arr)
            da = directional_accuracy(p_arr, y_arr)

            ci_bs = bootstrap_ci(p_arr, y_arr, brier_score, n_bootstrap, seed)
            ci_ll = bootstrap_ci(p_arr, y_arr, log_loss, n_bootstrap, seed)

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
