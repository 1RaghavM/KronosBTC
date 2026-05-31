"""Post-hoc probability calibration maps (FR-020, FR-021, FR-023).

A calibrator maps *raw* (uncalibrated) Monte-Carlo probabilities in [0, 1] to
*calibrated* probabilities in [0, 1]. It is fit on a validation split that is
disjoint from both the training and test windows (FR-020), then applied to all
raw probabilities before scoring or paper PnL (FR-021).

Two implementations are provided:

* :class:`IsotonicCalibrator` (primary) -- non-parametric isotonic regression
  (PAV). Handles any monotonic distortion; can overfit on small validation
  splits.
* :class:`PlattCalibrator` (fallback) -- logistic / sigmoid scaling. More
  stable when the validation split is small.

Both are monotonic non-decreasing in the raw probability, which preserves
confidence-interval ordering when applied to ``ci_low``/``ci_high``.

All fitted maps can be persisted as a versioned artifact tied to the model
checkpoint and the data window that produced them (FR-023), via
:meth:`Calibrator.save` and the module-level :func:`load_calibrator`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal, cast

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from strikecast.constants import CALIBRATION_ARTIFACT_VERSION

CalibrationMethod = Literal["isotonic", "platt"]

_FLOAT_EPS = 1e-12


def _as_2d_feature(p_raw: float | np.ndarray) -> np.ndarray:
    """Reshape raw probabilities into the ``(n, 1)`` feature matrix sklearn wants."""
    arr = np.asarray(p_raw, dtype=float).reshape(-1, 1)
    return arr


class _BaseCalibrator:
    """Shared metadata + save/load plumbing for calibrators.

    Attributes:
        model_checkpoint: Identifier of the model checkpoint these raw
            probabilities came from (e.g. ``"NeoQuasar/Kronos-small"``). Stored
            in the artifact so a calibrator is never mixed with another model.
        data_window: ``(start, end)`` date strings of the validation window the
            map was fit on, for provenance (FR-023).
    """

    method: CalibrationMethod

    def __init__(
        self,
        *,
        model_checkpoint: str | None = None,
        data_window: tuple[str, str] | None = None,
    ) -> None:
        self.model_checkpoint = model_checkpoint
        self.data_window = data_window
        self._fitted = False

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "calibrator must be fit before apply(); call .fit(p_raw, outcomes) first"
            )

    def save(self, path: str | Path) -> None:
        """Persist the fitted map as a versioned pickle artifact (FR-023).

        Args:
            path: Destination file path. Parent directories are created.
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CALIBRATION_ARTIFACT_VERSION,
            "method": self.method,
            "model_checkpoint": self.model_checkpoint,
            "data_window": self.data_window,
            "calibrator": self,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh)


class IsotonicCalibrator(_BaseCalibrator):
    """Isotonic-regression calibration map (PAV); primary method (FR-020)."""

    method: CalibrationMethod = "isotonic"

    def __init__(
        self,
        *,
        model_checkpoint: str | None = None,
        data_window: tuple[str, str] | None = None,
    ) -> None:
        super().__init__(model_checkpoint=model_checkpoint, data_window=data_window)
        self._model = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )

    def fit(self, p_raw: np.ndarray, outcomes: np.ndarray) -> IsotonicCalibrator:
        """Fit the isotonic map on a validation split.

        Args:
            p_raw: Raw probabilities in [0, 1] (1-D, length ``n``).
            outcomes: Binary outcomes (0.0/1.0) for the same windows.

        Returns:
            ``self`` (fitted), for chaining.
        """
        x = np.asarray(p_raw, dtype=float).reshape(-1)
        y = np.asarray(outcomes, dtype=float).reshape(-1)
        self._model.fit(x, y)
        self._fitted = True
        return self

    def apply(self, p_raw: float | np.ndarray) -> float | np.ndarray:
        """Map raw probability/probabilities to calibrated ones in [0, 1].

        Args:
            p_raw: Raw probability scalar or array in [0, 1].

        Returns:
            Calibrated probability as ``float`` for scalar input, else a 1-D
            ``np.ndarray``. Always clamped to [0, 1].
        """
        self._check_fitted()
        scalar = np.isscalar(p_raw) or np.asarray(p_raw).ndim == 0
        x = np.asarray(p_raw, dtype=float).reshape(-1)
        mapped: np.ndarray = np.clip(
            np.asarray(self._model.predict(x), dtype=float), 0.0, 1.0
        )
        if scalar:
            return float(mapped[0])
        return mapped


class PlattCalibrator(_BaseCalibrator):
    """Platt / sigmoid scaling calibration map; stable fallback (FR-020)."""

    method: CalibrationMethod = "platt"

    def __init__(
        self,
        *,
        model_checkpoint: str | None = None,
        data_window: tuple[str, str] | None = None,
    ) -> None:
        super().__init__(model_checkpoint=model_checkpoint, data_window=data_window)
        self._model = LogisticRegression(C=1e6, solver="lbfgs")
        self._degenerate_value: float | None = None

    def fit(self, p_raw: np.ndarray, outcomes: np.ndarray) -> PlattCalibrator:
        """Fit a 1-feature logistic regression (sigmoid) on a validation split.

        Args:
            p_raw: Raw probabilities in [0, 1] (1-D, length ``n``).
            outcomes: Binary outcomes (0.0/1.0) for the same windows.

        Returns:
            ``self`` (fitted), for chaining.
        """
        x = _as_2d_feature(p_raw)
        y = np.asarray(outcomes, dtype=float).reshape(-1)
        classes = np.unique(y)
        if classes.size < 2:
            # Single-class validation set: emit the constant base rate.
            self._degenerate_value = float(classes[0])
        else:
            self._degenerate_value = None
            self._model.fit(x, y)
            # Guard the monotonicity contract: if the fitted slope is negative
            # (pathological calibration data), flip to non-decreasing.
            if float(self._model.coef_.reshape(-1)[0]) < 0.0:
                self._model.coef_ = np.abs(self._model.coef_)
        self._fitted = True
        return self

    def apply(self, p_raw: float | np.ndarray) -> float | np.ndarray:
        """Map raw probability/probabilities to calibrated ones in [0, 1].

        Args:
            p_raw: Raw probability scalar or array in [0, 1].

        Returns:
            Calibrated probability as ``float`` for scalar input, else a 1-D
            ``np.ndarray``. Always clamped to [0, 1].
        """
        self._check_fitted()
        scalar = np.isscalar(p_raw) or np.asarray(p_raw).ndim == 0
        x = _as_2d_feature(p_raw)
        if self._degenerate_value is not None:
            raw_mapped = np.full(x.shape[0], self._degenerate_value, dtype=float)
        else:
            raw_mapped = np.asarray(self._model.predict_proba(x)[:, 1], dtype=float)
        mapped: np.ndarray = np.clip(raw_mapped, 0.0, 1.0)
        if scalar:
            return float(mapped[0])
        return mapped


def fit_calibrator(
    method: CalibrationMethod,
    p_raw: np.ndarray,
    outcomes: np.ndarray,
    *,
    model_checkpoint: str | None = None,
    data_window: tuple[str, str] | None = None,
) -> IsotonicCalibrator | PlattCalibrator:
    """Construct and fit a calibrator keyed off the configured method.

    Args:
        method: ``"isotonic"`` (primary) or ``"platt"`` (fallback); matches
            :class:`~strikecast.config.CalibrationConfig`.
        p_raw: Raw validation-split probabilities in [0, 1].
        outcomes: Binary outcomes (0.0/1.0) for the validation windows.
        model_checkpoint: Optional checkpoint id stored on the artifact (FR-023).
        data_window: Optional ``(start, end)`` validation window for provenance.

    Returns:
        A fitted calibrator.

    Raises:
        ValueError: If ``method`` is not a supported calibration method.
    """
    if method == "isotonic":
        return IsotonicCalibrator(
            model_checkpoint=model_checkpoint, data_window=data_window
        ).fit(p_raw, outcomes)
    if method == "platt":
        return PlattCalibrator(
            model_checkpoint=model_checkpoint, data_window=data_window
        ).fit(p_raw, outcomes)
    raise ValueError(
        f"unknown calibration method {method!r}; expected 'isotonic' or 'platt'"
    )


def load_calibrator(path: str | Path) -> IsotonicCalibrator | PlattCalibrator:
    """Load a fitted calibrator artifact written by :meth:`Calibrator.save`.

    Args:
        path: Path to a pickle artifact produced by ``save()``.

    Returns:
        The fitted calibrator instance.

    Raises:
        ValueError: If the artifact version is unsupported.
    """
    with Path(path).open("rb") as fh:
        payload = cast("dict[str, object]", pickle.load(fh))  # noqa: S301 - trusted local artifact
    version = payload.get("version")
    if version != CALIBRATION_ARTIFACT_VERSION:
        raise ValueError(
            f"unsupported calibration artifact version {version!r}; "
            f"expected {CALIBRATION_ARTIFACT_VERSION}"
        )
    return cast("IsotonicCalibrator | PlattCalibrator", payload["calibrator"])
