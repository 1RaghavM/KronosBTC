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
        expected = -(math.log(0.7) + math.log(0.7) + math.log(0.6) + math.log(0.2)) / 4.0

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

        results = score_predictions(predictions, labels, n_bootstrap=100, seed=42)

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

        results = score_predictions(predictions, labels, n_bootstrap=100, seed=42)

        estimators = {r.estimator for r in results}
        assert "randomwalk" in estimators
        assert "garch_mc" in estimators
