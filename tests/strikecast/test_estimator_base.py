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
