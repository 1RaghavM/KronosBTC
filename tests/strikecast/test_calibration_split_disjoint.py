"""FR-020: Calibration split must be disjoint from train and test."""

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

        assert train_set & val_set == set(), f"Train and val overlap: {train_set & val_set}"
        assert val_set & test_set == set(), f"Val and test overlap: {val_set & test_set}"
        assert train_set & test_set == set(), f"Train and test overlap: {train_set & test_set}"

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

        assert val_set.isdisjoint(test_set), "Calibration (val) and test sets must be disjoint"

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

        assert train_set.isdisjoint(val_set), "Training and calibration (val) sets must be disjoint"
