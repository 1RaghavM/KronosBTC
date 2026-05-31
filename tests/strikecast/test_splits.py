import numpy as np
import pytest

from strikecast.constants import WINDOW_SECONDS


def _make_timestamps(n: int = 100) -> list[int]:
    base = 1_700_000_000 - (1_700_000_000 % WINDOW_SECONDS)
    return [base + i * WINDOW_SECONDS for i in range(n)]


class TestWalkForwardSplit:
    def test_basic_split_sizes(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        ts = _make_timestamps(100)
        split = walk_forward_split(
            timestamps=ts,
            train_frac=0.6,
            val_frac=0.2,
            purge_windows=1,
            embargo_windows=1,
        )

        assert len(split.train) == 60
        assert len(split.val) > 0
        assert len(split.test) > 0
        assert len(split.train) + len(split.val) + len(split.test) <= 100

    def test_sets_are_disjoint(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        ts = _make_timestamps(200)
        split = walk_forward_split(
            timestamps=ts,
            train_frac=0.6,
            val_frac=0.2,
            purge_windows=1,
            embargo_windows=1,
        )

        train_set = set(split.train)
        val_set = set(split.val)
        test_set = set(split.test)

        assert train_set & val_set == set()
        assert val_set & test_set == set()
        assert train_set & test_set == set()

    def test_temporal_ordering(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        ts = _make_timestamps(200)
        split = walk_forward_split(
            timestamps=ts,
            train_frac=0.6,
            val_frac=0.2,
            purge_windows=1,
            embargo_windows=1,
        )

        assert max(split.train) < min(split.val)
        assert max(split.val) < min(split.test)

    def test_purge_gap_exists(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        ts = _make_timestamps(200)
        split = walk_forward_split(
            timestamps=ts,
            train_frac=0.6,
            val_frac=0.2,
            purge_windows=2,
            embargo_windows=1,
        )

        gap_train_val = min(split.val) - max(split.train)
        min_gap = (2 + 1) * WINDOW_SECONDS

        assert gap_train_val >= min_gap, (
            f"Purge+embargo gap between train and val is {gap_train_val}s, " f"need >= {min_gap}s"
        )

    def test_embargo_gap_exists(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        ts = _make_timestamps(200)
        split = walk_forward_split(
            timestamps=ts,
            train_frac=0.6,
            val_frac=0.2,
            purge_windows=1,
            embargo_windows=3,
        )

        gap_val_test = min(split.test) - max(split.val)
        min_gap = (1 + 3) * WINDOW_SECONDS

        assert gap_val_test >= min_gap

    def test_small_dataset_raises(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        ts = _make_timestamps(5)

        with pytest.raises(ValueError, match="timestamps"):
            walk_forward_split(
                timestamps=ts,
                train_frac=0.6,
                val_frac=0.2,
                purge_windows=1,
                embargo_windows=1,
            )

    def test_all_timestamps_grid_aligned(self) -> None:
        from strikecast.eval.splits import walk_forward_split

        ts = _make_timestamps(200)
        split = walk_forward_split(
            timestamps=ts,
            train_frac=0.6,
            val_frac=0.2,
            purge_windows=1,
            embargo_windows=1,
        )

        for t in split.train + split.val + split.test:
            assert t % WINDOW_SECONDS == 0

    def test_dataclass_fields(self) -> None:
        from strikecast.eval.splits import WalkForwardSplit, walk_forward_split

        ts = _make_timestamps(100)
        split = walk_forward_split(
            timestamps=ts,
            train_frac=0.6,
            val_frac=0.2,
            purge_windows=1,
            embargo_windows=1,
        )

        assert isinstance(split, WalkForwardSplit)
        assert hasattr(split, "train")
        assert hasattr(split, "val")
        assert hasattr(split, "test")
