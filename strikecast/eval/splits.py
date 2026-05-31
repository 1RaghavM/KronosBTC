from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardSplit:
    """Result of a purged + embargoed walk-forward split.

    All lists contain int timestamps (Unix epoch seconds).
    train, val, and test are pairwise disjoint and temporally ordered.
    """

    train: list[int]
    val: list[int]
    test: list[int]


def walk_forward_split(
    timestamps: list[int],
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    purge_windows: int = 1,
    embargo_windows: int = 1,
) -> WalkForwardSplit:
    """Split timestamps into train/val/test with purge + embargo gaps (FR-040)."""
    ts = sorted(timestamps)
    n = len(ts)
    gap = purge_windows + embargo_windows

    train_end = int(n * train_frac)
    val_start = train_end + gap
    val_end = val_start + int(n * val_frac)
    test_start = val_end + gap

    if train_end < 1 or val_start >= n or test_start >= n:
        raise ValueError(
            f"Not enough timestamps ({n}) for the requested split "
            f"(train_frac={train_frac}, val_frac={val_frac}, "
            f"purge={purge_windows}, embargo={embargo_windows}). "
            f"Need at least {train_end + 2 * gap + 2} timestamps."
        )

    train = ts[:train_end]
    val = ts[val_start:val_end]
    test = ts[test_start:]

    if not val or not test:
        raise ValueError(
            f"Not enough timestamps ({n}) to produce non-empty val and test sets. "
            f"Computed val_start={val_start}, val_end={val_end}, test_start={test_start}."
        )

    return WalkForwardSplit(train=train, val=val, test=test)
