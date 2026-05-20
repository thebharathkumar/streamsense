"""Leave-one-subject-out and fixed-split construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class FoldSpec:
    """Which subject IDs go to train, val, test for a single fold."""

    fold_id: int
    train: list[int]
    val: list[int]
    test: list[int]


def loso_folds(subjects: Sequence[int], val_subject: int | None = None) -> list[FoldSpec]:
    """Return one LOSO fold per held-out test subject.

    val_subject, if given, is removed from train and used as val across all
    folds (except when it coincides with the test subject, in which case the
    next subject in order is used).
    """
    folds: list[FoldSpec] = []
    sorted_subjects = sorted(set(subjects))
    for i, test_s in enumerate(sorted_subjects):
        remaining = [s for s in sorted_subjects if s != test_s]
        if val_subject is not None and val_subject in remaining:
            val = val_subject
        else:
            val = remaining[-1]
        train = [s for s in remaining if s != val]
        folds.append(FoldSpec(fold_id=i + 1, train=train, val=[val], test=[test_s]))
    return folds


def fixed_split(
    train: Sequence[int], val: Sequence[int], test: Sequence[int]
) -> FoldSpec:
    """Single FoldSpec for the fast iteration path."""
    return FoldSpec(fold_id=0, train=list(train), val=list(val), test=list(test))
