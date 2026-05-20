"""PyTorch dataset over memory-mapped PAMAP2 windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ProcessedSplit:
    """Filenames produced by prepare_data for one split (train/val/test)."""

    hand: Path
    chest: Path
    ankle: Path
    hr: Path
    labels: Path
    subjects: Path


def split_paths(processed_dir: Path, split: str) -> ProcessedSplit:
    """Return the canonical filenames for a split."""
    d = processed_dir
    return ProcessedSplit(
        hand=d / f"{split}_hand.npy",
        chest=d / f"{split}_chest.npy",
        ankle=d / f"{split}_ankle.npy",
        hr=d / f"{split}_hr.npy",
        labels=d / f"{split}_labels.npy",
        subjects=d / f"{split}_subjects.npy",
    )


def save_split(
    processed_dir: Path,
    split: str,
    hand: np.ndarray,
    chest: np.ndarray,
    ankle: np.ndarray,
    hr: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
) -> None:
    """Persist a split's arrays as .npy for memory-mapped loading."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    p = split_paths(processed_dir, split)
    np.save(p.hand, hand)
    np.save(p.chest, chest)
    np.save(p.ankle, ankle)
    np.save(p.hr, hr)
    np.save(p.labels, labels)
    np.save(p.subjects, subjects)


class WindowDataset:
    """Dataset returning a dict of per-modality torch tensors.

    Lazy import of torch so the data prep script (which only needs numpy)
    does not pay the torch import cost.
    """

    def __init__(self, processed_dir: Path, split: str, mmap: bool = True) -> None:
        import torch  # local import

        self._torch = torch
        p = split_paths(processed_dir, split)
        mode = "r" if mmap else None
        self.hand = np.load(p.hand, mmap_mode=mode)
        self.chest = np.load(p.chest, mmap_mode=mode)
        self.ankle = np.load(p.ankle, mmap_mode=mode)
        self.hr = np.load(p.hr, mmap_mode=mode)
        self.labels = np.load(p.labels, mmap_mode=mode)
        self.subjects = np.load(p.subjects, mmap_mode=mode)

        assert (
            self.hand.shape[0]
            == self.chest.shape[0]
            == self.ankle.shape[0]
            == self.hr.shape[0]
            == self.labels.shape[0]
            == self.subjects.shape[0]
        ), "split arrays have mismatched first dim"

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        t = self._torch
        return {
            "hand": t.from_numpy(np.ascontiguousarray(self.hand[idx])),
            "chest": t.from_numpy(np.ascontiguousarray(self.chest[idx])),
            "ankle": t.from_numpy(np.ascontiguousarray(self.ankle[idx])),
            "hr": t.from_numpy(np.ascontiguousarray(self.hr[idx])),
            "label": int(self.labels[idx]),
        }

    @property
    def num_classes(self) -> int:
        return int(self.labels.max()) + 1


def class_counts(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Count occurrences per class, returning an array of length num_classes."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        counts[c] = int((labels == c).sum())
    return counts


def inverse_freq_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Inverse-frequency class weights, normalized to mean=1."""
    counts = class_counts(labels, num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (num_classes * counts)
    w = (w / w.mean()).astype(np.float32)
    return w
