"""PAMAP2 parsing, sliding windows, and per-channel normalization.

Column layout for the protocol .dat files (1-indexed in the PAMAP2 docs,
0-indexed here):

    0       timestamp (s)
    1       activityID
    2       heartRate (bpm), 9 Hz, NaN at 100 Hz rows
    3-19    IMU hand   (17 cols: temp, 3 accel-16g, 3 accel-6g, 3 gyro,
                        3 mag, 4 orientation)
    20-36   IMU chest  (same layout)
    37-53   IMU ankle  (same layout)

We use the 3 accel-16g, 3 gyro, 3 mag channels per IMU (9 channels per
location, 27 IMU channels total), and the heart rate channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

IMU_BLOCK_SIZE = 17
IMU_ACC16G_OFFSETS = (1, 2, 3)
IMU_GYRO_OFFSETS = (7, 8, 9)
IMU_MAG_OFFSETS = (10, 11, 12)

IMU_BLOCK_STARTS = {
    "hand": 3,
    "chest": 20,
    "ankle": 37,
}
HR_COL = 2
ACTIVITY_COL = 1
TIMESTAMP_COL = 0

IMU_CHANNELS_PER_LOCATION = 9


def imu_column_indices(location: str) -> list[int]:
    """Return the 9 column indices for a given IMU location."""
    start = IMU_BLOCK_STARTS[location]
    offsets = IMU_ACC16G_OFFSETS + IMU_GYRO_OFFSETS + IMU_MAG_OFFSETS
    return [start + o for o in offsets]


@dataclass
class SubjectRecording:
    """One subject's parsed and resampled signals."""

    subject_id: int
    hand: np.ndarray
    chest: np.ndarray
    ankle: np.ndarray
    hr: np.ndarray
    activity: np.ndarray
    timestamps: np.ndarray


def read_subject_dat(path: Path) -> pd.DataFrame:
    """Read one subject{n}.dat into a DataFrame."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        na_values=["NaN"],
        engine="c",
    )
    return df


def parse_recording(df: pd.DataFrame, subject_id: int) -> SubjectRecording:
    """Extract per-modality arrays from one subject DataFrame.

    Heart rate is forward-filled to the 100 Hz IMU rate. Rows whose activity
    is the PAMAP2 'transient' label (0) are kept here; filtering happens
    downstream so we can also use them for HR ffill warmup.
    """
    timestamps = df.iloc[:, TIMESTAMP_COL].to_numpy(dtype=np.float64)
    activity = df.iloc[:, ACTIVITY_COL].to_numpy(dtype=np.int64)

    hr_series = df.iloc[:, HR_COL].ffill().bfill()
    hr = hr_series.to_numpy(dtype=np.float32).reshape(-1, 1)

    def take(loc: str) -> np.ndarray:
        cols = imu_column_indices(loc)
        block = df.iloc[:, cols].to_numpy(dtype=np.float32)
        block = pd.DataFrame(block).ffill().bfill().fillna(0.0).to_numpy(
            dtype=np.float32
        )
        return block

    hand = take("hand")
    chest = take("chest")
    ankle = take("ankle")

    return SubjectRecording(
        subject_id=subject_id,
        hand=hand,
        chest=chest,
        ankle=ankle,
        hr=hr,
        activity=activity,
        timestamps=timestamps,
    )


def make_windows(
    rec: SubjectRecording,
    activity_ids: Sequence[int],
    window_samples: int,
    step_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows for one subject.

    Returns (hand, chest, ankle, hr, labels, subjects).
    - hand/chest/ankle: (N, 9, window_samples)
    - hr:               (N, 1, window_samples)
    - labels:           (N,) mapped to 0..len(activity_ids)-1
    - subjects:         (N,) subject id

    A window is kept only if all samples share the same activity AND that
    activity is in the allowed set.
    """
    id_to_label = {a: i for i, a in enumerate(activity_ids)}
    allowed = set(activity_ids)

    T = rec.activity.shape[0]
    if T < window_samples:
        empty_imu = np.empty(
            (0, IMU_CHANNELS_PER_LOCATION, window_samples), dtype=np.float32
        )
        empty_hr = np.empty((0, 1, window_samples), dtype=np.float32)
        return (
            empty_imu, empty_imu, empty_imu, empty_hr,
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    starts = np.arange(0, T - window_samples + 1, step_samples)

    hand_w: list[np.ndarray] = []
    chest_w: list[np.ndarray] = []
    ankle_w: list[np.ndarray] = []
    hr_w: list[np.ndarray] = []
    labels: list[int] = []

    act = rec.activity
    for s in starts:
        e = s + window_samples
        seg = act[s:e]
        first = int(seg[0])
        if first not in allowed:
            continue
        if not (seg == first).all():
            continue
        hand_w.append(rec.hand[s:e].T)
        chest_w.append(rec.chest[s:e].T)
        ankle_w.append(rec.ankle[s:e].T)
        hr_w.append(rec.hr[s:e].T)
        labels.append(id_to_label[first])

    if not labels:
        empty_imu = np.empty(
            (0, IMU_CHANNELS_PER_LOCATION, window_samples), dtype=np.float32
        )
        empty_hr = np.empty((0, 1, window_samples), dtype=np.float32)
        return (
            empty_imu, empty_imu, empty_imu, empty_hr,
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    hand_arr = np.stack(hand_w).astype(np.float32)
    chest_arr = np.stack(chest_w).astype(np.float32)
    ankle_arr = np.stack(ankle_w).astype(np.float32)
    hr_arr = np.stack(hr_w).astype(np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64)
    subjects_arr = np.full_like(labels_arr, rec.subject_id)

    return hand_arr, chest_arr, ankle_arr, hr_arr, labels_arr, subjects_arr


@dataclass
class NormStats:
    """Per-channel mean and std for each modality, fit on train only."""

    hand_mean: np.ndarray
    hand_std: np.ndarray
    chest_mean: np.ndarray
    chest_std: np.ndarray
    ankle_mean: np.ndarray
    ankle_std: np.ndarray
    hr_mean: np.ndarray
    hr_std: np.ndarray

    def save(self, path: Path) -> None:
        np.savez(
            path,
            hand_mean=self.hand_mean, hand_std=self.hand_std,
            chest_mean=self.chest_mean, chest_std=self.chest_std,
            ankle_mean=self.ankle_mean, ankle_std=self.ankle_std,
            hr_mean=self.hr_mean, hr_std=self.hr_std,
        )

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        d = np.load(path)
        return cls(
            hand_mean=d["hand_mean"], hand_std=d["hand_std"],
            chest_mean=d["chest_mean"], chest_std=d["chest_std"],
            ankle_mean=d["ankle_mean"], ankle_std=d["ankle_std"],
            hr_mean=d["hr_mean"], hr_std=d["hr_std"],
        )


def _per_channel_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std over (N, C, T) -> (C,)."""
    flat = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
    mean = flat.mean(axis=1).astype(np.float32)
    std = flat.std(axis=1).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def fit_norm_stats(
    hand: np.ndarray, chest: np.ndarray, ankle: np.ndarray, hr: np.ndarray
) -> NormStats:
    """Fit per-channel z-score statistics on training windows."""
    hm, hs = _per_channel_stats(hand)
    cm, cs = _per_channel_stats(chest)
    am, as_ = _per_channel_stats(ankle)
    hrm, hrs = _per_channel_stats(hr)
    return NormStats(hm, hs, cm, cs, am, as_, hrm, hrs)


def apply_norm(
    hand: np.ndarray, chest: np.ndarray, ankle: np.ndarray, hr: np.ndarray,
    stats: NormStats,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply z-score normalization (returns new arrays)."""
    def _norm(x: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
        return ((x - m[None, :, None]) / s[None, :, None]).astype(np.float32)

    return (
        _norm(hand, stats.hand_mean, stats.hand_std),
        _norm(chest, stats.chest_mean, stats.chest_std),
        _norm(ankle, stats.ankle_mean, stats.ankle_std),
        _norm(hr, stats.hr_mean, stats.hr_std),
    )


def window_count_for(num_samples: int, window_samples: int, step_samples: int) -> int:
    """How many windows fit in a recording of length num_samples."""
    if num_samples < window_samples:
        return 0
    return 1 + (num_samples - window_samples) // step_samples


def iter_dat_files(protocol_dir: Path) -> Iterable[tuple[int, Path]]:
    """Yield (subject_id, path) for each subject*.dat in protocol_dir."""
    for path in sorted(protocol_dir.glob("subject*.dat")):
        stem = path.stem
        try:
            sid = int(stem.replace("subject10", ""))
        except ValueError:
            sid = int("".join(c for c in stem if c.isdigit())) % 100
        yield sid, path
