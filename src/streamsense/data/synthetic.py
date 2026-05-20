"""Synthetic PAMAP2-shaped recordings for offline development and CI.

Real PAMAP2 download is the default. This module exists so the rest of the
pipeline (windowing, normalization, dataset, model, training, ONNX export,
benchmarks, server) is exercisable in environments without internet access
or the bandwidth budget for the 1.4 GB archive.

Each activity gets its own class-conditional generator with distinct
frequency content and amplitude per IMU location, so the model can actually
learn to separate them rather than picking up noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .windowing import SubjectRecording

SAMPLING_RATE_HZ = 100


@dataclass
class ActivityProfile:
    """Frequency and amplitude knobs per activity per IMU location."""

    name: str
    freq_hand: float
    freq_chest: float
    freq_ankle: float
    amp_hand: float
    amp_chest: float
    amp_ankle: float
    hr_mean: float


def _profile_for(activity_id: int, idx: int) -> ActivityProfile:
    """Deterministic activity-specific signal profile."""
    rng = np.random.default_rng(seed=1000 + activity_id)
    base = 0.2 + 1.8 * (idx / 12.0)
    return ActivityProfile(
        name=f"act_{activity_id}",
        freq_hand=float(rng.uniform(0.5, 4.5) * base),
        freq_chest=float(rng.uniform(0.3, 3.0) * base),
        freq_ankle=float(rng.uniform(0.5, 5.0) * base),
        amp_hand=float(0.6 + 0.4 * rng.standard_normal()),
        amp_chest=float(0.4 + 0.3 * rng.standard_normal()),
        amp_ankle=float(0.8 + 0.5 * rng.standard_normal()),
        hr_mean=float(70 + 12 * idx + 5 * rng.standard_normal()),
    )


def _synth_imu(
    n_samples: int, freq: float, amp: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate one IMU's 9 channels (accel, gyro, mag) as (T, 9)."""
    t = np.arange(n_samples) / SAMPLING_RATE_HZ
    out = np.zeros((n_samples, 9), dtype=np.float32)
    # Accel: dominant fundamental + harmonics.
    for c in range(3):
        phase = rng.uniform(0, 2 * np.pi)
        out[:, c] = (
            amp * np.sin(2 * np.pi * freq * t + phase)
            + 0.3 * amp * np.sin(2 * np.pi * 2 * freq * t + phase)
            + 0.05 * rng.standard_normal(n_samples)
        )
    # Gyro: lower amplitude, frequency-shifted.
    for c in range(3, 6):
        phase = rng.uniform(0, 2 * np.pi)
        out[:, c] = (
            0.5 * amp * np.sin(2 * np.pi * (freq * 0.7) * t + phase)
            + 0.05 * rng.standard_normal(n_samples)
        )
    # Mag: slow drift + small oscillation.
    for c in range(6, 9):
        drift = rng.standard_normal()
        out[:, c] = (
            drift + 0.2 * amp * np.sin(2 * np.pi * (freq * 0.2) * t)
            + 0.02 * rng.standard_normal(n_samples)
        )
    return out


def synthesize_subject(
    subject_id: int,
    activity_ids: Sequence[int],
    seconds_per_activity: float = 60.0,
    seed: int = 0,
) -> SubjectRecording:
    """Build one synthetic subject recording covering all activities sequentially."""
    rng = np.random.default_rng(seed + subject_id * 31)

    blocks_act: list[np.ndarray] = []
    blocks_ts: list[np.ndarray] = []
    blocks_hand: list[np.ndarray] = []
    blocks_chest: list[np.ndarray] = []
    blocks_ankle: list[np.ndarray] = []
    blocks_hr: list[np.ndarray] = []

    cursor = 0.0
    for idx, aid in enumerate(activity_ids):
        prof = _profile_for(aid, idx)
        # Per-subject jitter on amplitude and HR baseline.
        amp_h = prof.amp_hand * float(np.exp(0.15 * rng.standard_normal()))
        amp_c = prof.amp_chest * float(np.exp(0.15 * rng.standard_normal()))
        amp_a = prof.amp_ankle * float(np.exp(0.15 * rng.standard_normal()))
        hr_base = prof.hr_mean + 3.0 * rng.standard_normal()

        n = int(seconds_per_activity * SAMPLING_RATE_HZ)
        hand = _synth_imu(n, prof.freq_hand, amp_h, rng)
        chest = _synth_imu(n, prof.freq_chest, amp_c, rng)
        ankle = _synth_imu(n, prof.freq_ankle, amp_a, rng)

        # HR drifts slowly toward the activity baseline.
        hr = np.zeros((n, 1), dtype=np.float32)
        cur = hr_base + 2.0 * rng.standard_normal()
        for i in range(n):
            cur += 0.005 * (hr_base - cur) + 0.3 * rng.standard_normal()
            hr[i, 0] = cur

        ts = cursor + np.arange(n) / SAMPLING_RATE_HZ
        cursor += n / SAMPLING_RATE_HZ

        blocks_act.append(np.full(n, aid, dtype=np.int64))
        blocks_ts.append(ts.astype(np.float64))
        blocks_hand.append(hand)
        blocks_chest.append(chest)
        blocks_ankle.append(ankle)
        blocks_hr.append(hr)

    return SubjectRecording(
        subject_id=subject_id,
        hand=np.concatenate(blocks_hand, axis=0),
        chest=np.concatenate(blocks_chest, axis=0),
        ankle=np.concatenate(blocks_ankle, axis=0),
        hr=np.concatenate(blocks_hr, axis=0),
        activity=np.concatenate(blocks_act, axis=0),
        timestamps=np.concatenate(blocks_ts, axis=0),
    )
