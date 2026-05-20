"""Tests for the data pipeline: shape, normalization, label correctness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamsense.data.loso import fixed_split, loso_folds  # noqa: E402
from streamsense.data.synthetic import synthesize_subject  # noqa: E402
from streamsense.data.windowing import (  # noqa: E402
    apply_norm,
    fit_norm_stats,
    make_windows,
)


ACTIVITY_IDS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
WINDOW = 512
STEP = 256


@pytest.fixture(scope="module")
def synth_recording():
    return synthesize_subject(1, ACTIVITY_IDS, seconds_per_activity=20.0, seed=0)


def test_synthetic_recording_shapes(synth_recording):
    rec = synth_recording
    assert rec.hand.shape[1] == 9
    assert rec.chest.shape[1] == 9
    assert rec.ankle.shape[1] == 9
    assert rec.hr.shape[1] == 1
    assert rec.hand.shape[0] == rec.chest.shape[0] == rec.ankle.shape[0] == rec.hr.shape[0]
    assert rec.activity.shape[0] == rec.hand.shape[0]


def test_make_windows_shapes_and_labels(synth_recording):
    rec = synth_recording
    hand, chest, ankle, hr, y, s = make_windows(rec, ACTIVITY_IDS, WINDOW, STEP)
    assert hand.shape[1:] == (9, WINDOW)
    assert chest.shape[1:] == (9, WINDOW)
    assert ankle.shape[1:] == (9, WINDOW)
    assert hr.shape[1:] == (1, WINDOW)
    assert y.shape[0] == hand.shape[0] == s.shape[0]
    assert y.min() >= 0 and y.max() < len(ACTIVITY_IDS)
    # Synthetic has 12 activities, each 20s at 100Hz = 2000 samples.
    # Each gets ~7 windows (window=512, step=256, count = 1+(2000-512)//256 = 6).
    # So total should be ~72 windows across all activities.
    assert y.shape[0] >= 6 * len(ACTIVITY_IDS) - 3
    assert s[0] == rec.subject_id


def test_make_windows_drops_mixed_activity():
    # Construct a recording where the first half is activity A and second is B,
    # straddled by a window: that window must be dropped.
    from streamsense.data.windowing import SubjectRecording

    n = 2 * WINDOW
    rec = SubjectRecording(
        subject_id=42,
        hand=np.zeros((n, 9), dtype=np.float32),
        chest=np.zeros((n, 9), dtype=np.float32),
        ankle=np.zeros((n, 9), dtype=np.float32),
        hr=np.zeros((n, 1), dtype=np.float32),
        activity=np.concatenate([np.full(WINDOW, 1), np.full(WINDOW, 2)]).astype(np.int64),
        timestamps=np.arange(n, dtype=np.float64) / 100.0,
    )
    _, _, _, _, y, _ = make_windows(rec, [1, 2], WINDOW, STEP)
    # Windows starting at 0 and at WINDOW are pure; everything in between mixes.
    assert (y == 0).sum() == 1
    assert (y == 1).sum() == 1
    assert y.shape[0] == 2


def test_fit_apply_norm_centers_to_unit_variance(synth_recording):
    rec = synth_recording
    hand, chest, ankle, hr, _, _ = make_windows(rec, ACTIVITY_IDS, WINDOW, STEP)
    stats = fit_norm_stats(hand, chest, ankle, hr)
    nh, nc, na, nhr = apply_norm(hand, chest, ankle, hr, stats)
    for x in (nh, nc, na, nhr):
        per_channel_mean = x.mean(axis=(0, 2))
        per_channel_std = x.std(axis=(0, 2))
        assert np.allclose(per_channel_mean, 0.0, atol=1e-3)
        assert np.allclose(per_channel_std, 1.0, atol=1e-3)


def test_norm_stats_round_trip(tmp_path, synth_recording):
    rec = synth_recording
    h, c, a, hr, _, _ = make_windows(rec, ACTIVITY_IDS, WINDOW, STEP)
    stats = fit_norm_stats(h, c, a, hr)
    stats.save(tmp_path / "norm.npz")
    from streamsense.data.windowing import NormStats

    loaded = NormStats.load(tmp_path / "norm.npz")
    np.testing.assert_allclose(loaded.hand_mean, stats.hand_mean)
    np.testing.assert_allclose(loaded.hr_std, stats.hr_std)


def test_loso_folds_have_disjoint_test_subjects():
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    folds = loso_folds(subjects)
    assert len(folds) == 9
    test_subjects = {f.test[0] for f in folds}
    assert test_subjects == set(subjects)
    for f in folds:
        assert set(f.train).isdisjoint(set(f.val))
        assert set(f.train).isdisjoint(set(f.test))
        assert set(f.val).isdisjoint(set(f.test))


def test_fixed_split_passthrough():
    spec = fixed_split([1, 2, 3], [4], [5])
    assert spec.train == [1, 2, 3] and spec.val == [4] and spec.test == [5]
