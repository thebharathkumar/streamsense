"""Prepare PAMAP2 windows for training.

Reads the raw .dat files (or synthesizes them with --synthetic), builds
sliding windows per subject, fits per-channel z-score statistics on the
training fold only, normalizes all splits, and persists arrays as .npy.

Default split is the fixed train/val/test split in configs/default.yaml.
Pass --fold N to use LOSO fold N instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamsense.data.dataset import save_split  # noqa: E402
from streamsense.data.loso import fixed_split, loso_folds  # noqa: E402
from streamsense.data.synthetic import synthesize_subject  # noqa: E402
from streamsense.data.windowing import (  # noqa: E402
    apply_norm,
    fit_norm_stats,
    iter_dat_files,
    make_windows,
    parse_recording,
    read_subject_dat,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--fold", type=int, default=None, help="LOSO fold id (1..9). Omit for fixed split.")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic PAMAP2-shaped data instead of disk.")
    p.add_argument("--synthetic-seconds", type=float, default=60.0, help="Seconds per activity per subject in synthetic mode.")
    p.add_argument("--limit-subjects", type=int, default=None, help="Cap on number of subjects to process.")
    return p.parse_args()


def load_subject_recordings(cfg, args):
    """Yield SubjectRecording objects according to source mode."""
    activity_ids = list(cfg.data.activity_ids)
    if args.synthetic:
        subjects = list(cfg.data.subjects)
        if args.limit_subjects is not None:
            subjects = subjects[: args.limit_subjects]
        for sid in subjects:
            yield synthesize_subject(
                sid, activity_ids=activity_ids,
                seconds_per_activity=args.synthetic_seconds, seed=0,
            )
        return

    protocol_dir = Path(cfg.paths.raw_dir) / "PAMAP2_Dataset" / "Protocol"
    if not protocol_dir.exists():
        raise FileNotFoundError(
            f"{protocol_dir} not found. Run scripts/download_pamap2.py first "
            "or pass --synthetic."
        )
    count = 0
    for sid, path in iter_dat_files(protocol_dir):
        if args.limit_subjects is not None and count >= args.limit_subjects:
            break
        df = read_subject_dat(path)
        yield parse_recording(df, sid)
        count += 1


def build_windows_for(recs, activity_ids, window_samples, step_samples):
    """Aggregate windows across a list of recordings."""
    hand_chunks: list[np.ndarray] = []
    chest_chunks: list[np.ndarray] = []
    ankle_chunks: list[np.ndarray] = []
    hr_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    subject_chunks: list[np.ndarray] = []

    for rec in recs:
        h, c, a, hr, y, s = make_windows(rec, activity_ids, window_samples, step_samples)
        if y.size == 0:
            continue
        hand_chunks.append(h)
        chest_chunks.append(c)
        ankle_chunks.append(a)
        hr_chunks.append(hr)
        label_chunks.append(y)
        subject_chunks.append(s)

    if not label_chunks:
        return None

    return {
        "hand": np.concatenate(hand_chunks, 0),
        "chest": np.concatenate(chest_chunks, 0),
        "ankle": np.concatenate(ankle_chunks, 0),
        "hr": np.concatenate(hr_chunks, 0),
        "labels": np.concatenate(label_chunks, 0),
        "subjects": np.concatenate(subject_chunks, 0),
    }


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    sr = int(cfg.data.sampling_rate_hz)
    window_samples = int(round(float(cfg.data.window_seconds) * sr))
    step_samples = int(round(window_samples * (1.0 - float(cfg.data.overlap))))
    activity_ids = list(cfg.data.activity_ids)

    print(f"[prepare] window_samples={window_samples} step_samples={step_samples}")
    print(f"[prepare] activities={activity_ids} ({len(activity_ids)} classes)")

    if args.fold is None:
        spec = fixed_split(
            train=cfg.data.fixed_split.train,
            val=cfg.data.fixed_split.val,
            test=cfg.data.fixed_split.test,
        )
        print(f"[prepare] fixed split: train={spec.train} val={spec.val} test={spec.test}")
    else:
        all_folds = loso_folds(cfg.data.subjects)
        if not 1 <= args.fold <= len(all_folds):
            raise ValueError(f"--fold must be in 1..{len(all_folds)}")
        spec = all_folds[args.fold - 1]
        print(
            f"[prepare] LOSO fold {spec.fold_id}: train={spec.train} val={spec.val} test={spec.test}"
        )

    print("[prepare] loading subjects...")
    all_recs = list(tqdm(load_subject_recordings(cfg, args)))
    by_subject = {rec.subject_id: rec for rec in all_recs}

    def gather(subject_ids):
        recs = [by_subject[s] for s in subject_ids if s in by_subject]
        return build_windows_for(recs, activity_ids, window_samples, step_samples)

    train = gather(spec.train)
    val = gather(spec.val)
    test = gather(spec.test)

    if train is None:
        raise RuntimeError("Empty train split.")
    print(
        f"[prepare] window counts: train={len(train['labels'])} "
        f"val={len(val['labels']) if val else 0} "
        f"test={len(test['labels']) if test else 0}"
    )

    print("[prepare] fitting normalization stats on TRAIN only")
    stats = fit_norm_stats(train["hand"], train["chest"], train["ankle"], train["hr"])

    processed_dir = Path(cfg.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    stats.save(processed_dir / "norm_stats.npz")

    for name, split in [("train", train), ("val", val), ("test", test)]:
        if split is None:
            continue
        nh, nc, na, nhr = apply_norm(
            split["hand"], split["chest"], split["ankle"], split["hr"], stats
        )
        save_split(
            processed_dir, name,
            hand=nh, chest=nc, ankle=na, hr=nhr,
            labels=split["labels"], subjects=split["subjects"],
        )
        print(
            f"[prepare] wrote {name}: hand={nh.shape} chest={nc.shape} "
            f"ankle={na.shape} hr={nhr.shape} labels={split['labels'].shape}"
        )

    print(f"[prepare] done. processed_dir={processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
