"""Evaluation CLI: load best checkpoint, score on test split, write reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ..data.dataset import WindowDataset
from ..models.fusion import ChestOnlyBaseline, MultimodalLateFusion, build_model, count_parameters
from ..train.lightning_module import HARLightning
from .metrics import classification_report, save_confusion_png


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--model-name", default=None)
    p.add_argument("--ckpt", default=None, help="Override checkpoint path.")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--tag", default=None)
    return p.parse_args()


def _resolve_ckpt(cfg, args) -> Path:
    if args.ckpt:
        return Path(args.ckpt)
    model_name = args.model_name or cfg.model.name
    run_name = f"{model_name}_fold{args.fold}" + (f"_{args.tag}" if args.tag else "")
    candidate = Path(cfg.paths.checkpoints_dir) / run_name / "best.ckpt"
    if not candidate.exists():
        candidate = candidate.with_name("last.ckpt")
    if not candidate.exists():
        raise FileNotFoundError(f"No checkpoint found under {candidate.parent}")
    return candidate


def _predict(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            hand = batch["hand"].to(device)
            chest = batch["chest"].to(device)
            ankle = batch["ankle"].to(device)
            hr = batch["hr"].to(device)
            label = batch["label"].to(device)
            if isinstance(model, MultimodalLateFusion):
                logits = model(hand, chest, ankle, hr if model.use_hr else None)
            elif isinstance(model, ChestOnlyBaseline):
                logits = model(chest)
            else:
                raise TypeError(type(model))
            preds_all.append(logits.argmax(dim=1).cpu().numpy())
            labels_all.append(label.cpu().numpy())
    return np.concatenate(preds_all), np.concatenate(labels_all)


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.model_name:
        cfg.model.name = args.model_name

    processed_dir = Path(cfg.paths.processed_dir)
    ds = WindowDataset(processed_dir, args.split)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    num_classes = int(cfg.model.num_classes)
    model = build_model(cfg.model, num_classes=num_classes)

    ckpt_path = _resolve_ckpt(cfg, args)
    print(f"[eval] loading {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # The checkpoint stores the wrapped HARLightning. Strip the 'model.' prefix.
    state_dict = state.get("state_dict", state)
    fixed = {}
    for k, v in state_dict.items():
        fixed[k.removeprefix("model.")] = v
    missing, unexpected = model.load_state_dict(fixed, strict=False)
    if missing:
        print(f"[eval] missing keys: {missing}")
    if unexpected:
        print(f"[eval] unexpected keys: {unexpected}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    preds, labels = _predict(model, loader, device)
    activity_names = OmegaConf.to_container(cfg.data.activity_names, resolve=True)
    class_names = [activity_names[a] for a in cfg.data.activity_ids]

    report = classification_report(labels, preds, class_names)
    print(report.to_markdown())

    reports_dir = Path(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{cfg.model.name}_fold{args.fold}" + (f"_{args.tag}" if args.tag else "")
    md_path = reports_dir / f"{run_name}_{args.split}_report.md"
    md_path.write_text(report.to_markdown())

    cm_path = reports_dir / f"{run_name}_{args.split}_confusion.png"
    if bool(cfg.eval.save_confusion_matrix):
        save_confusion_png(report.confusion, class_names, cm_path)

    summary = {
        "fold": args.fold,
        "split": args.split,
        "model_name": cfg.model.name,
        "checkpoint": str(ckpt_path),
        "num_params": count_parameters(model),
        "macro_f1": report.macro_f1,
        "weighted_f1": report.weighted_f1,
        "accuracy": report.accuracy,
        "confusion_png": str(cm_path) if cfg.eval.save_confusion_matrix else None,
        "report_md": str(md_path),
    }
    (reports_dir / f"{run_name}_{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
