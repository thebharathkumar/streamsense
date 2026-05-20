"""Training CLI: loads processed splits and runs Lightning Trainer."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from ..data.dataset import WindowDataset, inverse_freq_weights
from ..models.fusion import build_model, count_parameters
from .lightning_module import HARLightning


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--fold", type=int, default=0, help="Fold id used to name the checkpoint dir.")
    p.add_argument("--model-name", default=None, help="Override cfg.model.name (chest_only, multimodal_no_hr, multimodal_late_fusion).")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--fast", action="store_true", help="Limit batches per epoch for a smoke run.")
    p.add_argument("--limit-train-batches", type=float, default=None)
    p.add_argument("--limit-val-batches", type=float, default=None)
    p.add_argument("--tag", default=None, help="Optional tag appended to the run name.")
    return p.parse_args()


def build_dataloader(ds: WindowDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    if args.model_name:
        cfg.model.name = args.model_name

    seed_everything(int(cfg.seed))

    processed_dir = Path(cfg.paths.processed_dir)
    train_ds = WindowDataset(processed_dir, "train")
    val_ds = WindowDataset(processed_dir, "val")

    num_classes = int(cfg.model.num_classes)

    print(f"[train] train={len(train_ds)} val={len(val_ds)} classes={num_classes}")
    print(f"[train] torch={torch.__version__} platform={platform.platform()}")

    model = build_model(cfg.model, num_classes=num_classes)
    n_params = count_parameters(model)
    print(f"[train] model={cfg.model.name} params={n_params:,d}")

    weights = None
    if cfg.train.class_weighting == "inverse_freq":
        weights = inverse_freq_weights(train_ds.labels, num_classes)
        print(f"[train] class weights (mean=1): {np.round(weights, 3).tolist()}")

    max_epochs = int(args.max_epochs if args.max_epochs is not None else cfg.train.max_epochs)
    batch_size = int(cfg.train.batch_size)
    num_workers = int(cfg.train.num_workers)

    train_loader = build_dataloader(train_ds, batch_size, num_workers, shuffle=True)
    val_loader = build_dataloader(val_ds, batch_size, num_workers, shuffle=False)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * max_epochs

    lit = HARLightning(
        model=model,
        num_classes=num_classes,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        label_smoothing=float(cfg.train.label_smoothing),
        mixup_alpha=float(cfg.train.mixup_alpha),
        class_weights=weights,
        onecycle_pct_start=float(cfg.train.onecycle_pct_start),
        total_steps=total_steps,
        grad_clip=float(cfg.train.grad_clip),
    )

    run_name = f"{cfg.model.name}_fold{args.fold}" + (f"_{args.tag}" if args.tag else "")
    ckpt_dir = Path(cfg.paths.checkpoints_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tb_logger = TensorBoardLogger(
        save_dir=cfg.paths.tensorboard_dir, name=run_name, default_hp_metric=False,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best",
            monitor=cfg.train.monitor,
            mode=cfg.train.monitor_mode,
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor=cfg.train.monitor,
            mode=cfg.train.monitor_mode,
            patience=int(cfg.train.early_stopping_patience),
        ),
    ]

    fast_kwargs: dict = {}
    if args.fast:
        fast_kwargs.update(limit_train_batches=0.5, limit_val_batches=1.0)
    if args.limit_train_batches is not None:
        fast_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        fast_kwargs["limit_val_batches"] = args.limit_val_batches

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu" if not torch.cuda.is_available() else "auto",
        precision=str(cfg.train.precision),
        logger=tb_logger,
        callbacks=callbacks,
        gradient_clip_val=float(cfg.train.grad_clip),
        log_every_n_steps=10,
        deterministic=False,
        enable_progress_bar=True,
        **fast_kwargs,
    )

    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
    summary = {
        "fold": args.fold,
        "model_name": cfg.model.name,
        "num_params": n_params,
        "best_checkpoint": best,
        "max_epochs": max_epochs,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "best_val_macro_f1": float(trainer.callback_metrics.get("val_macro_f1", 0.0)),
        "best_val_acc": float(trainer.callback_metrics.get("val_acc", 0.0)),
    }
    out = Path(cfg.paths.reports_dir) / f"{run_name}_train_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[train] summary={out}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
