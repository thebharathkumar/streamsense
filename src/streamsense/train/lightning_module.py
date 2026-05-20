"""Lightning module wrapping the multimodal HAR model.

Wires up AdamW + OneCycleLR, mixup with label smoothing, optional class
weighting, and macro F1 / accuracy logging on validation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from ..models.fusion import ChestOnlyBaseline, MultimodalLateFusion
from .mixup import mixup_batch, soft_cross_entropy


def _macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Macro F1 in pure torch, no sklearn dependency at training time."""
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


class HARLightning(pl.LightningModule):
    """Training wrapper around the HAR models."""

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        mixup_alpha: float = 0.2,
        class_weights: np.ndarray | None = None,
        onecycle_pct_start: float = 0.1,
        total_steps: int | None = None,
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self.onecycle_pct_start = onecycle_pct_start
        self.total_steps = total_steps
        self.grad_clip = grad_clip

        if class_weights is not None:
            self.register_buffer(
                "class_weights", torch.as_tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

        # Persist counters via list buffers to keep evaluation hooks simple.
        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

        self.save_hyperparameters(ignore=["model", "class_weights"])

    def _forward_model(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(self.model, MultimodalLateFusion):
            hr = batch.get("hr") if self.model.use_hr else None
            return self.model(batch["hand"], batch["chest"], batch["ankle"], hr)
        if isinstance(self.model, ChestOnlyBaseline):
            return self.model(batch["chest"])
        raise TypeError(f"Unsupported model type: {type(self.model)}")

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        mixed, soft = mixup_batch(batch, self.mixup_alpha, self.num_classes)
        logits = self._forward_model(mixed)
        if self.class_weights is not None:
            # Reweight per-sample by the original label's class weight.
            weights = self.class_weights[batch["label"]]
            log_probs = torch.log_softmax(logits, dim=1)
            soft_smoothed = soft * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
            per_sample = -(soft_smoothed * log_probs).sum(dim=1)
            loss = (per_sample * weights).mean()
        else:
            loss = soft_cross_entropy(logits, soft, self.label_smoothing)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_preds.clear()
        self._val_targets.clear()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        logits = self._forward_model(batch)
        loss = nn.functional.cross_entropy(
            logits, batch["label"], label_smoothing=self.label_smoothing,
        )
        preds = logits.argmax(dim=1)
        self._val_preds.append(preds.detach().cpu())
        self._val_targets.append(batch["label"].detach().cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds)
        targets = torch.cat(self._val_targets)
        acc = (preds == targets).float().mean().item()
        macro = _macro_f1(preds, targets, self.num_classes)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_macro_f1", macro, prog_bar=True)
        # Always print one line per epoch so the run is followable when the
        # Lightning progress bar is disabled.
        train_loss = float(self.trainer.callback_metrics.get("train_loss", torch.tensor(float("nan"))))
        val_loss = float(self.trainer.callback_metrics.get("val_loss", torch.tensor(float("nan"))))
        print(
            f"[epoch {self.current_epoch:>2d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={acc:.4f} val_macro_f1={macro:.4f}",
            flush=True,
        )

    def configure_optimizers(self) -> Any:
        optim = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.total_steps is None or self.total_steps <= 0:
            return optim
        sched = OneCycleLR(
            optim,
            max_lr=self.lr,
            total_steps=self.total_steps,
            pct_start=self.onecycle_pct_start,
            anneal_strategy="cos",
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
