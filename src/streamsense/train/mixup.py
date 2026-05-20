"""Mixup augmentation for multimodal inputs.

Same lambda is applied across all modalities for a given batch so the four
streams stay temporally aligned.
"""

from __future__ import annotations

import numpy as np
import torch


def mixup_batch(
    batch: dict[str, torch.Tensor],
    alpha: float,
    num_classes: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Apply mixup to the multimodal batch.

    Returns the mixed batch and a (B, num_classes) soft-label tensor that
    callers should use with a cross-entropy variant supporting soft targets
    (e.g. torch.nn.functional.cross_entropy with one-hot floats).
    """
    if alpha <= 0:
        labels = batch["label"]
        soft = torch.zeros(labels.size(0), num_classes, device=labels.device)
        soft.scatter_(1, labels.unsqueeze(1), 1.0)
        return batch, soft

    lam = float(np.random.beta(alpha, alpha))
    batch_size = batch["label"].size(0)
    perm = torch.randperm(batch_size, device=batch["label"].device)

    mixed = {}
    for key in ("hand", "chest", "ankle", "hr"):
        if key in batch:
            mixed[key] = lam * batch[key] + (1.0 - lam) * batch[key][perm]
    mixed["label"] = batch["label"]

    labels = batch["label"]
    one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    permuted = one_hot[perm]
    soft = lam * one_hot + (1.0 - lam) * permuted
    return mixed, soft


def soft_cross_entropy(
    logits: torch.Tensor, soft_targets: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    """Cross-entropy against soft targets, with optional label smoothing."""
    if label_smoothing > 0:
        n = soft_targets.size(1)
        soft_targets = soft_targets * (1.0 - label_smoothing) + label_smoothing / n
    log_probs = torch.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()
