"""Multimodal late-fusion model and chest-only baseline.

Forward signature accepts a dict so the caller does not need to remember
positional ordering of modalities. Returns logits of shape (B, num_classes).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .branch import HRBranch, SensorBranch


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class MultimodalLateFusion(nn.Module):
    """Three IMU SensorBranches + an HRBranch concatenated into a head.

    use_hr=False produces the "full minus heart rate" ablation.
    """

    def __init__(
        self,
        num_classes: int = 12,
        imu_channels: int = 9,
        hr_channels: int = 1,
        d_model: int = 128,
        cnn_channels: Sequence[int] = (64, 128, 256),
        cnn_kernel: int = 5,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.2,
        classifier_hidden: int = 256,
        input_length: int = 512,
        use_hr: bool = True,
    ) -> None:
        super().__init__()
        self.use_hr = use_hr
        self.input_length = input_length

        def _branch() -> SensorBranch:
            return SensorBranch(
                in_channels=imu_channels,
                cnn_channels=cnn_channels,
                cnn_kernel=cnn_kernel,
                d_model=d_model,
                nhead=transformer_heads,
                num_layers=transformer_layers,
                dropout=dropout,
                input_length=input_length,
            )

        self.hand_branch = _branch()
        self.chest_branch = _branch()
        self.ankle_branch = _branch()
        self.hr_branch = HRBranch(d_model=d_model, dropout=dropout) if use_hr else None

        fused_dim = 3 * d_model + (d_model if use_hr else 0)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(
        self,
        hand: torch.Tensor,
        chest: torch.Tensor,
        ankle: torch.Tensor,
        hr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.hand_branch(hand)
        c = self.chest_branch(chest)
        a = self.ankle_branch(ankle)
        feats = [h, c, a]
        if self.use_hr:
            if hr is None:
                raise ValueError("use_hr=True but hr tensor not provided")
            feats.append(self.hr_branch(hr))
        z = torch.cat(feats, dim=1)
        return self.head(z)


class ChestOnlyBaseline(nn.Module):
    """Single-modality baseline using only the chest IMU.

    Same SensorBranch as the full model so any improvement is attributable
    to fusion rather than backbone differences.
    """

    def __init__(
        self,
        num_classes: int = 12,
        imu_channels: int = 9,
        d_model: int = 128,
        cnn_channels: Sequence[int] = (64, 128, 256),
        cnn_kernel: int = 5,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.2,
        classifier_hidden: int = 256,
        input_length: int = 512,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.chest_branch = SensorBranch(
            in_channels=imu_channels,
            cnn_channels=cnn_channels,
            cnn_kernel=cnn_kernel,
            d_model=d_model,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout,
            input_length=input_length,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(
        self,
        chest: torch.Tensor,
        hand: torch.Tensor | None = None,
        ankle: torch.Tensor | None = None,
        hr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self.chest_branch(chest)
        return self.head(z)


def build_model(cfg, num_classes: int) -> nn.Module:
    """Construct a model from an OmegaConf cfg.model section."""
    name = cfg.name
    common = dict(
        num_classes=num_classes,
        imu_channels=int(cfg.get("imu_channels", 9)),
        d_model=int(cfg.d_model),
        cnn_channels=list(cfg.cnn_channels),
        cnn_kernel=int(cfg.cnn_kernel),
        transformer_layers=int(cfg.transformer_layers),
        transformer_heads=int(cfg.transformer_heads),
        dropout=float(cfg.dropout),
        classifier_hidden=int(cfg.classifier_hidden),
        input_length=int(cfg.get("input_length", 512)),
    )
    if name == "multimodal_late_fusion":
        return MultimodalLateFusion(use_hr=True, **common)
    if name == "multimodal_no_hr":
        return MultimodalLateFusion(use_hr=False, **common)
    if name == "chest_only":
        return ChestOnlyBaseline(**common)
    raise ValueError(f"Unknown model name: {name}")
