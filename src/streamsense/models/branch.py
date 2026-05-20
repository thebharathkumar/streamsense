"""Per-sensor encoder branch: 1D CNN backbone followed by transformer encoder.

Input  : (B, C, T) where T is window length in samples (512 for 5.12s @ 100Hz).
Output : (B, d_model) embedding after temporal mean-pool.

The backbone uses three stride-1 conv blocks with max-pool stride 2 between
them, downsampling T by 8x before the transformer sees it.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> GELU -> MaxPool(stride=2)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class SensorBranch(nn.Module):
    """One IMU's CNN + transformer encoder.

    The transformer operates over time after the conv stack. We use a learned
    positional embedding sized to the post-CNN sequence length.
    """

    def __init__(
        self,
        in_channels: int,
        cnn_channels: Sequence[int] = (64, 128, 256),
        cnn_kernel: int = 5,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        input_length: int = 512,
    ) -> None:
        super().__init__()

        blocks: list[nn.Module] = []
        prev = in_channels
        for out in cnn_channels:
            blocks.append(ConvBlock(prev, out, cnn_kernel))
            prev = out
        self.cnn = nn.Sequential(*blocks)
        self.cnn_out_channels = prev

        self.project = nn.Conv1d(self.cnn_out_channels, d_model, kernel_size=1)

        # Post-CNN sequence length after three stride-2 pools.
        self.seq_len = input_length // (2 ** len(cnn_channels))
        if self.seq_len <= 0:
            raise ValueError(
                f"input_length={input_length} too small for {len(cnn_channels)} pools."
            )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        x = self.cnn(x)            # (B, C_cnn, T/8)
        x = self.project(x)        # (B, d_model, T/8)
        x = x.transpose(1, 2)      # (B, T/8, d_model)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.transformer(x)    # (B, T/8, d_model)
        x = self.norm(x)
        x = x.mean(dim=1)          # temporal mean-pool: (B, d_model)
        x = self.dropout(x)
        return x


class HRBranch(nn.Module):
    """Small MLP for the heart rate channel.

    HR is a single-channel slow-varying signal; a few statistics + a tiny MLP
    is enough. We use mean, std, min, max, and the per-quarter means as
    features, projected to d_model.
    """

    def __init__(self, d_model: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(8, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.d_model = d_model

    @staticmethod
    def _features(hr: torch.Tensor) -> torch.Tensor:
        # hr: (B, 1, T) -> (B, 8)
        x = hr.squeeze(1)
        T = x.size(1)
        quarter = T // 4
        q = [x[:, i * quarter : (i + 1) * quarter].mean(dim=1) for i in range(4)]
        feats = torch.stack(
            [x.mean(dim=1), x.std(dim=1), x.amin(dim=1), x.amax(dim=1), *q],
            dim=1,
        )
        return feats

    def forward(self, hr: torch.Tensor) -> torch.Tensor:
        return self.proj(self._features(hr))
