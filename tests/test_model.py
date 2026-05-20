"""Tests for model forward/backward shape and param count."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamsense.models.fusion import (  # noqa: E402
    ChestOnlyBaseline,
    MultimodalLateFusion,
    count_parameters,
)


B, T = 2, 512


@pytest.fixture
def dummy_inputs():
    return {
        "hand": torch.randn(B, 9, T),
        "chest": torch.randn(B, 9, T),
        "ankle": torch.randn(B, 9, T),
        "hr": torch.randn(B, 1, T),
    }


def test_multimodal_forward_shape(dummy_inputs):
    m = MultimodalLateFusion(num_classes=12)
    out = m(dummy_inputs["hand"], dummy_inputs["chest"], dummy_inputs["ankle"], dummy_inputs["hr"])
    assert out.shape == (B, 12)


def test_multimodal_no_hr_forward_shape(dummy_inputs):
    m = MultimodalLateFusion(num_classes=12, use_hr=False)
    out = m(dummy_inputs["hand"], dummy_inputs["chest"], dummy_inputs["ankle"], None)
    assert out.shape == (B, 12)


def test_chest_only_forward_shape(dummy_inputs):
    m = ChestOnlyBaseline(num_classes=12)
    out = m(dummy_inputs["chest"])
    assert out.shape == (B, 12)


def test_param_count_under_budget():
    m = MultimodalLateFusion(num_classes=12)
    n = count_parameters(m)
    assert n < 5_000_000, f"model has {n:,} params, exceeds 5M budget"


def test_backward_pass(dummy_inputs):
    m = MultimodalLateFusion(num_classes=12)
    out = m(dummy_inputs["hand"], dummy_inputs["chest"], dummy_inputs["ankle"], dummy_inputs["hr"])
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None and p.requires_grad]
    assert grads, "no gradients populated after backward"
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_use_hr_true_requires_hr_tensor(dummy_inputs):
    m = MultimodalLateFusion(num_classes=12, use_hr=True)
    with pytest.raises(ValueError):
        m(dummy_inputs["hand"], dummy_inputs["chest"], dummy_inputs["ankle"], None)


def test_dynamic_batch_dimension():
    m = MultimodalLateFusion(num_classes=12)
    m.eval()
    for batch in (1, 3, 16):
        h = torch.randn(batch, 9, T)
        c = torch.randn(batch, 9, T)
        a = torch.randn(batch, 9, T)
        hr = torch.randn(batch, 1, T)
        with torch.no_grad():
            out = m(h, c, a, hr)
        assert out.shape == (batch, 12)
