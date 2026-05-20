"""Torch to ONNX export with a dynamic batch dimension.

The exported graph accepts (hand, chest, ankle, hr) for the multimodal
model and (chest,) for the baseline. Window length is fixed at 512.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from ..models.fusion import ChestOnlyBaseline, MultimodalLateFusion


class MultimodalONNXWrapper(torch.nn.Module):
    """Wraps MultimodalLateFusion so ONNX export sees fixed positional args."""

    def __init__(self, model: MultimodalLateFusion) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        hand: torch.Tensor,
        chest: torch.Tensor,
        ankle: torch.Tensor,
        hr: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(hand, chest, ankle, hr if self.model.use_hr else None)


def export_to_onnx(
    model: torch.nn.Module,
    out_path: Path,
    window_samples: int = 512,
    opset: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export a HAR model to ONNX. Raises if the opset is too low for SDPA."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dynamic_axes: dict[str, dict[int, str]] | None
    if dynamic_batch:
        dynamic_axes = {name: {0: "batch"} for name in
                        ["hand", "chest", "ankle", "hr", "logits"]}
    else:
        dynamic_axes = None

    def _try_export(opset_in: int) -> None:
        if isinstance(model, MultimodalLateFusion):
            wrapper = MultimodalONNXWrapper(model)
            inputs = (
                torch.randn(1, 9, window_samples),
                torch.randn(1, 9, window_samples),
                torch.randn(1, 9, window_samples),
                torch.randn(1, 1, window_samples),
            )
            input_names = ["hand", "chest", "ankle", "hr"]
            torch.onnx.export(
                wrapper, inputs, str(out_path),
                input_names=input_names, output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=opset_in,
                dynamo=False,
            )
        elif isinstance(model, ChestOnlyBaseline):
            inputs = (torch.randn(1, 9, window_samples),)
            input_names = ["chest"]
            dax = (
                {name: {0: "batch"} for name in ["chest", "logits"]}
                if dynamic_batch else None
            )
            torch.onnx.export(
                model, inputs, str(out_path),
                input_names=input_names, output_names=["logits"],
                dynamic_axes=dax,
                opset_version=opset_in,
                dynamo=False,
            )
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")

    # nn.TransformerEncoder uses scaled_dot_product_attention internally,
    # which may need opset 18+ on some torch versions. Try the requested opset
    # first, fall back to 18 if it fails.
    try:
        _try_export(opset)
    except Exception as e:
        if opset < 18:
            print(f"[onnx] opset {opset} failed ({e!r}); retrying opset 18")
            _try_export(18)
        else:
            raise

    return out_path


def calibration_batches(
    hand: "any", chest: "any", ankle: "any", hr: "any",
    batch_size: int = 16,
    max_batches: int = 16,
) -> Iterable[dict]:
    """Yield calibration batches from preloaded numpy arrays."""
    n = hand.shape[0]
    for start in range(0, min(n, max_batches * batch_size), batch_size):
        end = min(start + batch_size, n)
        yield {
            "hand": hand[start:end].astype("float32"),
            "chest": chest[start:end].astype("float32"),
            "ankle": ankle[start:end].astype("float32"),
            "hr": hr[start:end].astype("float32"),
        }
