"""Torch vs ONNX Runtime numerical parity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamsense.export.onnx_export import export_to_onnx  # noqa: E402
from streamsense.models.fusion import ChestOnlyBaseline, MultimodalLateFusion  # noqa: E402

onnxruntime = pytest.importorskip("onnxruntime")


T = 512
TOL = 1e-4


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)


def test_onnx_parity_multimodal(tmp_path):
    torch.manual_seed(0)
    model = MultimodalLateFusion(num_classes=12)
    model.eval()

    hand = torch.randn(2, 9, T)
    chest = torch.randn(2, 9, T)
    ankle = torch.randn(2, 9, T)
    hr = torch.randn(2, 1, T)

    with torch.no_grad():
        torch_out = model(hand, chest, ankle, hr).cpu().numpy()

    out_path = tmp_path / "multi.onnx"
    export_to_onnx(model, out_path, window_samples=T, opset=17, dynamic_batch=True)

    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = 1
    sess = onnxruntime.InferenceSession(
        str(out_path), sess_options=so, providers=["CPUExecutionProvider"]
    )
    onnx_out = sess.run(None, {
        "hand": _np(hand), "chest": _np(chest), "ankle": _np(ankle), "hr": _np(hr),
    })[0]

    np.testing.assert_allclose(torch_out, onnx_out, atol=TOL, rtol=TOL)


def test_onnx_parity_chest_only(tmp_path):
    torch.manual_seed(1)
    model = ChestOnlyBaseline(num_classes=12)
    model.eval()
    chest = torch.randn(3, 9, T)

    with torch.no_grad():
        torch_out = model(chest).cpu().numpy()

    out_path = tmp_path / "chest.onnx"
    export_to_onnx(model, out_path, window_samples=T, opset=17, dynamic_batch=True)

    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = 1
    sess = onnxruntime.InferenceSession(
        str(out_path), sess_options=so, providers=["CPUExecutionProvider"]
    )
    onnx_out = sess.run(None, {"chest": _np(chest)})[0]
    np.testing.assert_allclose(torch_out, onnx_out, atol=TOL, rtol=TOL)


def test_dynamic_batch_in_onnx(tmp_path):
    torch.manual_seed(2)
    model = MultimodalLateFusion(num_classes=12)
    model.eval()
    out_path = tmp_path / "multi.onnx"
    export_to_onnx(model, out_path, window_samples=T, opset=17, dynamic_batch=True)

    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = 1
    sess = onnxruntime.InferenceSession(
        str(out_path), sess_options=so, providers=["CPUExecutionProvider"]
    )
    for b in (1, 4, 8):
        rng = np.random.default_rng(b)
        feeds = {
            "hand": rng.standard_normal((b, 9, T)).astype(np.float32),
            "chest": rng.standard_normal((b, 9, T)).astype(np.float32),
            "ankle": rng.standard_normal((b, 9, T)).astype(np.float32),
            "hr": rng.standard_normal((b, 1, T)).astype(np.float32),
        }
        out = sess.run(None, feeds)[0]
        assert out.shape == (b, 12)
