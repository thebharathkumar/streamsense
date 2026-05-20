"""Latency smoke test for the int8 ONNX path.

By default we assert a loose 100 ms p95 ceiling so this is green on slow CI
runners. Set STRICT_LATENCY=1 in the environment to tighten to 20 ms, which
is the spec target on a modern laptop CPU.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamsense.export.onnx_export import export_to_onnx  # noqa: E402
from streamsense.export.quantize import MultimodalCalibReader, quantize_int8  # noqa: E402
from streamsense.models.fusion import MultimodalLateFusion  # noqa: E402

onnxruntime = pytest.importorskip("onnxruntime")


T = 512


def _make_calib_arrays(n: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return (
        rng.standard_normal((n, 9, T)).astype(np.float32),
        rng.standard_normal((n, 9, T)).astype(np.float32),
        rng.standard_normal((n, 9, T)).astype(np.float32),
        rng.standard_normal((n, 1, T)).astype(np.float32),
    )


def test_int8_latency_p95_under_threshold(tmp_path):
    threshold_ms = 20.0 if os.environ.get("STRICT_LATENCY") == "1" else 100.0

    torch.manual_seed(0)
    model = MultimodalLateFusion(num_classes=12)
    model.eval()

    fp32_path = tmp_path / "fp32.onnx"
    int8_path = tmp_path / "int8.onnx"
    export_to_onnx(model, fp32_path, window_samples=T, opset=17, dynamic_batch=True)

    hand, chest, ankle, hr = _make_calib_arrays(32)
    reader = MultimodalCalibReader(hand, chest, ankle, hr, batch_size=8)
    quantize_int8(fp32_path, int8_path, reader)

    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = onnxruntime.InferenceSession(
        str(int8_path), sess_options=so, providers=["CPUExecutionProvider"]
    )

    rng = np.random.default_rng(1)
    def feeds():
        return {
            "hand": rng.standard_normal((1, 9, T)).astype(np.float32),
            "chest": rng.standard_normal((1, 9, T)).astype(np.float32),
            "ankle": rng.standard_normal((1, 9, T)).astype(np.float32),
            "hr": rng.standard_normal((1, 1, T)).astype(np.float32),
        }

    for _ in range(10):
        sess.run(None, feeds())

    times = []
    for _ in range(100):
        f = feeds()
        t0 = time.perf_counter()
        sess.run(None, f)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    p95 = float(np.percentile(times, 95))
    print(f"\nint8 p50={np.percentile(times, 50):.2f}ms p95={p95:.2f}ms p99={np.percentile(times, 99):.2f}ms")
    assert p95 < threshold_ms, f"int8 p95={p95:.2f}ms exceeds {threshold_ms}ms"
