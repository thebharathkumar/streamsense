"""Latency benchmark for fp32, fp16, and int8 ONNX models.

Measures batch=1 single-thread CPU latency with onnxruntime and writes a
markdown table with p50, p95, p99 in milliseconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort


def make_session(model_path: Path, intra_op_threads: int = 1) -> ort.InferenceSession:
    """Create a single-threaded CPU session."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_op_threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])


@dataclass
class LatencyStats:
    """Summary of one variant's latency distribution."""

    name: str
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    model_path: str

    def row(self) -> str:
        return (
            f"| {self.name} | {self.p50_ms:.2f} | {self.p95_ms:.2f} | "
            f"{self.p99_ms:.2f} | {self.mean_ms:.2f} | {self.max_ms:.2f} |"
        )


def benchmark_session(
    sess: ort.InferenceSession,
    name: str,
    feeds_factory,
    warmup: int = 20,
    iters: int = 200,
) -> LatencyStats:
    """Run latency measurement against an existing session."""
    # Warmup.
    for _ in range(warmup):
        sess.run(None, feeds_factory())
    # Measure.
    times: list[float] = []
    for _ in range(iters):
        feeds = feeds_factory()
        t0 = time.perf_counter()
        sess.run(None, feeds)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    arr = np.asarray(times)
    return LatencyStats(
        name=name, n=iters,
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(arr.mean()),
        max_ms=float(arr.max()),
        model_path=sess.get_modelmeta().graph_name or "",
    )


def write_latency_table(rows: list[LatencyStats], out_path: Path) -> Path:
    header = "| variant | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | max (ms) |"
    sep = "|---|---|---|---|---|---|"
    body = [r.row() for r in rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join([header, sep, *body, ""]))
    return out_path


def multimodal_feeds(window_samples: int = 512, seed: int = 0):
    """Factory returning fresh feeds each call (avoids ORT memory reuse skewing timings)."""
    rng = np.random.default_rng(seed)

    def _make() -> dict[str, np.ndarray]:
        return {
            "hand": rng.standard_normal((1, 9, window_samples)).astype(np.float32),
            "chest": rng.standard_normal((1, 9, window_samples)).astype(np.float32),
            "ankle": rng.standard_normal((1, 9, window_samples)).astype(np.float32),
            "hr": rng.standard_normal((1, 1, window_samples)).astype(np.float32),
        }
    return _make
