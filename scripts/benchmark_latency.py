"""Latency benchmark across fp32, fp16, int8 ONNX models.

Single-thread CPU, batch=1. Writes a markdown table under artifacts/reports/.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import onnxruntime as ort
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamsense.export.benchmark import (  # noqa: E402
    benchmark_session,
    make_session,
    multimodal_feeds,
    write_latency_table,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--model-name", default=None)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.model_name:
        cfg.model.name = args.model_name

    onnx_dir = Path(cfg.paths.onnx_dir)
    name = cfg.model.name
    fold = args.fold

    variants = [
        ("fp32", onnx_dir / f"{name}_fold{fold}_fp32.onnx"),
        ("fp16", onnx_dir / f"{name}_fold{fold}_fp16.onnx"),
        ("int8", onnx_dir / f"{name}_fold{fold}_int8.onnx"),
    ]

    print(f"[bench] onnxruntime={ort.__version__} platform={platform.platform()}")

    feeds = multimodal_feeds(window_samples=int(round(
        float(cfg.data.window_seconds) * float(cfg.data.sampling_rate_hz)
    )))

    rows = []
    for variant_name, path in variants:
        if not path.exists():
            print(f"[bench] skipping {variant_name}: {path} not found")
            continue
        sess = make_session(path, intra_op_threads=1)
        stats = benchmark_session(
            sess, variant_name, feeds, warmup=args.warmup, iters=args.iters,
        )
        rows.append(stats)
        print(
            f"[bench] {variant_name:5s} p50={stats.p50_ms:.2f}ms  "
            f"p95={stats.p95_ms:.2f}ms  p99={stats.p99_ms:.2f}ms  "
            f"mean={stats.mean_ms:.2f}ms"
        )

    if not rows:
        raise RuntimeError("No ONNX variants found. Run export and quantize first.")

    out_md = Path(cfg.paths.reports_dir) / f"{name}_fold{fold}_latency.md"
    write_latency_table(rows, out_md)
    print(f"[bench] wrote {out_md}")

    summary = {
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "iters": args.iters,
        "warmup": args.warmup,
        "variants": [
            {
                "name": r.name,
                "p50_ms": r.p50_ms,
                "p95_ms": r.p95_ms,
                "p99_ms": r.p99_ms,
                "mean_ms": r.mean_ms,
            }
            for r in rows
        ],
    }
    (out_md.with_suffix(".json")).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
