"""Static int8 quantization and fp16 conversion of a trained ONNX model.

Calibration uses a subset of the validation split (real activations, not
random tensors).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamsense.export.quantize import (  # noqa: E402
    MultimodalCalibReader,
    convert_fp16,
    quantize_int8,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--fp32", default=None, help="Path to fp32 ONNX file.")
    p.add_argument("--int8-out", default=None)
    p.add_argument("--fp16-out", default=None)
    p.add_argument("--skip-fp16", action="store_true")
    p.add_argument("--skip-int8", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    onnx_dir = Path(cfg.paths.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg.model.name
    fp32 = Path(args.fp32) if args.fp32 else onnx_dir / f"{model_name}_fold{args.fold}_fp32.onnx"
    int8 = Path(args.int8_out) if args.int8_out else onnx_dir / f"{model_name}_fold{args.fold}_int8.onnx"
    fp16 = Path(args.fp16_out) if args.fp16_out else onnx_dir / f"{model_name}_fold{args.fold}_fp16.onnx"

    if not fp32.exists():
        raise FileNotFoundError(f"fp32 ONNX not found: {fp32}. Run scripts/export_onnx.py first.")

    if not args.skip_fp16:
        print(f"[quantize] fp16: {fp32} -> {fp16}")
        convert_fp16(fp32, fp16)
        print(f"[quantize] wrote {fp16} ({fp16.stat().st_size / 1e6:.2f} MB)")

    if not args.skip_int8:
        processed_dir = Path(cfg.paths.processed_dir)
        n_calib = int(cfg.export.calibration_windows)
        hand = np.load(processed_dir / "val_hand.npy", mmap_mode="r")[:n_calib]
        chest = np.load(processed_dir / "val_chest.npy", mmap_mode="r")[:n_calib]
        ankle = np.load(processed_dir / "val_ankle.npy", mmap_mode="r")[:n_calib]
        hr = np.load(processed_dir / "val_hr.npy", mmap_mode="r")[:n_calib]
        # Materialize so mmap'd readers don't get re-read across the QDQ pass.
        reader = MultimodalCalibReader(
            np.ascontiguousarray(hand),
            np.ascontiguousarray(chest),
            np.ascontiguousarray(ankle),
            np.ascontiguousarray(hr),
            batch_size=8,
        )
        print(f"[quantize] int8 static: {fp32} -> {int8} (n_calib={n_calib})")
        quantize_int8(fp32, int8, reader)
        print(f"[quantize] wrote {int8} ({int8.stat().st_size / 1e6:.2f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
