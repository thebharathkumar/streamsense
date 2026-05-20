"""Export the best checkpoint of a trained model to ONNX (fp32)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamsense.export.onnx_export import export_to_onnx  # noqa: E402
from streamsense.models.fusion import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--model-name", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.model_name:
        cfg.model.name = args.model_name

    num_classes = int(cfg.model.num_classes)
    model = build_model(cfg.model, num_classes=num_classes)

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        run_name = f"{cfg.model.name}_fold{args.fold}"
        ckpt_path = Path(cfg.paths.checkpoints_dir) / run_name / "best.ckpt"
    print(f"[export] loading {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    fixed = {k.removeprefix("model."): v for k, v in sd.items()}
    model.load_state_dict(fixed, strict=False)

    out_path = Path(args.out) if args.out else Path(cfg.paths.onnx_dir) / f"{cfg.model.name}_fold{args.fold}_fp32.onnx"
    window_samples = int(round(float(cfg.data.window_seconds) * float(cfg.data.sampling_rate_hz)))
    export_to_onnx(
        model, out_path,
        window_samples=window_samples,
        opset=int(cfg.export.opset),
        dynamic_batch=bool(cfg.export.dynamic_batch),
    )
    print(f"[export] wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
