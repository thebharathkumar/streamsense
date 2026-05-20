"""Train the three model variants and write a combined ablation markdown table.

Variants:
    chest_only                    single-modality baseline
    multimodal_no_hr              three IMUs, no heart rate
    multimodal_late_fusion        all three IMUs plus heart rate
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamsense.eval.metrics import write_ablation_table  # noqa: E402


VARIANTS = ["chest_only", "multimodal_no_hr", "multimodal_late_fusion"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training, only re-aggregate from existing summaries.")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print(f"[ablation] $ {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed (exit {res.returncode}): {cmd}")


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    py = sys.executable
    train_py = str(REPO_ROOT / "scripts" / "train.py")
    eval_py = str(REPO_ROOT / "scripts" / "evaluate.py")

    if not args.skip_train:
        for name in VARIANTS:
            _run([py, train_py, "--config", args.config,
                  "--fold", str(args.fold),
                  "--model-name", name,
                  "--max-epochs", str(args.max_epochs)])
            _run([py, eval_py, "--config", args.config,
                  "--fold", str(args.fold),
                  "--model-name", name,
                  "--split", "test"])

    reports_dir = Path(cfg.paths.reports_dir)
    rows = []
    pretty = {
        "chest_only": "chest IMU only",
        "multimodal_no_hr": "all IMUs, no HR",
        "multimodal_late_fusion": "full multimodal",
    }
    for name in VARIANTS:
        summary_path = reports_dir / f"{name}_fold{args.fold}_test_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing {summary_path}; run without --skip-train first.")
        s = json.loads(summary_path.read_text())
        rows.append({
            "name": pretty[name],
            "macro_f1": s["macro_f1"],
            "weighted_f1": s["weighted_f1"],
            "accuracy": s["accuracy"],
            "params": s["num_params"],
        })

    out = reports_dir / f"ablation_fold{args.fold}.md"
    write_ablation_table(rows, out)
    print(f"[ablation] wrote {out}")
    print(out.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
