"""Fetch the PAMAP2 archive and extract Protocol/ into data/raw/.

Network access is required. The official source is the UCI ML Repository.
Set PAMAP2_SHA256=<hex> in the environment to pin the expected checksum.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamsense.data.download import fetch_and_extract  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    raw_dir = Path(cfg.paths.raw_dir)
    protocol_dir = fetch_and_extract(raw_dir, force=args.force)
    dats = sorted(protocol_dir.glob("subject*.dat"))
    print(f"[download] protocol_dir={protocol_dir} ({len(dats)} subject files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
