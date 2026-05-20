"""PAMAP2 download and extraction.

The official PAMAP2 archive lives at the UCI ML Repository:
    https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip

This module fetches the zip, optionally verifies a SHA256 checksum if one is
provided in the environment, and extracts the protocol .dat files under
data/raw/PAMAP2_Dataset/Protocol/.

Network access is not assumed. If the file is already present it is reused.
A --synthetic mode in prepare_data.py generates PAMAP2-shaped data so the
rest of the pipeline can run in environments without internet.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

PAMAP2_URL = (
    "https://archive.ics.uci.edu/static/public/231/"
    "pamap2+physical+activity+monitoring.zip"
)
PAMAP2_ZIP_NAME = "pamap2.zip"
# 41 PAMAP2 columns per protocol .dat file.
PAMAP2_NUM_COLUMNS = 54


def expected_sha256() -> Optional[str]:
    """Return the SHA256 expected for the official zip, if pinned via env."""
    return os.environ.get("PAMAP2_SHA256")


def _sha256_of_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def download_pamap2(raw_dir: Path, url: str = PAMAP2_URL, force: bool = False) -> Path:
    """Fetch the PAMAP2 zip into raw_dir and return its path.

    Idempotent: re-runs are no-ops unless force=True. Verifies SHA256 only if
    PAMAP2_SHA256 is set in the environment, since UCI rebuilds the zip
    occasionally and we do not want to break the build on a benign change.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / PAMAP2_ZIP_NAME

    if zip_path.exists() and not force:
        print(f"[download] reusing {zip_path}")
    else:
        print(f"[download] fetching {url}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with zip_path.open("wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc="pamap2"
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
                        pbar.update(len(chunk))

    pinned = expected_sha256()
    if pinned:
        digest = _sha256_of_file(zip_path)
        if digest.lower() != pinned.lower():
            raise ValueError(
                f"PAMAP2 zip checksum mismatch: expected {pinned}, got {digest}"
            )
        print(f"[download] checksum OK ({digest[:12]}...)")
    else:
        print("[download] PAMAP2_SHA256 not set, skipping checksum verification")

    return zip_path


def extract_pamap2(zip_path: Path, raw_dir: Path, force: bool = False) -> Path:
    """Extract the protocol .dat files. Returns the protocol directory."""
    target_root = raw_dir / "PAMAP2_Dataset"
    protocol_dir = target_root / "Protocol"

    if protocol_dir.exists() and any(protocol_dir.glob("subject*.dat")) and not force:
        print(f"[extract] reusing {protocol_dir}")
        return protocol_dir

    if force and target_root.exists():
        shutil.rmtree(target_root)

    print(f"[extract] unpacking {zip_path} into {raw_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        # The official zip contains a single nested zip in some mirrors and
        # the dataset folder directly in others. Handle both.
        members = zf.namelist()
        inner_zips = [m for m in members if m.lower().endswith(".zip")]
        if inner_zips:
            for m in inner_zips:
                zf.extract(m, raw_dir)
                with zipfile.ZipFile(raw_dir / m) as inner:
                    inner.extractall(raw_dir)
        else:
            zf.extractall(raw_dir)

    if not protocol_dir.exists():
        # Some mirrors lay the data out as raw_dir/Protocol/ rather than
        # raw_dir/PAMAP2_Dataset/Protocol/. Normalize.
        candidate = raw_dir / "Protocol"
        if candidate.exists():
            target_root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidate), str(protocol_dir))

    if not protocol_dir.exists():
        raise FileNotFoundError(
            f"Could not locate Protocol/ under {raw_dir} after extraction"
        )

    return protocol_dir


def fetch_and_extract(raw_dir: Path, force: bool = False) -> Path:
    """Download + extract in one shot. Returns the protocol directory."""
    zip_path = download_pamap2(raw_dir, force=force)
    return extract_pamap2(zip_path, raw_dir, force=force)
