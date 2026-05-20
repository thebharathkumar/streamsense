"""Generate tests/fixtures/example_window.json for README curl examples.

Run with: python tests/fixtures/generate_example.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

T = 512

rng = np.random.default_rng(42)
payload = {
    "sampling_rate_hz": 100,
    "hand":  {"channels": rng.standard_normal((9, T)).round(4).tolist()},
    "chest": {"channels": rng.standard_normal((9, T)).round(4).tolist()},
    "ankle": {"channels": rng.standard_normal((9, T)).round(4).tolist()},
    "hr":    {"values":   rng.standard_normal(T).round(4).tolist()},
}

out = Path(__file__).resolve().parent / "example_window.json"
out.write_text(json.dumps(payload))
print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
