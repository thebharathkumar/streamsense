"""Top-level eval entry. Delegates to streamsense.eval.cli."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamsense.eval.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
