"""Top-level training entry. Delegates to streamsense.train.cli."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamsense.train.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
