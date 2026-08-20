"""Put the repo root and tests/ on sys.path so the flat layout imports cleanly."""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
for _p in (_ROOT, _ROOT / "tests", _ROOT / "experiments", _ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
