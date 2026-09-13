#!/usr/bin/env python3
"""Back-compat shim: the runner now lives in musecpeval/runner.py.

Prefer the installed console script (``musecpeval``) or ``python -m musecpeval``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from musecpeval.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
