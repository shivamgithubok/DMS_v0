"""Re-run an entry-point script inside the project's .venv when it was started with another
interpreter (e.g. `python3 main.py` after setup_jetson.sh). Must be imported before any
third-party module. Set DMS_NO_VENV=1 to disable."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure_venv() -> None:
    venv = ROOT / ".venv"
    python = venv / "bin" / "python"
    if os.environ.get("DMS_NO_VENV") or not python.exists():
        return
    if Path(sys.prefix).resolve() == venv.resolve():
        return
    os.environ["DMS_NO_VENV"] = "1"  # never loop
    os.execv(str(python), [str(python)] + sys.argv)
