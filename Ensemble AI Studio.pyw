"""Ensemble AI Studio — double-click launcher for Windows (.pyw runs without a console window)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aichatlab.ui import main  # noqa: E402

if __name__ == "__main__":
    main()
