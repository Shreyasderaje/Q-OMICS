#!/usr/bin/env python
"""Start the Q-OMICS API and web interface.

    python scripts/serve.py
    python scripts/serve.py --port 9000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qomics.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["serve", *sys.argv[1:]]))
