#!/usr/bin/env python
"""Download and assemble the TCGA-UCEC multi-omics cohort.

    python scripts/fetch_data.py

Results are cached under ``data/``; re-running is cheap and makes no network
calls once the cache is warm.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qomics.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["fetch", *sys.argv[1:]]))
