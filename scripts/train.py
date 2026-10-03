#!/usr/bin/env python
"""Train Q-OMICS end to end.

    python scripts/train.py                       # full protocol (15 folds)
    python scripts/train.py --repeats 1 --splits 5   # quick pass
    python scripts/train.py --skip-transformer      # baselines only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qomics.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
