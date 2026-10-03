#!/usr/bin/env python
"""Train Q-OMICS end to end.

    python scripts/train.py                            # full protocol (5 splits x 2 repeats)
    python scripts/train.py --repeats 1 --splits 5     # quick pass
    python scripts/train.py --skip-transformer          # baselines + quantum only

The cohort is downloaded and assembled on first run; the response cache in
``data/cache/`` makes subsequent runs fast. No fetch step is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qomics.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
