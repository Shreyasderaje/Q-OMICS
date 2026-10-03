"""Smoke test for the data layer against the live public archives."""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from qomics.data.cohort import build_cohort, save_manifest

t0 = time.time()
cohort = build_cohort()
print(f"\n=== built in {time.time() - t0:.1f}s ===")

s = cohort.summary()
for k, v in s.items():
    if k == "provenance":
        continue
    print(f"  {k}: {v}")

print("\nprovenance:")
for k, v in s["provenance"].items():
    print(f"  {k}: {v}")

print("\n=== blocks ===")
for name, frame in cohort.blocks.items():
    print(f"\n[{name}] {frame.shape}")
    print(frame.iloc[:3, :6].to_string())
    nan = frame.isna().mean().mean()
    print(f"  mean NaN fraction: {nan:.3f}")

print("\n=== clinical ===")
print(cohort.clinical.head(6).to_string())
print("\nlabel counts:", cohort.class_counts)
print("\n=== feature meta ===")
print(cohort.feature_meta.groupby(["block", "axis"]).size().to_string())

path = save_manifest(cohort)
print("\nmanifest ->", path)
