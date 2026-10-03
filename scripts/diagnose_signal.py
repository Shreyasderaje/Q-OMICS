"""Diagnostic: how much stage signal is actually in each layer?

Answers the question that determines whether the model can work at all,
before any model is trained.  Reports per-layer univariate AUC for the
FIGO stage-I vs stage-II-IV target, plus the best achievable single-marker
AUC.  A layer whose univariate AUC sits at 0.5 cannot rescue a model no
matter how the model is built.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qomics.data.cohort import build_cohort
from qomics.pipeline.preprocess import MultiOmicsPreprocessor

cohort = build_cohort()
y = cohort.y
print(f"cohort {cohort.n_samples} samples | stage I {(y == 0).sum()} vs stage II-IV {(y == 1).sum()}")
print(f"blocks: {cohort.n_features}\n")

pre = MultiOmicsPreprocessor(
    cohort.blocks,
    max_features_per_block={"transcriptome": cohort.blocks["transcriptome"].shape[1],
                            "genome": cohort.blocks["genome"].shape[1],
                            "proteome": cohort.blocks["proteome"].shape[1]},
    fused_components={"transcriptome": 6, "genome": 4, "proteome": 2},
)
dense, fused = pre.fit_transform(None)

print(f"{'block':<15} {'n':>5} {'median AUC':>11} {'mean AUC':>9} {'>0.65':>6} {'>0.70':>6}  top markers")
print("-" * 112)

summary = {}
for block, matrix in dense.items():
    names = pre.selected[block]
    aucs = []
    for j, column in enumerate(names):
        x = matrix[:, j]
        if np.nanstd(x) < 1e-12:
            aucs.append(0.5)
            continue
        aucs.append(roc_auc_score(y, x))
    aucs = np.asarray(aucs)
    order = np.argsort(-np.abs(aucs - 0.5))[:6]
    top = ", ".join(f"{names[i]}({aucs[i]:.3f})" for i in order)
    frac65 = float((aucs > 0.65).mean() + (aucs < 0.35).mean())
    frac70 = float((aucs > 0.70).mean() + (aucs < 0.30).mean())
    print(f"{block:<15} {len(names):>5} {np.median(aucs):>11.4f} {aucs.mean():>9.4f} "
          f"{frac65:>6.2f} {frac70:>6.2f}  {top}")
    summary[block] = {
        "n": len(names), "median_auc": float(np.median(aucs)),
        "best_auc": float(np.max(np.maximum(aucs, 1 - aucs))),
        "frac_above_0.65": frac65,
    }

print("\nBest single marker per layer (|AUC - 0.5| maximised):")
for block, stats in summary.items():
    print(f"  {block:<15} best |AUC-0.5| -> AUC {stats['best_auc']:.4f} "
          f"({stats['frac_above_0.65'] * 100:.1f}% of markers exceed 0.65)")

print("\nInterpretation:")
for block, stats in summary.items():
    if stats["best_auc"] > 0.75 and stats["frac_above_0.65"] > 0.05:
        print(f"  {block}: strong univariate signal - worth encoding.")
    elif stats["best_auc"] > 0.65:
        print(f"  {block}: moderate signal - useful as a supplement.")
    else:
        print(f"  {block}: weak signal on its own - needs the other layers or adds noise.")
