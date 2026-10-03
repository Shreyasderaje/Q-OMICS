"""Render the publication figures from a completed run.

    python scripts/make_figures.py

Reads ``artifacts/run.json`` and writes PNGs to ``artifacts/figures/``.  The
style is deliberately plain: these are analysis figures for a lab notebook,
not marketing assets.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from qomics.config import ARTIFACT_DIR  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)

ACCENT = "#1f6f78"
MUTED = "#9aa3ad"
GREY = "#4d545c"
PALETTE = ["#1f6f78", "#c1666b", "#d9a441", "#5b7fa6", "#7d6b9c", "#6f8f6f", "#b0763f"]

FIG_DIR = ARTIFACT_DIR / "figures"


def load() -> dict:
    path = ARTIFACT_DIR / "run.json"
    if not path.exists():
        raise SystemExit(f"{path} not found; run `python scripts/train.py` first")
    return json.loads(path.read_text(encoding="utf-8"))


def figure_roc(run: dict) -> Path:
    curves = run.get("metrics", {}).get("roc_curve", {}).get("curves", [])
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.0, ls="--", label="chance", zorder=1)

    models = {m["name"]: m for m in run.get("metrics", {}).get("models", [])}
    for i, curve in enumerate(curves):
        name = curve.get("name", f"model {i}")
        primary = name == "Hybrid Transformer (Q-OMICS)"
        stats = models.get(name, {})
        label = f"{name}  AUC {stats.get('auc_mean', float('nan')):.3f}" if stats else name
        ax.plot(
            curve.get("fpr", []), curve.get("tpr", []),
            lw=2.2 if primary else 1.1,
            color=ACCENT if primary else PALETTE[i % len(PALETTE)],
            alpha=1.0 if primary else 0.75,
            label=label, zorder=3 if primary else 2,
        )

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Cross-validated ROC\nFIGO stage I vs stage II-IV", loc="left", fontsize=10)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    # The legend goes outside the axes: inside it covers the part of the ROC
    # that distinguishes the models, which is the only reason to draw it.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7,
              frameon=False, handlelength=1.6)
    fig.tight_layout()
    out = FIG_DIR / "roc_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_metric_bars(run: dict) -> Path:
    models = sorted(
        run.get("metrics", {}).get("models", []),
        key=lambda m: m.get("auc_mean") or 0,
    )
    if not models:
        raise SystemExit("no models in run.json")
    names = [m["name"] for m in models]
    aucs = [m.get("auc_mean") or 0.0 for m in models]
    errs = [m.get("auc_std") or 0.0 for m in models]
    colours = [ACCENT if n == "Hybrid Transformer (Q-OMICS)" else GREY for n in names]

    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(models) + 1.4))
    positions = np.arange(len(names))
    ax.barh(positions, aucs, xerr=errs, color=colours, height=0.62,
            error_kw={"ecolor": "#33383e", "lw": 1.0, "capsize": 2.5})
    ax.axvline(0.5, color=MUTED, lw=1.0, ls="--")
    for pos, value in zip(positions, aucs):
        ax.text(value + 0.012, pos, f"{value:.3f}", va="center", fontsize=8)
    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean ROC-AUC over patient-disjoint cross-validation folds")
    ax.set_xlim(0, 1.0)
    ax.set_title("Model comparison", loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    out = FIG_DIR / "model_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_modality(run: dict) -> Path:
    contribution = run.get("biomarker_summary", {}).get("modality_contribution", [])
    if not contribution:
        raise SystemExit("no modality contribution in run.json")
    labels = [c["block"] for c in contribution]
    values = [c["importance"] for c in contribution]

    fig, ax = plt.subplots(figsize=(5.0, 0.5 * len(labels) + 1.3))
    positions = np.arange(len(labels))
    ax.barh(positions, values, color=[PALETTE[i % len(PALETTE)] for i in range(len(labels))],
            height=0.55)
    for pos, value in zip(positions, values):
        ax.text(value + 0.006, pos, f"{value * 100:.1f}%", va="center", fontsize=8)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Share of pooled attention importance")
    ax.set_xlim(0, max(values) * 1.25 if values else 1)
    ax.set_title("Modality contribution to the ranked panel", loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    out = FIG_DIR / "modality_contribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_kernel(run: dict) -> Path:
    separation = run.get("quantum", {}).get("labelled_separation", {})
    spectrum = run.get("quantum", {}).get("quantum_spectrum", {})

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    ax = axes[0]
    ratios = spectrum.get("explained_variance_ratio", [])[:10]
    if ratios:
        ax.bar(np.arange(1, len(ratios) + 1), ratios, color=ACCENT, width=0.62)
        cumulative = spectrum.get("cumulative", [])[:10]
        ax2 = ax.twinx()
        ax2.plot(np.arange(1, len(cumulative) + 1), cumulative, color="#c1666b", lw=1.4)
        ax2.set_ylabel("Cumulative", color="#c1666b", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#c1666b", labelsize=7)
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)
    ax.set_xlabel("Principal kernel component")
    ax.set_ylabel("Explained variance")
    ax.set_title("Quantum kernel spectrum", loc="left", fontsize=9)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    names = [k for k in separation if k in {"quantum", "rbf", "cosine", "linear"}]
    values = [separation[k] for k in names]
    colours = [ACCENT if n == "quantum" else GREY for n in names]
    ax.barh(np.arange(len(names)), values, color=colours, height=0.5)
    ax.axvline(0, color=MUTED, lw=1.0)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Between-class minus within-class mean similarity")
    ax.set_title("Labelled separation by kernel", loc="left", fontsize=9)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    out = FIG_DIR / "kernel_analysis.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_panel() -> Path | None:
    import pandas as pd

    path = ARTIFACT_DIR / "biomarkers.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path).head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(6.6, 0.34 * len(table) + 1.3))
    positions = np.arange(len(table))
    colours = [PALETTE[i % len(PALETTE)] for i in table["block"].astype(str).map(
        {"transcriptome": 0, "genome": 1, "proteome": 2}
    ).fillna(3).astype(int)]
    ax.barh(positions, table["importance"], color=colours, height=0.6)
    for pos, row in zip(positions, table.itertuples()):
        marker = "*" if bool(getattr(row, "fdr_significant", False)) else " "
        ax.text(row.importance + table["importance"].max() * 0.02, pos,
                f"{row.symbol}  d={row.effect_size:+.2f}{marker}", va="center", fontsize=7)
    ax.set_yticks(positions)
    ax.set_yticklabels(table["feature"], fontsize=7)
    ax.set_xlabel("Attention-rollout importance")
    ax.set_xlim(0, table["importance"].max() * 1.7)
    ax.set_title("Top 20 ranked biomarkers\n* = FDR-significant (Benjamini-Hochberg q < 0.05)",
                 loc="left", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    out = FIG_DIR / "biomarker_panel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    run = load()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written = [
        figure_roc(run),
        figure_metric_bars(run),
        figure_modality(run),
        figure_kernel(run),
    ]
    panel = figure_panel()
    if panel:
        written.append(panel)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
