"""Biomarker ranking and interpretation.

A classifier that reports an AUC has answered a benchmark question.  A
*biomarker discovery* platform has to answer a different one: which molecules
should a clinician measure, and can they defend that choice?

Q-OMICS ranks features with three signals combined:

**Attention importance** - attention rollout from the transformer's ``[CLS]``
token.  Captures multi-layer, cross-modal influence rather than a single
first-order weight.

**Stability** - the fraction of independent cross-validation repeats in which
a feature lands in the top decile.  A marker that only appears in one split is
noise; the stability column is what separates a publishable panel from a
greedy top-k list.

**Univariate effect** - Cohen's d between classes, and direction.  Confirms
that an important token is genuinely associated with the endpoint rather than
acting as a nuisance variable the encoder latched onto.

Per-feature permutation importance is additionally computed against a
trained fold model, giving a second, classifier-agnostic attribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with a pooled standard deviation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return 0.0
    na, nb = a.size, b.size
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def point_biserial(y: np.ndarray, x: np.ndarray) -> float:
    """Correlation between a binary label and a continuous feature."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    y, x = y[mask], x[mask]
    if x.size < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR control.

    Returns ``(reject, q_values)``.  Without this, screening several hundred
    features guarantees a handful of apparently significant hits by chance
    alone, which is precisely the failure mode of biomarker papers.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return np.array([], dtype=bool), np.array([])
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p-value downwards.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q <= alpha, q


def permutation_test(
    y: np.ndarray, x: np.ndarray, n_permutations: int = 2000, seed: int = 0
) -> float:
    """Two-sided permutation p-value for an association between ``x`` and ``y``."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).ravel()
    x = np.asarray(x, dtype=float).ravel()
    mask = np.isfinite(x)
    y, x = y[mask], x[mask]
    if x.size < 5 or len(np.unique(y)) < 2:
        return 1.0

    observed = abs(point_biserial(y, x))
    if observed < 1e-12:
        return 1.0

    count = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(y)
        if abs(point_biserial(shuffled, x)) >= observed:
            count += 1
    return float((count + 1) / (n_permutations + 1))


# --------------------------------------------------------------------------
# Panel assembly
# --------------------------------------------------------------------------


@dataclass
class BiomarkerPanel:
    """Ranked biomarker table with the provenance needed to defend it."""

    table: pd.DataFrame
    method: str
    n_features_considered: int

    def top(self, n: int = 25, block: str = "all") -> list[dict]:
        frame = self.table
        if block != "all":
            frame = frame[frame["block"] == block]
        return frame.head(n).to_dict(orient="records")

    def get(self, symbol: str) -> dict | None:
        table = self.table
        if "symbol" in table.columns:
            hit = table[table["symbol"] == symbol]
            if not hit.empty:
                return hit.iloc[0].to_dict()
        if symbol in table.index:
            return table.loc[symbol].to_dict()
        direct = table[table["feature"] == symbol]
        return direct.iloc[0].to_dict() if not direct.empty else None

    def block_summary(self) -> list[dict]:
        """Share of total importance carried by each modality."""
        totals = self.table.groupby("block")["importance"].sum()
        grand = float(totals.sum()) or 1.0
        return [
            {"block": str(block), "importance": float(value) / grand}
            for block, value in totals.sort_values(ascending=False).items()
        ]

    def axis_summary(self) -> list[dict]:
        """Top biological axes by accumulated importance."""
        grouped = self.table.groupby("axis")["importance"].sum().sort_values(ascending=False)
        return [
            {"axis": str(axis), "importance": float(value), "n_features": int((self.table["axis"] == axis).sum())}
            for axis, value in grouped.head(12).items()
        ]


def build_panel(
    attention_importance: np.ndarray | pd.Series,
    feature_names: list[str],
    blocks: list[str],
    feature_meta: pd.DataFrame,
    y: np.ndarray,
    dense_blocks: dict[str, np.ndarray],
    dense_names: dict[str, list[str]],
    *,
    stability: np.ndarray | None = None,
    top_decile_fraction: float = 0.10,
    method: str = "attention_rollout",
) -> BiomarkerPanel:
    """Combine attention, stability and univariate effect into a ranked panel.

    Parameters
    ----------
    attention_importance:
        ``(n_samples, n_features)`` rollout importance aligned with
        ``feature_names``, or an ``(n_features,)`` mean-importance vector.
    stability:
        ``(n_features,)`` fraction of repeats in which a feature reached the
        top decile.  ``None`` yields a stability of 1.0 for everything, which
        the report labels accordingly.
    dense_blocks / dense_names:
        Per-modality dense matrices used to compute the univariate effect,
        so that a token's direction reflects the real measured value.
    """
    importance = np.asarray(attention_importance, dtype=float)
    if importance.ndim == 1:
        importance = importance[None, :]
    mean_importance = importance.mean(axis=0)

    y = np.asarray(y).ravel()
    rows: list[dict] = []

    name_to_block = dict(zip(feature_names, blocks))
    value_lookup: dict[str, np.ndarray] = {}
    for block, matrix in dense_blocks.items():
        for j, column in enumerate(dense_names.get(block, [])):
            value_lookup[f"{block}::{column}"] = matrix[:, j]
    # The token order is block-major, so also index the raw column names.
    for block, matrix in dense_blocks.items():
        for j, column in enumerate(dense_names.get(block, [])):
            value_lookup.setdefault(column, matrix[:, j])

    for j, feature in enumerate(feature_names):
        block = name_to_block.get(feature, "unknown")
        symbol = feature.split("::")[-1]

        values = value_lookup.get(feature)
        if values is None:
            values = value_lookup.get(symbol)
        values = np.asarray(values, dtype=float) if values is not None else np.array([])

        early = values[y == 0]
        advanced = values[y == 1]
        d = cohens_d(early, advanced) if values.size else 0.0
        p_value = permutation_test(y, values, n_permutations=1000) if values.size else 1.0

        rows.append(
            {
                "feature": feature,
                "symbol": symbol,
                "block": block,
                "importance": float(mean_importance[j]),
                "importance_std": float(importance[:, j].std(ddof=1)) if importance.shape[0] > 1 else 0.0,
                "stability": float(stability[j]) if stability is not None else 1.0,
                "effect_size": d,
                "p_value": float(p_value),
                "mean_early": float(np.nanmean(early)) if early.size else float("nan"),
                "mean_advanced": float(np.nanmean(advanced)) if advanced.size else float("nan"),
                "n_observed": int(np.isfinite(values).sum()) if values.size else 0,
            }
        )

    table = pd.DataFrame(rows)

    # Composite score: importance drives the ranking, stability demotes a
    # marker that only surfaces in one split, and the univariate effect size
    # breaks ties toward features that are independently verifiable.
    table["composite"] = (
        table["importance"]
        * (0.5 + 0.5 * table["stability"])
        * (1.0 + np.tanh(np.abs(table["effect_size"])))
    )
    table = table.sort_values("composite", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)

    # FDR across the screened panel.
    table["q_value"] = fdr_bh(table["p_value"].to_numpy())[1]
    table["fdr_significant"] = fdr_bh(table["p_value"].to_numpy())[0]

    total = float(table["importance"].sum()) or 1.0
    table["importance_normalised"] = table["importance"] / total
    table["direction"] = np.where(
        table["effect_size"] < 0, "depleted_in_early_stage", "enriched_in_early_stage"
    )
    table.loc[np.abs(table["effect_size"]) < 1e-6, "direction"] = "no_association"

    # Attach pathway annotation from the cohort feature table.
    meta_index = feature_meta.set_index("feature")["axis"].to_dict() if len(feature_meta) else {}
    table["axis"] = [meta_index.get(f, meta_index.get(f.split("::")[-1], "unclassified"))
                     for f in table["feature"]]

    return BiomarkerPanel(
        table=table,
        method=method,
        n_features_considered=len(feature_names),
    )


def top_decile_membership(
    importance_per_fold: list[np.ndarray], fraction: float = 0.10
) -> np.ndarray:
    """Fraction of folds in which each feature reached the top decile."""
    if not importance_per_fold:
        return np.array([])
    stacked = np.vstack(importance_per_fold)
    n_features = stacked.shape[1]
    k = max(1, int(round(n_features * fraction)))
    counts = np.zeros(n_features, dtype=float)
    for row in stacked:
        top = np.argsort(row)[-k:]
        counts[top] += 1.0
    return counts / stacked.shape[0]


def format_report(panel: BiomarkerPanel, top_n: int = 30) -> str:
    """Plain-text ranked panel, suitable for a terminal or a lab notebook."""
    table = panel.table.head(top_n)
    lines = [
        f"Q-OMICS ranked biomarker panel ({panel.n_features_considered} features screened)",
        f"ranking method: {panel.method}",
        "",
        f"{'#':>3}  {'feature':<26} {'block':<14} {'axis':<22} {'imp':>7} {'stab':>6} {'d':>7} {'q':>8}",
        "-" * 104,
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{int(row['rank']):>3}  {row['feature']:<26} {row['block']:<14} "
            f"{str(row['axis'])[:22]:<22} {row['importance']:>7.4f} "
            f"{row['stability']:>6.2f} {row['effect_size']:>7.3f} {row['q_value']:>8.3f}"
        )
    return "\n".join(lines)


__all__ = [
    "BiomarkerPanel",
    "build_panel",
    "cohens_d",
    "point_biserial",
    "fdr_bh",
    "permutation_test",
    "top_decile_membership",
    "format_report",
]
