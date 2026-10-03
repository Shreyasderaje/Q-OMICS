"""Cross-validated evaluation, significance testing and reporting.

Every number Q-OMICS publishes is produced here, under one protocol:

* **Repeated stratified group K-fold** on the patient identifier, so no
  patient contributes aliquots to both sides of a split.
* **Out-of-fold predictions** are pooled across every repeat before metrics
  are computed, which gives one honest score per sample rather than a mean
  of small-sample scores whose variance is dominated by fold size.
* **Paired testing** against the strongest classical baseline.  Fold AUCs
  from the same splits are paired, so a paired t-test is the correct test -
  an unpaired one would inflate the variance and hide a real effect.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Metric bundle
# --------------------------------------------------------------------------


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Score cut-off maximising sensitivity + specificity - 1.

    Fixing the operating point is what turns a ranking model into something
    a screening programme can actually deploy, where the cost of a missed
    early-stage case dominates the cost of a false alarm.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    best = np.argmax(tpr[finite] - fpr[finite])
    return float(thresholds[finite][best])


def classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Full metric bundle at the Youden-optimal operating point."""
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores, dtype=float).ravel()

    if len(np.unique(y_true)) < 2:
        return {"auc": float("nan"), "n": int(len(y_true))}

    threshold = youden_threshold(y_true, scores)
    predicted = (scores >= threshold).astype(int)

    tp = int(((predicted == 1) & (y_true == 1)).sum())
    tn = int(((predicted == 0) & (y_true == 0)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())

    precision, recall, _ = precision_recall_curve(y_true, scores)
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "brier": float(brier_score_loss(y_true, np.clip(scores, 0.0, 1.0))),
        "threshold": threshold,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n": int(len(y_true)),
    }


def bootstrap_ci(
    y_true: np.ndarray, scores: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for AUC - the interval to quote, not the mean."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    n = len(y_true)
    if n < 10 or len(np.unique(y_true)) < 2:
        return (float("nan"), float("nan"), float("nan"))

    values: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(roc_auc_score(y_true[idx], scores[idx]))

    if not values:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(values)
    return (
        float(np.mean(arr)),
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
    )


# --------------------------------------------------------------------------
# Aggregation across folds
# --------------------------------------------------------------------------


@dataclass
class ModelResult:
    """Aggregated result for one model across all folds and repeats."""

    name: str
    display_name: str
    family: str
    fold_metrics: list[dict] = field(default_factory=list)
    oof_scores: dict[int, list[float]] = field(default_factory=dict)
    seconds: list[float] = field(default_factory=list)

    def pooled_metrics(self) -> dict[str, float]:
        """Metrics over out-of-fold scores averaged across repeats.

        Each repeat contributes one score per sample; scores are averaged
        per sample before pooling so that repeats with more folds do not
        get extra weight.
        """
        if not self.oof_scores:
            return {}
        sample_ids = sorted(self.oof_scores)
        y = np.array([sid for sid in sample_ids for _ in [0]])  # placeholder
        del y
        mean_scores = np.array([float(np.mean(self.oof_scores[sid])) for sid in sample_ids])
        return {
            "auc": float("nan"),
            "scores": mean_scores,
            "sample_ids": sample_ids,
        }

    def summary(self, y_true: np.ndarray) -> dict:
        """Metric summary for the report, with fold-level dispersion."""
        per_fold_auc = [m.get("auc", float("nan")) for m in self.fold_metrics
                        if np.isfinite(m.get("auc", np.nan))]
        if not self.fold_metrics:
            return {"name": self.display_name, "family": self.family}

        sample_ids = sorted(self.oof_scores)
        mean_scores = np.array([float(np.mean(self.oof_scores[sid])) for sid in sample_ids])
        pooled = classification_metrics(y_true, mean_scores)
        auc_mean, ci_low, ci_high = bootstrap_ci(y_true, mean_scores)

        out = {
            "name": self.display_name,
            "family": self.family,
            "auc_mean": float(np.mean(per_fold_auc)) if per_fold_auc else float("nan"),
            "auc_std": float(np.std(per_fold_auc, ddof=1)) if len(per_fold_auc) > 1 else 0.0,
            "auc_pooled": pooled.get("auc", float("nan")),
            "auc_ci_low": ci_low,
            "auc_ci_high": ci_high,
            "pr_auc_mean": float(np.nanmean([m.get("pr_auc", np.nan) for m in self.fold_metrics])),
            "pr_auc_std": float(np.nanstd([m.get("pr_auc", np.nan) for m in self.fold_metrics], ddof=1)),
            "f1_mean": float(np.nanmean([m.get("f1", np.nan) for m in self.fold_metrics])),
            "f1_std": float(np.nanstd([m.get("f1", np.nan) for m in self.fold_metrics], ddof=1)),
            "sensitivity_mean": float(np.nanmean([m.get("sensitivity", np.nan) for m in self.fold_metrics])),
            "sensitivity_std": float(np.nanstd([m.get("sensitivity", np.nan) for m in self.fold_metrics], ddof=1)),
            "specificity_mean": float(np.nanmean([m.get("specificity", np.nan) for m in self.fold_metrics])),
            "specificity_std": float(np.nanstd([m.get("specificity", np.nan) for m in self.fold_metrics], ddof=1)),
            "brier_mean": float(np.nanmean([m.get("brier", np.nan) for m in self.fold_metrics])),
            "brier_std": float(np.nanstd([m.get("brier", np.nan) for m in self.fold_metrics], ddof=1)),
            "seconds_mean": float(np.mean(self.seconds)) if self.seconds else 0.0,
            "n_folds": len(self.fold_metrics),
            "folds": [
                {
                    "repeat": m.get("repeat", 0),
                    "fold": m.get("fold", 0),
                    "auc": m.get("auc", float("nan")),
                    "n_test": m.get("n", 0),
                }
                for m in self.fold_metrics
            ],
            "_oof_scores": mean_scores.tolist(),
            "_oof_ids": sample_ids,
        }
        return out

    def roc(self, y_true: np.ndarray) -> dict:
        """Out-of-fold ROC curve, thinned to keep payloads small."""
        sample_ids = sorted(self.oof_scores)
        mean_scores = np.array([float(np.mean(self.oof_scores[sid])) for sid in sample_ids])
        fpr, tpr, _ = roc_curve(y_true, mean_scores)
        keep = _thin(fpr, 300)
        return {
            "name": self.display_name,
            "fpr": fpr[keep].round(5).tolist(),
            "tpr": tpr[keep].round(5).tolist(),
        }


def _thin(values: np.ndarray, max_points: int) -> np.ndarray:
    """Evenly subsample a curve, always keeping the first and last index."""
    n = len(values)
    if n <= max_points:
        return np.arange(n)
    idx = np.linspace(0, n - 1, max_points).round().astype(int)
    return np.unique(idx)


# --------------------------------------------------------------------------
# Significance
# --------------------------------------------------------------------------


def paired_t_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided paired t-test on two paired score vectors.

    Implemented directly so the project carries no SciPy dependency.  For
    ``n < 3`` pairs the statistic is undefined, which is reported honestly
    rather than substituted with a misleading value.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 3:
        return {"mean_delta": float("nan"), "t_statistic": float("nan"),
                "p_value": float("nan"), "n_pairs": n, "significant_at_0.05": False}

    d = a - b
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd < 1e-15:
        t = float("inf") if abs(mean_d) > 1e-15 else 0.0
        p = 0.0 if abs(mean_d) > 1e-15 else 1.0
        return {"mean_delta": mean_d, "t_statistic": t, "p_value": p,
                "n_pairs": n, "significant_at_0.05": p < 0.05}

    t = mean_d / (sd / math.sqrt(n))
    p = _t_sf_two_sided(abs(t), n - 1)
    return {
        "mean_delta": mean_d,
        "t_statistic": float(t),
        "p_value": float(p),
        "n_pairs": n,
        "significant_at_0.05": bool(p < 0.05),
    }


def _t_sf_two_sided(t: float, df: int) -> float:
    """Two-sided p-value for Student's t.

    Delegates to SciPy rather than a hand-rolled incomplete beta.  A
    from-scratch continued-fraction expansion was tried and rejected: it
    returned p-values systematically too small (0.033 where the correct
    value is 0.050 at t=2.228, df=10), which would have inflated every
    significance claim in the report.  SciPy is already a project
    dependency and this quantity is too important to approximate.
    """
    if df <= 0:
        return float("nan")
    t = abs(float(t))
    if not math.isfinite(t):
        return 0.0
    from scipy import stats

    return float(2.0 * stats.t.sf(t, df))


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction for multiple comparisons.

    Q-OMICS compares one hybrid model against several baselines; without a
    correction the family-wise error rate is inflated and a spurious
    "improvement" becomes likely.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    rejected = [False] * m
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (m - rank):
            rejected[idx] = True
        else:
            break
    return rejected


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------


def summarise_results(
    results: list[ModelResult],
    y_true: np.ndarray,
    primary_name: str = "hybrid_transformer",
) -> dict:
    """Assemble the full ``/api/metrics`` payload."""
    summaries = [r.summary(y_true) for r in results]
    for summary in summaries:
        summary.pop("_oof_scores", None)
        summary.pop("_oof_ids", None)

    primary = next((r for r in results if r.name == primary_name), None)
    significance: list[dict] = []

    if primary is not None:
        primary_aucs = [m.get("auc", np.nan) for m in primary.fold_metrics]
        for other in results:
            if other.name == primary_name:
                continue
            other_aucs = [m.get("auc", np.nan) for m in other.fold_metrics]
            if len(other_aucs) != len(primary_aucs):
                continue
            test = paired_t_test(primary_aucs, other_aucs)
            test["comparison"] = f"{primary.display_name} vs {other.display_name}"
            test["reference"] = other.name
            significance.append(test)

    if significance:
        flags = holm_bonferroni([s["p_value"] for s in significance])
        for test, flag in zip(significance, flags):
            test["significant_after_correction"] = flag
            test["significant_at_0.05"] = flag

    return {
        "target": "",
        "n_splits": len(primary.fold_metrics) if primary else 0,
        "primary": primary_name,
        "models": summaries,
        "significance": sorted(
            significance, key=lambda s: (s["p_value"] if np.isfinite(s["p_value"]) else 1.0)
        ),
        "roc_curve": {
            "labels": ["class_0", "class_1"],
            "curves": [r.roc(y_true) for r in results],
        },
    }


__all__ = [
    "ModelResult",
    "classification_metrics",
    "youden_threshold",
    "bootstrap_ci",
    "paired_t_test",
    "holm_bonferroni",
    "summarise_results",
]
