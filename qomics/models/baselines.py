"""Classical reference models.

A hybrid quantum-classical claim is only meaningful against strong classical
baselines.  Q-OMICS therefore trains a full ladder of references on exactly
the same folds and features:

``logistic``
    L2 logistic regression on standardised features.  The linear floor.
``rbf_svm``
    RBF-kernel SVM.  This is the model a fidelity quantum kernel is *meant*
    to beat, since swapping the Gram matrix is the only difference.
``random_forest``
    Bagged trees; strong on tabular omics, immune to scaling.
``hist_gb``
    Histogram gradient boosting; the usual tabular accuracy leader.
``pca_logreg_transcriptome``
    Expression-only reference, which is what most published UCEC stage
    models use.  It quantifies how much the genome and proteome actually add.

If the quantum kernel cannot separate itself from the RBF SVM, the honest
conclusion is that it adds nothing - and Q-OMICS reports that rather than
hiding it.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..config import BaselineConfig

log = logging.getLogger(__name__)

#: Display names are part of the public API contract.
DISPLAY_NAMES: dict[str, str] = {
    "logreg": "Logistic Regression",
    "rbf_svm": "RBF SVM",
    "random_forest": "Random Forest",
    "hist_gb": "HistGradientBoosting",
    "pca_logreg_tx": "PCA + Logistic Regression (transcriptome only)",
    "quantum_svm": "Quantum SVM",
    "classical_transformer": "Classical Transformer",
    "hybrid_transformer": "Hybrid Transformer (Q-OMICS)",
}

FAMILY: dict[str, str] = {
    "logreg": "linear",
    "rbf_svm": "kernel",
    "random_forest": "ensemble",
    "hist_gb": "ensemble",
    "pca_logreg_tx": "linear",
    "quantum_svm": "kernel",
    "classical_transformer": "neural",
    "hybrid_transformer": "hybrid",
}


@dataclass
class FittedBaseline:
    """A fitted baseline plus the metadata needed to report it."""

    name: str
    estimator: object
    seconds: float
    n_features: int

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)  # type: ignore[union-attr]
        scores = self.estimator.decision_function(X)  # type: ignore[union-attr]
        scores = np.asarray(scores, dtype=float).ravel()
        return np.column_stack([1.0 - scores, scores])


def _suppress() -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def fit_baseline(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    config: BaselineConfig | None = None,
) -> FittedBaseline:
    """Fit one reference model and return it with its training cost.

    Probabilities for the margin-based models come from a Platt-style
    calibration fitted on the *training* scores, never on test data.
    """
    cfg = config or BaselineConfig()
    _suppress()
    start = time.perf_counter()

    if name == "logreg":
        estimator: object = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(
                    C=cfg.logreg_c, max_iter=5000, class_weight="balanced",
                    random_state=cfg.random_state,
                )),
            ]
        )
    elif name == "rbf_svm":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", SVC(
                    C=1.0, kernel="rbf", gamma=cfg.rbf_svm_gamma,
                    probability=True, class_weight="balanced",
                    random_state=cfg.random_state,
                )),
            ]
        )
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=cfg.rf_trees, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=cfg.random_state,
        )
    elif name == "hist_gb":
        estimator = HistGradientBoostingClassifier(
            learning_rate=cfg.hgb_learning_rate, max_iter=cfg.hgb_max_iter,
            max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.15, random_state=cfg.random_state,
        )
    else:
        raise ValueError(f"unknown baseline: {name!r}")

    estimator.fit(X_train, y_train)  # type: ignore[attr-defined]
    return FittedBaseline(
        name=name,
        estimator=estimator,
        seconds=time.perf_counter() - start,
        n_features=int(X_train.shape[1]),
    )


def fit_quantum_svm(
    K_train: np.ndarray,
    y_train: np.ndarray,
    c: float = 1.0,
    class_weight: str | dict | None = "balanced",
    random_state: int = 20240517,
) -> FittedBaseline:
    """SVM on a precomputed quantum Gram matrix.

    ``class_weight="balanced"`` matters here: the SVM is otherwise free to
    discard every support vector from the minority class once the kernel
    produces a wide margin.
    """
    _suppress()
    start = time.perf_counter()
    estimator = SVC(
        C=c, kernel="precomputed", class_weight=class_weight,
        decision_function_shape="ovr", random_state=random_state,
    )
    estimator.fit(K_train, y_train)
    return FittedBaseline(
        name="quantum_svm",
        estimator=estimator,
        seconds=time.perf_counter() - start,
        n_features=int(K_train.shape[0]),
    )


def baseline_registry(
    config: BaselineConfig | None = None,
) -> dict[str, Callable[..., FittedBaseline]]:
    """Names of the dense-feature baselines evaluated in every fold."""
    return {
        "logreg": lambda Xtr, ytr, Xte: fit_baseline("logreg", Xtr, ytr, Xte, config),
        "rbf_svm": lambda Xtr, ytr, Xte: fit_baseline("rbf_svm", Xtr, ytr, Xte, config),
        "random_forest": lambda Xtr, ytr, Xte: fit_baseline("random_forest", Xtr, ytr, Xte, config),
        "hist_gb": lambda Xtr, ytr, Xte: fit_baseline("hist_gb", Xtr, ytr, Xte, config),
    }


__all__ = [
    "DISPLAY_NAMES",
    "FAMILY",
    "FittedBaseline",
    "fit_baseline",
    "fit_quantum_svm",
    "baseline_registry",
]
