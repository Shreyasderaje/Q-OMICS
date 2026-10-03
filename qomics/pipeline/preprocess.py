"""Fold-safe preprocessing of multi-omics blocks.

Two distinct representations are produced from the same cohort:

``dense``
    One sample x feature matrix per modality, variance-filtered and
    standardised.  This is the token sequence consumed by the transformer
    and the source of every per-gene importance score the report emits.

``fused``
    A single compact vector per sample obtained by concatenating per-modality
    principal components, each weighted so that no modality dominates the
    Euclidean geometry.  This is what the quantum register encodes, because
    a six-qubit circuit cannot accept 1 300 features.

Every transform is fitted on the training split alone.  Fitting a scaler or
a PCA on the full cohort would let test-fold variance leak into training and
would inflate every reported score, so the estimator is explicitly scoped to
an index set and re-fitted per fold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Modality block selection
# --------------------------------------------------------------------------


def select_features(
    frame: pd.DataFrame,
    keep: int,
    *,
    min_variance: float = 1e-8,
) -> pd.DataFrame:
    """Keep the ``keep`` highest-variance columns of a modality block.

    Variance ranking is label-free, so it can be computed on the training
    split without contaminating the estimate.  Missing values are imputed
    with the column median *before* ranking so that a sparse but genuinely
    variable marker is not discarded.
    """
    if frame.shape[1] <= keep:
        return frame
    filled = frame.fillna(frame.median(numeric_only=True)).fillna(0.0)
    variances = filled.var(axis=0, ddof=1).fillna(0.0)
    eligible = variances[variances > min_variance]
    if eligible.empty:
        return frame.iloc[:, :keep]
    order = eligible.sort_values(ascending=False).index[:keep]
    return frame[order]


# --------------------------------------------------------------------------
# Per-modality reduction
# --------------------------------------------------------------------------


@dataclass
class ModalityReducer:
    """Standardise then PCA a single modality block.

    Parameters
    ----------
    name:
        Modality label, used for logging and weight lookup.
    n_components:
        Number of principal components to retain.  Clamped to the rank
        actually available in the training split.
    weight:
        Post-PCA scale applied to every component of this modality, which
        controls the modality's contribution to the fused Euclidean geometry.
    """

    name: str
    n_components: int
    weight: float = 1.0

    scaler: StandardScaler | None = field(default=None, repr=False)
    pca: PCA | None = field(default=None, repr=False)
    n_seen_: int = 0

    def fit(self, frame: pd.DataFrame) -> ModalityReducer:
        dense = self._impute(frame)
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(dense)
        n_available = min(self.n_components, scaled.shape[0] - 1, scaled.shape[1])
        n_available = max(1, n_available)
        self.pca = PCA(n_components=n_available, svd_solver="full", random_state=0)
        self.pca.fit(scaled)
        self.n_seen_ = frame.shape[0]
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.scaler is None or self.pca is None:
            raise RuntimeError(f"ModalityReducer({self.name}) is not fitted")
        scaled = self.scaler.transform(self._impute(frame))
        return self.pca.transform(scaled) * self.weight

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    @staticmethod
    def _impute(frame: pd.DataFrame) -> np.ndarray:
        # ``copy=True`` matters on pandas 3.x, where ``to_numpy`` may hand
        # back a read-only view of the block and the median fill below would
        # otherwise raise.
        dense = frame.to_numpy(dtype=float, copy=True)
        if np.isnan(dense).any():
            col_median = np.nanmedian(dense, axis=0)
            col_median = np.where(np.isfinite(col_median), col_median, 0.0)
            inds = np.where(np.isnan(dense))
            dense[inds] = np.take(col_median, inds[1])
        return dense
# --------------------------------------------------------------------------
# Whole-cohort preprocessor
# --------------------------------------------------------------------------

#: Default number of principal components per modality for the fused block.
DEFAULT_FUSED_COMPONENTS: dict[str, int] = {
    "transcriptome": 6,
    "genome": 4,
    "proteome": 2,
}


@dataclass
class TokenScaler:
    """Per-modality standardisation for the token matrices.

    The three blocks arrive on incompatible scales: transcriptomic values are
    cohort z-scores centred on zero, the genomic block is mostly 0/1 mutation
    indicators with rare ±2 copy-number calls, and antibody abundances are
    unit-variance.  Without rescaling, a somatic mutation column with 10 %
    prevalence produces a token that is nearly identical across samples and
    is drowned out by the expression channels - the model then has to spend
    capacity learning to amplify it, and usually does not.

    The genome block is also *sparse* in a way that makes z-scoring unstable,
    so columns with negligible variance are centred by their mean and left at
    unit scale rather than divided by a near-zero standard deviation.
    """

    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray, min_variance: float = 1e-8) -> TokenScaler:
        matrix = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0)
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(matrix.shape[1])
        scale = np.where(np.isfinite(scale) & (scale > min_variance), scale, 1.0)
        return cls(mean_=mean, scale_=scale)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0)
        return (matrix - self.mean_) / self.scale_


class MultiOmicsPreprocessor:
    """Assemble the dense token blocks and the fused quantum input.

    Parameters
    ----------
    max_features_per_block:
        Hard cap on tokens per modality.  A 700-token sequence is already
        long for a 3-layer encoder, and beyond this point the extra tokens
        cost quadratic attention without adding signal.
    fused_components:
        Principal components per modality inside the fused vector.  The
        values must sum to at most ``quantum.n_input_features`` because the
        register width bounds the input dimension.
    weights:
        Per-modality post-PCA scale.
    """

    def __init__(
        self,
        blocks: dict[str, pd.DataFrame],
        *,
        max_features_per_block: dict[str, int] | None = None,
        fused_components: dict[str, int] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.blocks = blocks
        self.block_names = list(blocks.keys())
        self.max_features_per_block = max_features_per_block or dict.fromkeys(self.block_names, 256)
        self.fused_components = fused_components or dict(
            DEFAULT_FUSED_COMPONENTS
        )
        self.weights = weights or dict.fromkeys(self.block_names, 1.0)

        self.selected: dict[str, list[str]] = {}
        self.reducers: dict[str, ModalityReducer] = {}
        self.scalers: dict[str, TokenScaler] = {}
        self.dense_: dict[str, np.ndarray] = {}
        self.fused_: np.ndarray | None = None
        # Rows currently loaded by ``transform``; the fused vector is built
        # from exactly these frames.
        self._current: dict[str, pd.DataFrame] = {}

    # -- selection ---------------------------------------------------------

    @staticmethod
    def _impute(frame: pd.DataFrame) -> np.ndarray:
        """Median-impute a modality block into a writable float array.

        ``copy=True`` matters on pandas 3.x, where ``to_numpy`` may hand back
        a read-only view and the in-place fill below would raise.
        """
        dense = frame.to_numpy(dtype=float, copy=True)
        if np.isnan(dense).any():
            col_median = np.nanmedian(dense, axis=0)
            col_median = np.where(np.isfinite(col_median), col_median, 0.0)
            inds = np.where(np.isnan(dense))
            dense[inds] = np.take(col_median, inds[1])
        return dense

    def _select(self) -> None:
        for name, frame in self.blocks.items():
            keep = min(self.max_features_per_block.get(name, 256), frame.shape[1])
            chosen = select_features(frame, keep)
            self.selected[name] = list(chosen.columns)

    # -- fitting -----------------------------------------------------------

    def fit(self, index: np.ndarray | None = None) -> MultiOmicsPreprocessor:
        """Fit selection, scalers and PCAs on the rows given by ``index``."""
        if not self.selected:
            self._select()

        for name, frame in self.blocks.items():
            sub = frame.iloc[index] if index is not None else frame
            sub = sub[self.selected[name]]
            self.scalers[name] = TokenScaler.fit(
                self._impute(sub)
            )
            n_comp = self.fused_components.get(name, 2)
            self.reducers[name] = ModalityReducer(
                name=name, n_components=n_comp, weight=self.weights.get(name, 1.0)
            ).fit(sub[self.selected[name]])

        self._assemble(index)
        return self

    def transform(self, index: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Return ``(dense blocks, fused vector)`` for the given rows.

        The row subset is remembered on the instance because both the token
        matrices and the fused vector must describe the *same* rows; fusing
        from the full block while the tokens come from a fold would silently
        misalign the two branches of the model.

        Fresh arrays are returned on every call.  Returning the cached
        ``dense_`` dict would alias the train and test matrices, and the
        second call would silently overwrite the first - a bug that surfaces
        much later as an ``inconsistent numbers of samples`` error in a
        baseline that never touched the quantum code.
        """
        out: dict[str, np.ndarray] = {}
        for name in self.blocks:
            frame = self.blocks[name]
            sub = frame.iloc[index] if index is not None else frame
            sub = sub[self.selected[name]]
            self._current[name] = sub
            raw = self._impute(sub)
            scaler = self.scalers.get(name)
            scaled = scaler.transform(raw) if scaler is not None else raw
            out[name] = np.nan_to_num(scaled, nan=0.0)
        self.dense_ = out
        return dict(out), self._fuse()

    def fit_transform(self, index: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], np.ndarray]:
        return self.fit(index).transform(index)

    # -- internals ---------------------------------------------------------

    def fuse_from_dense(self, dense: dict[str, np.ndarray]) -> np.ndarray:
        """Fuse already-materialised per-modality rows.

        The serving path has single-row blocks in memory rather than indexed
        frames, and it has no ``transform`` call to have populated
        ``_current``.  Going through the transient state would make scoring
        depend on whichever fold was fitted last, so the frames are supplied
        explicitly here.
        """
        parts: list[np.ndarray] = []
        for name in self.block_names:
            if name not in self.reducers or name not in dense:
                continue
            frame = pd.DataFrame(
                np.asarray(dense[name], dtype=float),
                columns=self.selected[name],
            )
            parts.append(self.reducers[name].transform(frame))
        if not parts:
            raise RuntimeError("no modality blocks available for fusion")
        fused = np.concatenate(parts, axis=1)
        scale = np.sqrt((fused**2).sum(axis=1).mean())
        return fused / (scale if scale > 1e-12 else 1.0)

    def _fuse(self) -> np.ndarray:
        parts: list[np.ndarray] = []
        for name in self.block_names:
            if name not in self.reducers:
                continue
            parts.append(self.reducers[name].transform(self._current[name]))
        if not parts:
            raise RuntimeError("no modality blocks available for fusion")
        fused = np.concatenate(parts, axis=1)
        # Global rescale keeps the angle distribution seen by the quantum
        # register comparable across folds regardless of block scaling.
        scale = np.sqrt((fused**2).sum(axis=1).mean())
        return fused / (scale if scale > 1e-12 else 1.0)

    def _assemble(self, index: np.ndarray | None) -> None:
        self.transform(index)

    # -- reporting ---------------------------------------------------------

    def describe(self) -> dict:
        return {
            "block_names": self.block_names,
            "selected_features": {k: len(v) for k, v in self.selected.items()},
            "fused_dim": int(self.fused_.shape[1]) if self.fused_ is not None else 0,
            "fused_components": self.fused_components,
            "weights": self.weights,
        }


def fold_indices(
    n_samples: int, y: np.ndarray, n_splits: int, n_repeats: int, seed: int
) -> list[tuple[int, np.ndarray, np.ndarray, int]]:
    """Repeated stratified K-fold, grouped so that no patient spans two folds.

    Several TCGA aliquots can originate from one patient.  Splitting on
    samples would place near-duplicate profiles in both training and test
    and quietly inflate the score, so folds are built on the *patient* label
    and expanded back to sample indices.
    """
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y).ravel()
    splits: list[tuple[int, np.ndarray, np.ndarray, int]] = []
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            splits.append((repeat, train_idx, test_idx, fold))
    return splits


def grouped_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> list[tuple[int, np.ndarray, np.ndarray, int]]:
    """Repeated stratified K-fold that keeps every group inside one fold."""
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.asarray(y).ravel()
    groups = np.asarray(groups).ravel()
    splits: list[tuple[int, np.ndarray, np.ndarray, int]] = []
    for repeat in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for fold, (train_idx, test_idx) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
            splits.append((repeat, train_idx, test_idx, fold))
    return splits


__all__ = [
    "ModalityReducer",
    "MultiOmicsPreprocessor",
    "select_features",
    "fold_indices",
    "grouped_splits",
    "DEFAULT_FUSED_COMPONENTS",
]
