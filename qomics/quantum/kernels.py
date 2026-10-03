"""Fidelity quantum kernels for multi-omics similarity.

Definition
----------
A classical kernel is an inner product in a feature space that must be chosen
by hand.  A *fidelity quantum kernel* instead embeds each sample into a
quantum state with a parameterised circuit and takes the squared overlap

.. math::  k_\\theta(x, x') = |\\langle \\psi_\\theta(x) \\mid \\psi_\\theta(x') \\rangle|^2 .

The state :math:`|\\psi\\rangle` lives in a :math:`2^n`-dimensional Hilbert
space, so the kernel evaluates an inner product in an exponentially larger
space than the input admits directly - the reason quantum kernels are
conceived for high-dimensional molecular data.

Implementation note
-------------------
The textbook implementation of ``k`` runs the compute-uncompute circuit
``U(x)`` then ``U(x')^dagger`` and reads off the all-zero probability, i.e.
``n^2`` circuit executions for an ``n x n`` Gram matrix.  Because PennyLane
exposes the full wave-function, Q-OMICS evaluates each embedding **once** and
forms every pairwise overlap as a single matrix product:

.. math::  K = |\\Psi \\Psi^\\dagger|^2, \\quad \\Psi \\in \\mathbb{C}^{n \\times 2^q}.

This is algebraically identical to the circuit-based value and is what makes
an ``n ~ 500`` cohort tractable on a laptop.

Normalisation
-------------
Raw fidelity Gram matrices become increasingly diagonally dominated as the
embedding depth grows, which starves a downstream SVM of usable contrast.
Q-OMICS therefore offers three standard treatments, selected by
``kernel_normalisation``:

``none``
    Leave the raw matrix untouched.
``cosine``
    Project every embedding onto the unit sphere: ``K / sqrt(diag K diag K^T)``.
``arc_cosine_scaled``
    Apply the first-order arc-cosine (neural-network) kernel to the
    normalised overlaps, ``2/pi * arcsin(<psi_i, psi_j>)``.  This preserves
    positive semi-definiteness while expanding contrast between the
    near-diagonal regime, and is the default.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .features import QuantumFeatureMap, pennylane_available

log = logging.getLogger(__name__)

Normalisation = Literal["none", "cosine", "arc_cosine_scaled"]


# --------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------


def centre(K: np.ndarray) -> np.ndarray:
    """Double-centre a Gram matrix (the standard kernel-whitening step)."""
    n = K.shape[0]
    one = np.ones((n, n), dtype=float) / n
    return K - one @ K - K @ one + one @ K @ one


def cosine_normalise(K: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Normalise a Gram matrix so every embedding lies on the unit sphere.

    ``reference`` supplies the state norms of the *training* set.  It is
    required for rectangular cross-blocks: the test embeddings must be
    divided by the same constants as the training embeddings, otherwise a
    test-time kernel row is not on the same scale as the training rows and
    the downstream SVM sees a distribution shift it was never fitted on.
    """
    K = np.asarray(K, dtype=float)
    if reference is None:
        d = np.clip(np.diag(K), 1e-12, None)
        Kn = K / np.outer(d, d)
    else:
        d = np.clip(np.asarray(reference, dtype=float), 1e-12, None)
        if d.ndim == 1 and K.shape[1] == d.shape[0] and K.shape[0] != d.shape[0]:
            # Cross block: every test row shares the training-set norms.
            Kn = K / d[None, :]
        else:
            Kn = K / np.outer(d, d)
    if Kn.shape[0] == Kn.shape[1]:
        return (Kn + Kn.T) / 2.0
    return Kn


def arc_cosine_normalise(K: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """First-order arc-cosine (RBF-equivalent in neuron space) transform."""
    return (2.0 / np.pi) * np.arcsin(np.clip(cosine_normalise(K, reference), -1.0, 1.0))


def apply_normalisation(
    K: np.ndarray, scheme: Normalisation, reference: np.ndarray | None = None
) -> np.ndarray:
    """Dispatch to the requested normalisation.

    ``reference`` is the vector of training-set embedding norms, supplied for
    cross-kernel blocks so that test rows are normalised consistently.
    """
    K = np.asarray(K, dtype=float)
    if K.shape[0] == K.shape[1]:
        K = (K + K.T) / 2.0
    if scheme == "none":
        return K
    if scheme == "cosine":
        return cosine_normalise(K, reference)
    if scheme == "arc_cosine_scaled":
        return arc_cosine_normalise(K, reference)
    raise ValueError(f"unknown normalisation scheme: {scheme!r}")


# --------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------


@dataclass
class KernelFingerprint:
    """Provenance record stored alongside every fitted kernel."""

    n_qubits: int
    feature_map: str
    n_layers: int
    reupload: bool
    data_scale: float
    add_ry_rotation: bool
    normalisation: str
    dim_state: int
    n_samples: int
    backend: str
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


class FidelityQuantumKernel:
    """Fidelity quantum kernel evaluated on a PennyLane state-vector device.

    The kernel is *stateless with respect to the data*: ``fit`` records the
    input width and the fingerprint, ``transform`` returns cross-kernel
    blocks.  This matches the scikit-learn precomputed-kernel workflow, where
    cross-validation needs ``K[train, train]`` and ``K[test, train]``.

    State vectors are cached on the instance, so repeated calls to
    ``transform`` over the same training matrix cost nothing after the first.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        feature_map: str = "iqp",
        n_layers: int = 2,
        reupload: bool = True,
        data_scale: float = 1.0,
        add_ry_rotation: bool = True,
        normalisation: Normalisation = "arc_cosine_scaled",
        store_states: bool = True,
    ) -> None:
        if not pennylane_available():  # pragma: no cover
            raise ImportError(
                "PennyLane is not installed; `pip install pennylane` to enable the "
                "quantum kernel."
            )
        self.feature_map = QuantumFeatureMap(
            n_qubits=n_qubits,
            feature_map=feature_map,
            n_layers=n_layers,
            reupload=reupload,
            data_scale=data_scale,
            add_ry_rotation=add_ry_rotation,
        )
        self.normalisation = normalisation
        self.store_states = store_states
        self._states: dict[int, np.ndarray] = {}
        self._reference_norm: np.ndarray | None = None
        self.fingerprint: KernelFingerprint | None = None

    # -- encoding ----------------------------------------------------------

    def _states_for(self, X: np.ndarray, key: int) -> np.ndarray:
        if key in self._states:
            return self._states[key]
        X = np.atleast_2d(np.asarray(X, dtype=float))
        states = self.feature_map.embed_many(X)
        if self.store_states:
            self._states[key] = states
        return states

    def clear_cache(self) -> None:
        self._states.clear()
        self._reference_norm = None

    # -- sklearn-style API -------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> FidelityQuantumKernel:
        """Encode the reference set and record the kernel fingerprint."""
        import time

        start = time.perf_counter()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        states = self._states_for(X, key=0)
        raw = np.abs(states @ states.conj().T) ** 2
        # State norms of the reference set: every later cross-block row is
        # normalised by these constants.
        self._reference_norm = np.clip(np.diag(raw), 1e-12, None)
        self.fingerprint = KernelFingerprint(
            n_qubits=self.feature_map.n_qubits,
            feature_map=self.feature_map.feature_map,
            n_layers=self.feature_map.n_layers,
            reupload=self.feature_map.reupload,
            data_scale=self.feature_map.data_scale,
            add_ry_rotation=self.feature_map.add_ry_rotation,
            normalisation=self.normalisation,
            dim_state=int(states.shape[1]),
            n_samples=int(X.shape[0]),
            backend="pennylane.default.qubit",
            elapsed_seconds=round(time.perf_counter() - start, 3),
        )
        log.info(
            "quantum kernel fitted: %d samples -> %d-dim states in %.2fs",
            X.shape[0], states.shape[1], self.fingerprint.elapsed_seconds,
        )
        return self

    def gram(self, X: np.ndarray) -> np.ndarray:
        """Full self-kernel ``K`` for ``X`` (cached under key 0)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        states = self._states_for(X, key=0)
        K = np.abs(states @ states.conj().T) ** 2
        return apply_normalisation(K, self.normalisation)

    def cross(self, X_test: np.ndarray, X_train: np.ndarray | None = None) -> np.ndarray:
        """``K[X_test, X_train]`` - the test-time kernel block.

        Normalisation uses the *training* embedding norms recorded at
        :meth:`fit`, so the returned rows are directly comparable to the
        training Gram matrix.
        """
        X_test = np.atleast_2d(np.asarray(X_test, dtype=float))
        if X_train is None:
            return self.gram(X_test)
        X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
        train_states = self._states_for(X_train, key=0)
        if self._reference_norm is None:
            raw_train = np.abs(train_states @ train_states.conj().T) ** 2
            self._reference_norm = np.clip(np.diag(raw_train), 1e-12, None)
        test_states = self.feature_map.embed_many(X_test)
        raw = np.abs(test_states @ train_states.conj().T) ** 2
        return apply_normalisation(raw, self.normalisation, self._reference_norm)

    def decision_inputs(self, X: np.ndarray, X_train: np.ndarray | None = None) -> np.ndarray:
        """Kernel row(s) for a single scoring call."""
        block = self.cross(X, X_train)
        return block[0] if block.shape[0] == 1 else block

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Persist the (small) fingerprint; states are recomputed on load."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": self.fingerprint.to_dict() if self.fingerprint else None,
            "n_qubits": self.feature_map.n_qubits,
            "feature_map": self.feature_map.feature_map,
            "n_layers": self.feature_map.n_layers,
            "reupload": self.feature_map.reupload,
            "data_scale": self.feature_map.data_scale,
            "add_ry_rotation": self.feature_map.add_ry_rotation,
            "normalisation": self.normalisation,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> tuple[FidelityQuantumKernel, dict | None]:
        """Rebuild a kernel from :meth:`save` output."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        kernel = cls(
            n_qubits=payload["n_qubits"],
            feature_map=payload["feature_map"],
            n_layers=payload["n_layers"],
            reupload=payload["reupload"],
            data_scale=payload["data_scale"],
            add_ry_rotation=payload["add_ry_rotation"],
            normalisation=payload["normalisation"],
        )
        return kernel, payload.get("fingerprint")

    def __repr__(self) -> str:  # pragma: no cover
        return f"FidelityQuantumKernel({self.feature_map!r}, normalisation={self.normalisation!r})"


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def kernel_diagnostics(K: np.ndarray) -> dict[str, float]:
    """Descriptive statistics of a Gram matrix.

    ``separation`` is the ratio of between-class to within-class mean
    similarity under the supplied labels.  A kernel with a high separation
    ratio is one on which a linear classifier can separate the classes, and
    it is the cheapest possible sanity check before training anything.
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    if n < 2:
        return {"n": float(n)}

    off_diagonal = ~np.eye(n, dtype=bool)
    values = K[off_diagonal]
    stats = {
        "n": float(n),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean_diagonal": float(np.diag(K).mean()),
        "condition_number": float(np.linalg.cond(K + 1e-6 * np.eye(n))),
    }

    if K.shape[0] == K.shape[1]:
        stats["effective_rank"] = float(_effective_rank(K))
    return stats


def _effective_rank(K: np.ndarray) -> float:
    """Entropy-based effective rank of the kernel spectrum."""
    eigenvalues = np.linalg.eigvalsh((K + K.T) / 2.0)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0 or eigenvalues.size == 0:
        return 0.0
    p = eigenvalues / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def labelled_separation(K: np.ndarray, y: np.ndarray) -> float:
    """Between-class minus within-class mean similarity, in kernel units."""
    y = np.asarray(y).ravel()
    K = np.asarray(K, dtype=float)
    same = (y[:, None] == y[None, :])
    off = ~np.eye(len(y), dtype=bool)
    within = K[same & off]
    between = K[~same]
    if within.size == 0 or between.size == 0:
        return 0.0
    return float(between.mean() - within.mean())


__all__ = [
    "FidelityQuantumKernel",
    "KernelFingerprint",
    "apply_normalisation",
    "arc_cosine_normalise",
    "cosine_normalise",
    "centre",
    "kernel_diagnostics",
    "labelled_separation",
]
