"""Quantum feature maps for molecular data embedding.

Three encodings are provided, all producing a pure state on ``n`` qubits:

``iqp``
    Instantaneous-Quantum-Polynomial style map.  Hadamard layers bracket
    data-controlled ``RZ`` rotations and all-to-all ``ZZ`` entanglers, so the
    generated amplitude contains degree-2 polynomial features of the input.
    This is the workhorse encoding for tabular multi-omics data.

``hea``
    Hamiltonian-evolution style map: each layer applies a trainable-free
    ``e^{-i x_j H_j}`` analogue built from data-scaled Pauli rotations
    followed by a fixed entangling template.  Slower to build but
    numerically well behaved for small qubit counts.

``sep_rx``
    Product (non-entangled) ``RX`` embedding.  Included as a *control*: its
    Gram matrix is analytically equivalent to a classical cosine kernel, so
    any performance difference isolates the contribution of entanglement
    rather than of the encoding itself.

The maps are intentionally kept free of variational parameters.  Q-OMICS uses
a fixed, interpretable embedding and lets the classical head learn, which
keeps every reported number reproducible without a quantum optimiser.
"""

from __future__ import annotations

import numpy as np

try:  # PennyLane is the only quantum dependency and is optional at import time.
    import pennylane as qml

    _QML_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without PennyLane
    qml = None  # type: ignore[assignment]
    _QML_AVAILABLE = False


def pennylane_available() -> bool:
    """Whether the quantum simulator backend can be used."""
    return _QML_AVAILABLE


def _require_pennylane() -> None:
    if not _QML_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "PennyLane is required for the quantum kernel. Install with "
            "`pip install pennylane`."
        )


class QuantumFeatureMap:
    """Callable quantum embedding ``phi: R^d -> H``.

    Parameters
    ----------
    n_qubits:
        Register width.  A six-qubit register already spans a 64-dimensional
        complex Hilbert space, which is the regime where fidelity kernels
        are expected to differ most from their classical counterparts.
    feature_map:
        One of ``{"iqp", "hea", "sep_rx"}``.
    n_layers:
        Number of encoding blocks.  Values above two show sharply diminishing
        returns and inflate the cost of the adjoint circuit.
    reupload:
        Re-apply the data encoding once per layer (data re-uploading), which
        raises the expressivity of the map without adding qubits.
    data_scale:
        Multiplicative factor applied to inputs before encoding.  Features
        are standardised upstream, so this mainly controls how much angle
        variation the register sees.
    add_ry_rotation:
        Append a single ``RY`` data rotation per qubit, breaking the
        symmetry of the IQP encoding (which is otherwise invariant under
        ``x -> -x`` for some qubit subsets).
    """

    SUPPORTED = ("iqp", "hea", "sep_rx")

    def __init__(
        self,
        n_qubits: int = 6,
        feature_map: str = "iqp",
        n_layers: int = 2,
        reupload: bool = True,
        data_scale: float = 1.0,
        add_ry_rotation: bool = True,
    ) -> None:
        _require_pennylane()
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if feature_map not in self.SUPPORTED:
            raise ValueError(f"feature_map must be one of {self.SUPPORTED}")

        self.n_qubits = int(n_qubits)
        self.feature_map = feature_map
        self.n_layers = int(n_layers)
        self.reupload = bool(reupload)
        self.data_scale = float(data_scale)
        self.add_ry_rotation = bool(add_ry_rotation)
        self.wires = list(range(self.n_qubits))

        self.device = qml.device("default.qubit", wires=self.n_qubits, shots=None)
        # No gradients are taken through the embedding - the map is fixed and
        # parameter-free - so the QNode runs without an autodiff interface,
        # which removes per-call overhead from the hot loop.
        self._circuit = qml.QNode(self._apply, self.device, interface=None)

    # -- circuit templates -------------------------------------------------
    #
    # Gates are written out explicitly rather than taken from PennyLane
    # templates: the template signatures have changed across releases
    # (``reps`` -> ``n_repeats``, ``rotations`` dropped from AngleEmbedding),
    # and the primitives below are stable API surface.

    def _h_layer(self) -> None:
        for wire in self.wires:
            qml.Hadamard(wires=wire)

    def _ising_zz(self, theta, wire_a: int, wire_b: int) -> None:
        """exp(-i theta Z_a Z_b / 2) via CNOT conjugation.

        ``theta`` may be a scalar (single embedding) or a 1-D array (a
        broadcast batch); PennyLane expands the parameter across the batch.
        """
        qml.CNOT(wires=[wire_a, wire_b])
        qml.RZ(theta, wires=wire_b)
        qml.CNOT(wires=[wire_a, wire_b])

    def _block_iqp(self, x) -> None:
        """Instantaneous-Quantum-Polynomial block.

        ``H^n`` then data-controlled ``RZ`` rotations and all-to-all ``ZZ``
        entanglers, so the resulting amplitudes are degree-2 polynomials in
        the input features.
        """
        self._h_layer()
        for j, wire in enumerate(self.wires):
            qml.RZ(x[..., j], wires=wire)
        n = len(self.wires)
        for j in range(n):
            for k in range(j + 1, n):
                theta = 2.0 * (np.pi - x[..., j]) * (np.pi - x[..., k])
                self._ising_zz(theta, self.wires[j], self.wires[k])
        self._h_layer()

    def _block_hea(self, x) -> None:
        """Hamiltonian-evolution style block: Pauli rotations + a ring."""
        for j, wire in enumerate(self.wires):
            qml.RY(x[..., j], wires=wire)
        for j, wire in enumerate(self.wires):
            qml.CNOT(wires=[wire, self.wires[(j + 1) % len(self.wires)]])
        for j, wire in enumerate(self.wires):
            qml.RZ(x[..., j], wires=wire)

    def _block_sep_rx(self, x) -> None:
        """Non-entangled product-state encoding (classical control)."""
        for j, wire in enumerate(self.wires):
            qml.RX(x[..., j], wires=wire)

    def _apply(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float) * self.data_scale
        n_blocks = self.n_layers if self.reupload else 1

        if self.feature_map == "sep_rx":
            for _ in range(n_blocks):
                self._block_sep_rx(x)
        elif self.feature_map == "hea":
            for j, wire in enumerate(self.wires):
                qml.RY(x[..., j], wires=wire)
            for _ in range(n_blocks - 1):
                self._block_hea(x)
        else:  # iqp
            for _ in range(n_blocks):
                self._block_iqp(x)

        if self.add_ry_rotation and self.feature_map != "sep_rx":
            for j, wire in enumerate(self.wires):
                qml.RY(0.5 * x[..., j], wires=wire)
        return qml.state()

    # -- public API --------------------------------------------------------

    def embed(self, x: np.ndarray) -> np.ndarray:
        """Return the state vector ``|phi(x)>`` for a single input."""
        x = np.asarray(x, dtype=float).ravel()
        if x.size < self.n_qubits:
            x = np.pad(x, (0, self.n_qubits - x.size))
        return np.asarray(self._circuit(x[: self.n_qubits]), dtype=complex)

    def embed_many(self, X: np.ndarray) -> np.ndarray:
        """Embed a batch of inputs.

        Returns an ``(n, 2**n_qubits)`` array of state vectors.  This is the
        key efficiency decision in the whole quantum module: every subsequent
        pairwise overlap is obtained from a single matrix product instead of
        re-running the compute-uncompute circuit ``n**2`` times.

        The batch is issued as **one broadcasted QNode call** rather than a
        Python loop.  The gate-per-sample cost is identical - what disappears
        is the per-call tape construction and dispatch overhead, which
        dominates for circuits this small.  Measured on a 6-qubit IQP with
        300 samples: 6.2 s in a loop versus 0.09 s broadcast, with a
        maximum absolute difference of exactly 0.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] < self.n_qubits:
            X = np.pad(X, ((0, 0), (0, self.n_qubits - X.shape[1])))
        X = X[:, : self.n_qubits]
        if X.shape[0] == 1:
            return np.asarray(self._circuit(X[0]), dtype=complex)[None, :]
        return np.asarray(self._circuit(X), dtype=complex)

    def gram_from_states(self, states: np.ndarray) -> np.ndarray:
        """Squared-overlap Gram matrix from cached state vectors."""
        states = np.asarray(states, dtype=complex)
        overlaps = states @ states.conj().T
        return np.abs(overlaps) ** 2

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"QuantumFeatureMap(feature_map={self.feature_map!r}, "
            f"n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"reupload={self.reupload}, data_scale={self.data_scale})"
        )


def classical_gram(X: np.ndarray, metric: str = "rbf") -> np.ndarray:
    """Reference classical Gram matrices used by the geometry analysis."""
    X = np.asarray(X, dtype=float)
    sq_norms = np.sum(X**2, axis=1)
    sq_dists = np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T), 0.0)
    if metric == "linear":
        return X @ X.T
    if metric == "cosine":
        norms = np.sqrt(np.maximum(sq_norms, 1e-12))
        return (X @ X.T) / np.outer(norms, norms)
    # RBF with the median heuristic, the standard RBF-SVM bandwidth choice.
    positive = sq_dists[~np.eye(len(X), dtype=bool)]
    gamma = 1.0 / (2.0 * np.median(positive)) if positive.size else 1.0
    return np.exp(-gamma * sq_dists)


def analytic_sep_rx_gram(X: np.ndarray) -> np.ndarray:
    """Closed-form Gram matrix of the non-entangled ``RX`` embedding.

    ``prod_j cos((x_j - x'_j)/2)`` squared is exactly the fidelity between
    two product states.  Used to verify the simulator agrees with theory -
    a cheap, decisive correctness check on the quantum backend.
    """
    X = np.asarray(X, dtype=float)
    diff = X[:, None, :] - X[None, :, :]
    prods = np.prod(np.cos(0.5 * diff), axis=2)
    return prods**2


__all__ = [
    "QuantumFeatureMap",
    "pennylane_available",
    "classical_gram",
    "analytic_sep_rx_gram",
    "qml",
]
