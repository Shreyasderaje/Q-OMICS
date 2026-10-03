"""Tests for the quantum feature maps and kernel construction."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pennylane", reason="PennyLane is required for the quantum module")

from qomics.quantum.features import (  # noqa: E402
    QuantumFeatureMap,
    analytic_sep_rx_gram,
    classical_gram,
)
from qomics.quantum.kernels import (  # noqa: E402
    apply_normalisation,
    kernel_diagnostics,
    labelled_separation,
)


def test_embedding_is_normalised_and_correctly_sized():
    fmap = QuantumFeatureMap(n_qubits=4, feature_map="iqp", n_layers=2)
    psi = fmap.embed(np.array([0.3, -0.8, 0.1, 0.5]))
    assert psi.shape == (2**4,)
    assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-10)


def test_product_state_gram_matches_closed_form():
    """The simulator must agree with the analytic product-state fidelity.

    This is the decisive correctness check on the quantum backend: if the
    simulator disagreed with theory here, no downstream result would mean
    anything.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 4))
    sep = QuantumFeatureMap(n_qubits=4, feature_map="sep_rx", n_layers=1, add_ry_rotation=False)
    simulated = sep.gram_from_states(sep.embed_many(X))
    theoretical = analytic_sep_rx_gram(X)
    assert np.abs(simulated - theoretical).max() < 1e-9


def test_gram_matrix_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 3))
    fmap = QuantumFeatureMap(n_qubits=3, feature_map="iqp", n_layers=1)
    K = fmap.gram_from_states(fmap.embed_many(X))
    assert np.allclose(K, K.T, atol=1e-12)
    assert np.allclose(np.diag(K), 1.0, atol=1e-10)


def test_entangled_embedding_couples_features():
    """An IQP map must not factorise into independent single-qubit states."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(6, 3))
    iqp = QuantumFeatureMap(n_qubits=3, feature_map="iqp", n_layers=1, add_ry_rotation=False)
    K = iqp.gram_from_states(iqp.embed_many(X))
    off_diagonal = K[~np.eye(6, dtype=bool)]
    assert off_diagonal.max() > 1e-3


def test_input_padding_for_wide_features():
    """More input features than qubits must be truncated, not crash."""
    fmap = QuantumFeatureMap(n_qubits=3, feature_map="iqp", n_layers=1)
    psi = fmap.embed(np.arange(10, dtype=float))
    assert psi.shape == (8,)
    wide = fmap.embed_many(np.random.default_rng(3).normal(size=(4, 10)))
    assert wide.shape == (4, 8)


@pytest.mark.parametrize("scheme", ["none", "cosine", "arc_cosine_scaled"])
def test_normalisation_schemes_are_psd(scheme):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(12, 3))
    fmap = QuantumFeatureMap(n_qubits=3, feature_map="iqp", n_layers=1)
    K = apply_normalisation(fmap.gram_from_states(fmap.embed_many(X)), scheme)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, atol=1e-10)
    assert np.linalg.eigvalsh((K + K.T) / 2).min() > -1e-8


def test_cross_block_uses_training_reference_norms():
    """A test kernel row must share the training rows' scale.

    If the cross block is normalised by its own diagonal, the test rows are
    systematically shrunk relative to training and the SVM's margin is
    computed in the wrong geometry.
    """
    from qomics.quantum.kernels import FidelityQuantumKernel

    rng = np.random.default_rng(5)
    X_train = rng.normal(size=(9, 3))
    X_test = rng.normal(size=(4, 3))

    kernel = FidelityQuantumKernel(n_qubits=3, feature_map="iqp", n_layers=1,
                                  normalisation="arc_cosine_scaled")
    kernel.fit(X_train)
    K_train = kernel.gram(X_train)
    K_cross = kernel.cross(X_test, X_train)

    # A duplicated training point must reproduce its own training row.
    duplicate = kernel.cross(X_train[:1], X_train)
    assert np.allclose(duplicate[0], K_train[0], atol=1e-8)
    assert K_cross.shape == (4, 9)


def test_unknown_normalisation_raises():
    with pytest.raises(ValueError, match="unknown normalisation"):
        apply_normalisation(np.eye(3), "not-a-scheme")  # type: ignore[arg-type]


def test_diagnostics_and_separation():
    rng = np.random.default_rng(6)
    y = np.array([0, 1] * 8)
    X = np.column_stack([rng.normal(size=16), rng.normal(size=16)])
    K = classical_gram(X, "rbf")
    diag = kernel_diagnostics(K)
    assert diag["n"] == 16.0
    assert labelled_separation(K, y) > 0  # separable by construction
