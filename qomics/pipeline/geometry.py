"""Geometric analysis of quantum versus classical kernel structure.

Does a quantum kernel actually induce a *different* geometry, or is it a
costly re-parameterisation of one a classical kernel already spans?  This is
the sharpest version of the question, and Huang et al. (2024) give a single
scalar that answers it: the **geometric difference**

.. math::

   g(K_q, K_c) = \\sqrt{\\; \\big\\| \\sqrt{K_q}\\, K_c^{-1} \\sqrt{K_q} \\big\\|_\\infty \\;}

The reading is direct.  ``g = 1`` means the two kernels define the *same*
geometry, and no kernel learner - SVM, GP, anything - can do better with one
than the other.  ``g > 1`` means the quantum geometry is not reproducible by
that classical family and may carry independent signal.

Q-OMICS reports ``g`` for the RBF and cosine references so the claim of
quantum-specific structure is falsifiable rather than assumed.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _eig_floor(K: np.ndarray, relative: float = 1e-3) -> float:
    """Eigenvalue floor proportional to the matrix scale.

    An *absolute* floor is the wrong choice here.  A 370x370 RBF Gram matrix
    has many near-zero eigenvalues, and clipping them at 1e-6 makes
    ``K_c^-1`` astronomically large, which sends the geometric difference to
    arbitrary values (observed: 181 and 406) that say nothing about the
    kernels.  Flooring relative to the mean eigenvalue keeps the inverse
    bounded and makes ``g`` comparable across cohorts and qubit widths.
    """
    values = np.linalg.eigvalsh((K + K.T) / 2.0)
    if values.size == 0:
        return relative
    return float(max(np.mean(np.abs(values)) * relative, 1e-12))


def _psd_inverse_root(K: np.ndarray, relative: float = 1e-3) -> np.ndarray:
    """``sqrt(K)`` for a positive semi-definite matrix, via eigendecomposition."""
    K = (K + K.T) / 2.0
    values, vectors = np.linalg.eigh(K)
    # Guard against the null space: fidelity kernels are frequently singular
    # because the state vectors live in a lower-dimensional manifold.
    values = np.clip(values, _eig_floor(K, relative), None)
    return vectors @ np.diag(np.sqrt(values)) @ vectors.T


def geometric_difference(K_quantum: np.ndarray, K_classical: np.ndarray) -> float:
    """Geometric difference ``g`` between a quantum and a classical kernel."""
    sqrt_q = _psd_inverse_root(np.asarray(K_quantum, dtype=float))
    K_c = np.asarray(K_classical, dtype=float)
    K_c = (K_c + K_c.T) / 2.0
    # Invert the classical kernel under the same relative floor.
    values, vectors = np.linalg.eigh(K_c)
    values = np.clip(values, _eig_floor(K_c), None)
    K_c_inv = vectors @ np.diag(1.0 / values) @ vectors.T

    middle = sqrt_q @ K_c_inv @ sqrt_q
    middle = (middle + middle.T) / 2.0
    return float(np.sqrt(max(np.linalg.norm(middle, ord=np.inf), 0.0)))


def kernel_similarity(K_a: np.ndarray, K_b: np.ndarray) -> dict[str, float]:
    """Cheap similarity descriptors between two Gram matrices.

    ``cKA`` is centred kernel alignment, the standard similarity in the
    kernel literature: 1.0 means the two kernels induce identical geometry.
    """
    a = (np.asarray(K_a, dtype=float) + np.asarray(K_a, dtype=float).T) / 2.0
    b = (np.asarray(K_b, dtype=float) + np.asarray(K_b, dtype=float).T) / 2.0
    n = a.shape[0]
    one = np.ones((n, n)) / n
    a_c = a - one @ a - a @ one + one @ a @ one
    b_c = b - one @ b - b @ one + one @ b @ one

    cka = float(np.sum(a_c * b_c) / np.sqrt(np.sum(a_c**2) * np.sum(b_c**2) + 1e-12))

    d = a - b
    off = ~np.eye(n, dtype=bool)
    return {
        "cka": cka,
        "frobenius_distance": float(np.linalg.norm(d)),
        "mean_abs_difference": float(np.abs(d[off]).mean()),
        "max_abs_difference": float(np.abs(d[off]).max()),
    }


def effective_rank_from_spectrum(eigenvalues: np.ndarray) -> float:
    """Entropy-based effective rank: ``exp(H(p))`` over the spectrum."""
    values = np.clip(np.asarray(eigenvalues, dtype=float), 0.0, None)
    total = float(values.sum())
    if total <= 0 or values.size == 0:
        return 0.0
    p = values / total
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(np.exp(-(p * np.log(p)).sum()))


def spectrum(K: np.ndarray, n_components: int = 10) -> dict[str, list[float]]:
    """Leading eigenvalues of a Gram matrix, for a spectral decay plot."""
    values = np.linalg.eigvalsh((np.asarray(K, dtype=float) + np.asarray(K, dtype=float).T) / 2.0)
    values = np.sort(np.clip(values, 0.0, None))[::-1]
    total = float(values.sum()) or 1.0
    cumulative = np.cumsum(values) / total
    return {
        "eigenvalues": [float(v) for v in values[:n_components]],
        "explained_variance_ratio": [float(v / total) for v in values[:n_components]],
        "cumulative": [float(v) for v in cumulative[:n_components]],
        "effective_rank": effective_rank_from_spectrum(values),
    }


def entanglement_gain(
    K_entangled: np.ndarray, K_product: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """How much label structure entanglement adds over a product encoding.

    Compares the labelled separation of the entangled (IQP/HEA) embedding
    with the non-entangled (``sep_rx``) one on the *same* inputs.  Because
    both are fidelity kernels over the same register width, the difference
    isolates the contribution of entanglement rather than of encoding style
    or qubit count.
    """
    from ..quantum.kernels import labelled_separation

    sep_e = labelled_separation(K_entangled, y)
    sep_p = labelled_separation(K_product, y)
    return {
        "separation_entangled": sep_e,
        "separation_product": sep_p,
        "absolute_gain": sep_e - sep_p,
        "relative_gain": (sep_e - sep_p) / abs(sep_p) if abs(sep_p) > 1e-12 else 0.0,
    }


def interpret_geometric_difference(
    g: float,
    *,
    quantum_separation: float | None = None,
    classical_separation: float | None = None,
) -> str:
    """Plain-language reading of ``g``, qualified by the label signal.

    ``g`` on its own is easy to misread.  A very large value does not mean
    "the quantum kernel is better"; it usually means the classical kernel has
    a large near-null space in the direction the quantum kernel uses, and
    ``g`` is measuring that conditioning rather than any useful structure.
    So the reading is qualified by the labelled separation - if the quantum
    kernel does not separate the classes any better than the classical one,
    a large ``g`` is a numerical artefact and is reported as such.
    """
    if not np.isfinite(g):
        return "Geometric difference undefined; the classical kernel is singular."

    separated = (
        quantum_separation is not None
        and classical_separation is not None
        and quantum_separation <= classical_separation
    )
    uninformative = (
        quantum_separation is not None and abs(quantum_separation) < 1e-3
    )

    if uninformative:
        return (
            f"g = {g:.3f}, but the quantum kernel carries essentially no label "
            f"separation (between-class minus within-class similarity "
            f"{quantum_separation:.2e}, below the 1e-3 resolution of this "
            "cohort). A large or small g is not evidence of an advantage: the "
            "kernel does not separate the classes at all, so no kernel learner "
            "can benefit from its geometry. On this cohort the quantum branch "
            "does not outperform its classical references."
        )

    if g < 0.95:
        return (
            f"g = {g:.3f} is below 1, so the quantum geometry is close to a "
            "re-parameterisation of the classical one and offers no structural "
            "advantage for kernel methods on this cohort."
        )
    if g <= 1.05:
        return (
            f"g = {g:.3f} is within 5% of 1, so the quantum and classical "
            "geometries are effectively equivalent; any performance "
            "difference is attributable to regularisation, not to quantum "
            "structure."
        )
    if g > 10.0:
        detail = ""
        if separated:
            detail = (
                " The quantum kernel does separate the classes slightly better, "
                "but by a margin unlikely to survive cross-validation."
            )
        return (
            f"g = {g:.3f} is very large, which indicates that the classical "
            "kernel has a near-null space in the directions the quantum kernel "
            "uses. This is a conditioning artefact of inverting a near-singular "
            f"Gram matrix, not evidence that the quantum geometry is "
            f"advantageous.{detail}"
        )
    return (
        f"g = {g:.3f} exceeds 1, so the quantum geometry is not reproducible by "
        "this classical kernel family and carries independent structure."
    )


def full_analysis(
    K_quantum: np.ndarray,
    classical_kernels: dict[str, np.ndarray],
    y: np.ndarray,
) -> dict:
    """Run the complete geometry report for one cohort."""
    from ..quantum.kernels import kernel_diagnostics, labelled_separation

    separations = {"quantum": labelled_separation(K_quantum, y)}
    differences: dict[str, float] = {}
    similarities: dict[str, dict[str, float]] = {}

    for name, K_c in classical_kernels.items():
        separations[name] = labelled_separation(K_c, y)
        differences[name] = geometric_difference(K_quantum, K_c)
        similarities[name] = kernel_similarity(K_quantum, K_c)

    primary = next(iter(differences), None)
    quantum_sep = separations.get("quantum", 0.0)
    classical_sep = separations.get(primary, 0.0) if primary else 0.0
    return {
        "labelled_separation": separations,
        "geometric_difference": differences,
        "kernel_similarity": similarities,
        "primary_reference": primary,
        "interpretation": interpret_geometric_difference(
            differences[primary],
            quantum_separation=quantum_sep,
            classical_separation=classical_sep,
        )
        if primary
        else "",
        "quantum_diagnostics": kernel_diagnostics(K_quantum),
        "classical_diagnostics": {name: kernel_diagnostics(K) for name, K in classical_kernels.items()},
        "quantum_spectrum": spectrum(K_quantum),
    }


__all__ = [
    "geometric_difference",
    "kernel_similarity",
    "spectrum",
    "entanglement_gain",
    "interpret_geometric_difference",
    "full_analysis",
]
