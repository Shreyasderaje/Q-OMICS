"""Quantum kernel computation and diagnostics."""

from .features import (
    QuantumFeatureMap,
    analytic_sep_rx_gram,
    classical_gram,
    pennylane_available,
)
from .kernels import (
    FidelityQuantumKernel,
    KernelFingerprint,
    apply_normalisation,
    arc_cosine_normalise,
    centre,
    cosine_normalise,
    kernel_diagnostics,
    labelled_separation,
)

__all__ = [
    "QuantumFeatureMap",
    "pennylane_available",
    "classical_gram",
    "analytic_sep_rx_gram",
    "FidelityQuantumKernel",
    "KernelFingerprint",
    "apply_normalisation",
    "arc_cosine_normalise",
    "cosine_normalise",
    "centre",
    "kernel_diagnostics",
    "labelled_separation",
]
