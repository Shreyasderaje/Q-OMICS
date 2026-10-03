"""Q-OMICS: an open hybrid quantum-classical foundation model for multi-omics
biomarker discovery.

The package is organised in layers::

    qomics.data      cohort assembly from public archives
    qomics.quantum   fidelity quantum kernels on a PennyLane simulator
    qomics.models    classical transformer encoder and reference baselines
    qomics.pipeline  preprocessing, training, evaluation, biomarkers
    qomics.serving   FastAPI application exposing the trained artefacts
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
