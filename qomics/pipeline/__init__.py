"""Training, evaluation and biomarker discovery pipelines."""

from .biomarkers import BiomarkerPanel, build_panel, format_report
from .evaluate import (
    ModelResult,
    bootstrap_ci,
    classification_metrics,
    paired_t_test,
    summarise_results,
)
from .experiment import run_experiment
from .geometry import full_analysis, geometric_difference, interpret_geometric_difference
from .preprocess import MultiOmicsPreprocessor, grouped_splits, select_features
from .train import resolve_device, set_seed, train_transformer_fold

__all__ = [
    "BiomarkerPanel",
    "build_panel",
    "format_report",
    "ModelResult",
    "classification_metrics",
    "bootstrap_ci",
    "paired_t_test",
    "summarise_results",
    "run_experiment",
    "geometric_difference",
    "interpret_geometric_difference",
    "full_analysis",
    "MultiOmicsPreprocessor",
    "select_features",
    "grouped_splits",
    "set_seed",
    "resolve_device",
    "train_transformer_fold",
]
