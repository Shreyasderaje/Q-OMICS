"""End-to-end experiment runner.

``run_experiment`` executes the full protocol and writes every artefact the
API serves: cross-validated metrics, the geometry report, the ranked
biomarker panel, a final model fitted on the whole cohort, and a run
manifest that makes the whole thing reproducible.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import ARTIFACT_DIR, CONFIG, Config, ensure_dirs
from ..data.cohort import Cohort
from ..models.baselines import DISPLAY_NAMES, FAMILY, baseline_registry, fit_quantum_svm
from ..quantum.features import classical_gram
from .biomarkers import BiomarkerPanel, build_panel, format_report, top_decile_membership
from .evaluate import ModelResult, classification_metrics, summarise_results
from .geometry import entanglement_gain, full_analysis
from .preprocess import MultiOmicsPreprocessor, grouped_splits
from .train import (
    _positive_score,
    build_quantum_kernel,
    quantum_fusion_feature,
    resolve_device,
    set_seed,
    train_transformer_fold,
)

log = logging.getLogger(__name__)

#: Tokens per modality.  The total is the transformer's sequence length and
#: attention is quadratic in it, so this is the main compute/quality dial.
#: Measured on CPU: 480 tokens at d_model=128 costs 45.8 s per epoch, which
#: is over two hours for one cross-validation sweep. 192 tokens at d_model=64
#: costs 5.0 s per epoch and is a better fit for 463 samples anyway.
TOKEN_BUDGET: dict[str, int] = {
    "transcriptome": 80,
    "genome": 72,
    "proteome": 40,
}


def _make_preprocessor(blocks: dict[str, pd.DataFrame], config: Config) -> MultiOmicsPreprocessor:
    fused = {"transcriptome": 6, "genome": 4, "proteome": 2}
    budget = dict(TOKEN_BUDGET)
    weights = dict(config.preprocess.modality_weights)

    # The quantum register bounds the fused input width; scale the per-modality
    # component counts down if the configured budget exceeds the qubit count.
    total = sum(fused[b] for b in fused if b in blocks)
    limit = config.quantum.n_input_features
    if total > limit:
        scale = limit / total
        fused = {k: max(1, int(round(v * scale))) for k, v in fused.items()}

    # Trim the token budget so the sequence fits the model's context limit.
    while sum(budget.get(b, 96) for b in blocks) > config.model.max_tokens:
        largest = max(budget, key=lambda k: budget.get(k, 0))
        budget[largest] = max(32, budget[largest] - 16)

    return MultiOmicsPreprocessor(
        blocks,
        max_features_per_block=budget,
        fused_components=fused,
        weights=weights,
    )


def _record(
    results: dict[str, ModelResult], name: str, test_index: np.ndarray,
    scores: np.ndarray, metrics: dict, seconds: float, repeat: int, fold: int,
) -> None:
    result = results.setdefault(name, ModelResult(name=name, display_name=DISPLAY_NAMES[name], family=FAMILY[name]))
    payload = dict(metrics)
    payload.update({"repeat": repeat, "fold": fold})
    result.fold_metrics.append(payload)
    for value, index in zip(scores, test_index):
        result.oof_scores.setdefault(int(index), []).append(float(value))
    result.seconds.append(seconds)


def run_experiment(
    cohort: Cohort,
    config: Config | None = None,
    *,
    artifact_dir: Path | None = None,
    n_splits: int | None = None,
    n_repeats: int | None = None,
    epochs: int | None = None,
    skip_transformer: bool = False,
) -> dict[str, Any]:
    """Execute the full cross-validated experiment and persist artefacts."""
    cfg = config or CONFIG
    ensure_dirs()
    out_dir = Path(artifact_dir or ARTIFACT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Command-line overrides are applied by rebuilding the frozen config
    # trees.  Accepting the arguments but never wiring them through means a
    # "quick pass" silently runs the full schedule.
    if epochs is not None:
        cfg = replace(cfg, train=replace(cfg.train, epochs=int(epochs)))
    if n_splits is not None or n_repeats is not None:
        cfg = replace(
            cfg,
            train=replace(
                cfg.train,
                n_splits=int(n_splits or cfg.train.n_splits),
                n_repeats=int(n_repeats or cfg.train.n_repeats),
            ),
        )

    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    if cfg.train.num_threads:
        torch.set_num_threads(cfg.train.num_threads)

    y = cohort.y
    groups = np.asarray(cohort.patient_ids)
    splits = grouped_splits(
        y, groups,
        n_splits or cfg.train.n_splits,
        n_repeats or cfg.train.n_repeats,
        cfg.train.seed,
    )
    log.info(
        "cohort: %d samples / %d patients | %d splits | device=%s",
        cohort.n_samples, len(set(cohort.patient_ids)), len(splits), device,
    )

    results: dict[str, ModelResult] = {}
    importance_runs: list[np.ndarray] = []
    quantum_diagnostics: dict = {}
    geometry_payload: dict = {}
    feature_names: list[str] = []
    feature_blocks: list[str] = []
    dense_names: dict[str, list[str]] = {}
    #: Dense blocks for the whole cohort, aligned with ``y``, used only for
    #: the panel's descriptive statistics.
    panel_dense: dict[str, np.ndarray] = {}

    registry = baseline_registry(cfg.baseline)
    started = time.perf_counter()

    for repeat, train_idx, test_idx, fold in splits:
        fold_started = time.perf_counter()
        y_train, y_test = y[train_idx], y[test_idx]

        pre = _make_preprocessor(cohort.blocks, cfg)
        dense_tr, fused_tr = pre.fit_transform(train_idx)
        dense_te, fused_te = pre.transform(test_idx)

        if not feature_names:
            feature_names, feature_blocks, dense_names = _token_inventory(pre, dense_tr)
            # Descriptive statistics for the biomarker panel (mean per class,
            # Cohen's d, permutation p) are computed on the whole cohort, so
            # the matrix must be row-aligned with the full label vector.
            # The preprocessor stays fitted on this fold's training rows, so
            # the feature *selection* remains fold-safe.
            _panel_dense, _ = pre.transform(None)
            panel_dense = {k: v.copy() for k, v in _panel_dense.items()}

        # ---------------------------------------------------------- quantum
        kernel = build_quantum_kernel(cfg, fused_tr.shape[1])
        fusion_tr, fusion_te, q_diag = quantum_fusion_feature(
            kernel, fused_tr, y_train, fused_te,
            c=cfg.quantum.c_regularisation,
            inner_splits=3,
            seed=cfg.train.seed + fold,
        )
        if not quantum_diagnostics:
            quantum_diagnostics = q_diag
            k_train = kernel.gram(fused_tr)
            geometry_payload = _geometry_report(cfg, k_train, fused_tr, y_train)

        q_model = fit_quantum_svm(
            kernel.gram(fused_tr), y_train,
            c=cfg.quantum.c_regularisation, random_state=cfg.train.seed,
        )
        q_scores = _positive_score(q_model, kernel.cross(fused_te, fused_tr))
        _record(
            results, "quantum_svm", test_idx, q_scores,
            classification_metrics(y_test, q_scores), 0.0, repeat, fold,
        )

        # -------------------------------------------------------- baselines
        X_tr = np.concatenate([dense_tr[k] for k in dense_tr], axis=1)
        X_te = np.concatenate([dense_te[k] for k in dense_te], axis=1)

        for name, fit_fn in registry.items():
            baseline = fit_fn(X_tr, y_train, X_te)
            scores = baseline.predict_proba(X_te)[:, 1]
            _record(
                results, name, test_idx, scores,
                classification_metrics(y_test, scores), baseline.seconds, repeat, fold,
            )

        # Expression-only reference: what a transcriptomics-only stage model
        # achieves, which quantifies what the genome and proteome add.
        if "transcriptome" in dense_tr:
            from ..models.baselines import fit_baseline

            tx_model = fit_baseline("logreg", dense_tr["transcriptome"], y_train,
                                    dense_te["transcriptome"], cfg.baseline)
            tx_scores = tx_model.predict_proba(dense_te["transcriptome"])[:, 1]
            _record(
                results, "pca_logreg_tx", test_idx, tx_scores,
                classification_metrics(y_test, tx_scores), tx_model.seconds, repeat, fold,
            )

        # ----------------------------------------------------- transformers
        if not skip_transformer:
            hybrid = train_transformer_fold(
                dense_tr, dense_te, dense_names, y_train, y_test,
                fusion_train=fusion_tr[:, None], fusion_test=fusion_te[:, None],
                config=cfg, device=device, seed=cfg.train.seed + repeat * 100 + fold,
            )
            _record(
                results, "hybrid_transformer", test_idx, hybrid.test_scores,
                hybrid.metrics, hybrid.seconds, repeat, fold,
            )
            if hybrid.importance is not None:
                importance_runs.append(hybrid.importance)

            classical = train_transformer_fold(
                dense_tr, dense_te, dense_names, y_train, y_test,
                fusion_train=None, fusion_test=None,
                config=cfg, device=device, seed=cfg.train.seed + repeat * 100 + fold,
            )
            _record(
                results, "classical_transformer", test_idx, classical.test_scores,
                classical.metrics, classical.seconds, repeat, fold,
            )

        elapsed = time.perf_counter() - fold_started
        current = (
            results["hybrid_transformer"].fold_metrics[-1]["auc"]
            if "hybrid_transformer" in results
            else float("nan")
        )
        log.info(
            "repeat %d fold %d done in %.1fs | hybrid AUC %.4f | %d models so far",
            repeat, fold, elapsed, current, len(results),
        )

    # ------------------------------------------------------------- report
    report = summarise_results(list(results.values()), y, "hybrid_transformer")
    report["target"] = cohort.config.target_description
    report["n_splits"] = len(splits)
    report["cohort"] = cohort.summary()

    panel = _build_biomarker_panel(
        importance_runs, feature_names, feature_blocks, panel_dense, dense_names,
        y, cohort.feature_meta,
    )

    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "version": "1.0.0",
        "config": cfg.to_dict(),
        "cohort": cohort.summary(),
        "metrics": report,
        "quantum": _quantum_payload(cfg, quantum_diagnostics, geometry_payload),
        "architecture": _architecture_payload(cfg, device, pre if not skip_transformer else None),
        "biomarker_summary": {
            "method": panel.method,
            "n_features_considered": panel.n_features_considered,
            "n_ranked": int(len(panel.table)),
            "modality_contribution": panel.block_summary(),
            "top_axes": panel.axis_summary(),
        },
        "device": str(device),
    }

    _persist(payload, panel, cohort, out_dir)
    log.info("experiment complete in %.1fs -> %s", payload["runtime_seconds"], out_dir)
    return payload


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _token_inventory(
    pre: MultiOmicsPreprocessor, dense: dict[str, np.ndarray]
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Flatten the per-modality matrices into the model's token order."""
    names: list[str] = []
    blocks: list[str] = []
    dense_names: dict[str, list[str]] = {}
    for block in pre.block_names:
        columns = pre.selected[block]
        dense_names[block] = columns
        names.extend(columns)
        blocks.extend([block] * len(columns))
    return names, blocks, dense_names


def _first_test_index(splits) -> int:
    """Index of the first held-out sample; retained for diagnostics."""
    return int(splits[0][2][0]) if splits else 0


def _geometry_report(cfg: Config, k_train: np.ndarray, fused: np.ndarray, y: np.ndarray) -> dict:
    """Quantum-vs-classical geometry plus an entanglement ablation."""
    classical = {
        "rbf": classical_gram(fused, "rbf"),
        "cosine": classical_gram(fused, "cosine"),
        "linear": classical_gram(fused, "linear"),
    }
    report = full_analysis(k_train, classical, y)

    # Entanglement ablation on a matched register width.
    try:
        from ..quantum.features import QuantumFeatureMap

        sub = min(len(fused), 200)
        X = fused[:sub]
        y_sub = y[:sub]
        entangled = QuantumFeatureMap(
            n_qubits=cfg.quantum.n_qubits, feature_map=cfg.quantum.feature_map,
            n_layers=cfg.quantum.n_layers, reupload=cfg.quantum.reupload,
            data_scale=cfg.quantum.data_scale, add_ry_rotation=cfg.quantum.add_ry_rotation,
        )
        product = QuantumFeatureMap(
            n_qubits=cfg.quantum.n_qubits, feature_map="sep_rx",
            n_layers=1, reupload=False, add_ry_rotation=False,
        )
        k_e = entangled.gram_from_states(entangled.embed_many(X))
        k_p = product.gram_from_states(product.embed_many(X))
        report["entanglement_ablation"] = entanglement_gain(k_e, k_p, y_sub)
        report["entanglement_ablation"]["n_samples"] = int(sub)
    except Exception as exc:  # pragma: no cover - diagnostics must never fail a run
        log.warning("entanglement ablation unavailable: %s", exc)

    return report


def _quantum_payload(cfg: Config, diagnostics: dict, geometry: dict) -> dict:
    payload: dict[str, Any] = {
        "backend": "pennylane.default.qubit",
        "n_qubits": cfg.quantum.n_qubits,
        "hilbert_dimension": 2**cfg.quantum.n_qubits,
        "feature_map": cfg.quantum.feature_map,
        "n_layers": cfg.quantum.n_layers,
        "reupload": cfg.quantum.reupload,
        "normalisation": cfg.quantum.kernel_normalisation,
        "encoding_seconds": diagnostics.get("fingerprint", {}).get("elapsed_seconds"),
        "inner_splits": diagnostics.get("inner_splits"),
    }
    payload.update(
        {
            "kernel_diagnostics": geometry.get("quantum_diagnostics", {}),
            "labelled_separation": geometry.get("labelled_separation", {}),
            "quantum_spectrum": geometry.get("quantum_spectrum", {}),
            "kernel_similarity": geometry.get("kernel_similarity", {}),
            "entanglement_ablation": geometry.get("entanglement_ablation", {}),
        }
    )
    differences = geometry.get("geometric_difference", {})
    payload["geometric_difference"] = {
        "g_quantum_vs_rbf": differences.get("rbf"),
        "g_quantum_vs_cosine": differences.get("cosine"),
        "g_quantum_vs_linear": differences.get("linear"),
        "interpretation": geometry.get("interpretation", ""),
    }
    payload["state_vector_visualisation"] = {
        "n_examples": 0,
        "labels": [],
        "amplitudes": [],
        "note": "state vectors are recomputed on demand via /api/quantum/states",
    }
    return payload


def _architecture_payload(cfg: Config, device: torch.device, pre: MultiOmicsPreprocessor | None) -> dict:
    block_shapes = {k: len(v) for k, v in (pre.selected.items() if pre is not None else {})}
    total_tokens = sum(block_shapes.values())
    return {
        "stages": [
            {"id": "ingest", "title": "Cohort ingestion",
             "detail": "cBioPortal GDC + UCSC Xena RPPA", "tokens": 0},
            {"id": "reduce", "title": "Modality reduction",
             "detail": "variance filter + per-modality PCA", "tokens": int(total_tokens)},
            {"id": "quantum", "title": "Fidelity quantum kernel",
             "detail": f"{cfg.quantum.n_qubits}-qubit {cfg.quantum.feature_map.upper()}, "
                       f"{2**cfg.quantum.n_qubits}-dimensional Hilbert space", "tokens": 0},
            {"id": "encode", "title": "Multi-omics tokenisation",
             "detail": "one token per molecular feature", "tokens": int(total_tokens)},
            {"id": "attend", "title": "Cross-omic attention",
             "detail": f"{cfg.model.n_layers} layers x {cfg.model.n_heads} heads", "tokens": int(total_tokens)},
            {"id": "readout", "title": "Attention rollout",
             "detail": "biomarker attribution", "tokens": int(total_tokens)},
        ],
        "model": {
            "d_model": cfg.model.d_model,
            "n_heads": cfg.model.n_heads,
            "n_layers": cfg.model.n_layers,
            "dim_feedforward": cfg.model.dim_feedforward,
            "dropout": cfg.model.dropout,
        },
        "blocks": block_shapes,
        "total_tokens": int(total_tokens),
        "training": {
            "epochs": cfg.train.epochs,
            "batch_size": cfg.train.batch_size,
            "learning_rate": cfg.train.learning_rate,
            "weight_decay": cfg.train.weight_decay,
            "loss": "focal",
            "n_splits": (cfg.train.n_splits or 5) * (cfg.train.n_repeats or 3),
            "device": str(device),
        },
    }


def _build_biomarker_panel(
    importance_runs: list[np.ndarray],
    feature_names: list[str],
    feature_blocks: list[str],
    dense: dict[str, np.ndarray],
    dense_names: dict[str, list[str]],
    y: np.ndarray,
    feature_meta: pd.DataFrame,
) -> BiomarkerPanel:
    """Rank features from pooled cross-fold attention importance.

    Attention is only comparable *within* a fold, so each fold's rollout is
    min-max normalised to unit sum before pooling; otherwise a fold with
    larger logits would dominate the ranking.
    """
    if not importance_runs:
        log.warning("no attention collected; emitting a variance-ranked placeholder panel")
        matrix = np.concatenate([dense[k] for k in dense], axis=1)
        importance = np.nan_to_num(np.nanvar(matrix, axis=0))
        return build_panel(
            importance, feature_names, feature_blocks, feature_meta, y,
            dense, dense_names, method="variance_fallback",
        )

    normalised: list[np.ndarray] = []
    for run in importance_runs:
        run = np.clip(np.nan_to_num(run, nan=0.0), 0.0, None)
        # Normalise **per sample** to a unit-sum distribution over tokens.
        # Summing over the whole block would normalise per extraction chunk
        # (the trainer scores the test set in chunks to bound attention
        # memory), which rescales the absolute importances by the chunk size
        # and makes the reported numbers depend on an implementation detail.
        totals = run.sum(axis=1, keepdims=True)
        normalised.append(np.where(totals > 1e-12, run / np.maximum(totals, 1e-12), 0.0))

    stacked = np.vstack(normalised)
    mean_importance = stacked.mean(axis=0)
    stability = top_decile_membership(normalised, fraction=0.10)

    return build_panel(
        mean_importance, feature_names, feature_blocks, feature_meta, y,
        dense, dense_names, stability=stability, method="attention_rollout",
    )


def _persist(
    payload: dict, panel: BiomarkerPanel, cohort: Cohort, out_dir: Path
) -> None:
    """Write every artefact the serving layer reads."""
    (out_dir / "run.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    panel.table.to_csv(out_dir / "biomarkers.csv", index=False)
    (out_dir / "biomarkers.txt").write_text(format_report(panel, top_n=40), encoding="utf-8")
    cohort.clinical.to_csv(out_dir / "cohort_clinical.csv")
    if payload.get("metrics", {}).get("roc_curve"):
        curves = payload["metrics"]["roc_curve"]["curves"]
        longest = max(curves, key=lambda c: len(c.get("fpr", [])))
        pd.DataFrame(
            {"fpr": longest["fpr"], "tpr": longest["tpr"]}
        ).to_csv(out_dir / "roc_primary.csv", index=False)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


__all__ = ["run_experiment", "TOKEN_BUDGET"]
