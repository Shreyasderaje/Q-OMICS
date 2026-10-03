"""Training and cross-validation for the hybrid quantum-classical model.

One run performs the entire experiment and writes every artefact the API
serves.  The protocol per split is:

1. Fit the preprocessor (variance filter, scaler, PCA) on the **training**
   rows only.
2. Encode the fused training vector into a quantum state per sample and form
   the fidelity Gram matrix; compute the test block by a single cross-kernel.
3. Train a quantum-kernel SVM on ``K[train, train]``.
4. Produce the *fusion feature* the transformer consumes.  For training rows
   this is a **stacked out-of-fold** score: a second, inner split trains the
   SVM on part of the training set and scores the rest, so the transformer
   never sees a quantum score that was fitted on the same sample.
5. Train the dense-feature baselines and both transformers, score the test
   rows, and record fold metrics.

Leakage control is the whole point of the ordering above.  The naive
alternative - fitting the kernel SVM on all training rows and feeding its
training-set scores to the transformer - makes the fusion token nearly
perfect on training data and worthless at test time.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from ..config import Config
from ..models.baselines import (
    fit_quantum_svm,
)
from ..models.transformer import (
    MultiOmicsTransformer,
    count_weights,
    focal_loss,
    parameter_count,
)
from ..quantum.kernels import FidelityQuantumKernel
from .evaluate import classification_metrics

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


# --------------------------------------------------------------------------
# Quantum branch
# --------------------------------------------------------------------------


def build_quantum_kernel(config: Config, n_input: int) -> FidelityQuantumKernel:
    return FidelityQuantumKernel(
        n_qubits=config.quantum.n_qubits,
        feature_map=config.quantum.feature_map,
        n_layers=config.quantum.n_layers,
        reupload=config.quantum.reupload,
        data_scale=config.quantum.data_scale,
        add_ry_rotation=config.quantum.add_ry_rotation,
        normalisation=config.quantum.kernel_normalisation,
    )


def quantum_fusion_feature(
    kernel: FidelityQuantumKernel,
    fused_train: np.ndarray,
    y_train: np.ndarray,
    fused_test: np.ndarray,
    *,
    c: float,
    inner_splits: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(train_score, test_score, diagnostics)`` for the fusion token.

    The training scores are out-of-fold with respect to an *inner* split, so
    they carry the same information content a test score would carry.  The
    test scores come from an SVM refitted on the full training set.
    """
    n = len(y_train)
    min_class = int(np.bincount(y_train).min())
    n_inner = max(2, min(inner_splits, min_class))

    train_score = np.zeros(n, dtype=float)
    if n_inner >= 2:
        skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        for inner_train, inner_val in skf.split(fused_train, y_train):
            sub = build_quantum_kernel_from(kernel)
            X_in, X_val = fused_train[inner_train], fused_train[inner_val]
            sub.fit(X_in, y_train[inner_train])
            # A precomputed-kernel SVC must be *fitted* on a square Gram
            # matrix; the rectangular block is only valid for prediction.
            model = fit_quantum_svm(
                sub.gram(X_in), y_train[inner_train], c=c, random_state=seed
            )
            train_score[inner_val] = _positive_score(model, sub.cross(X_val, X_in))

    # Final model on the full training set, used for the test block.
    kernel.fit(fused_train, y_train)
    k_test = kernel.cross(fused_test, fused_train)
    model = fit_quantum_svm(kernel.gram(fused_train), y_train, c=c, random_state=seed)
    test_score = _positive_score(model, k_test)

    diagnostics = {
        "inner_splits": n_inner,
        "fingerprint": kernel.fingerprint.to_dict() if kernel.fingerprint else {},
        "train_score_std": float(train_score.std()),
        "test_score_std": float(test_score.std()),
    }
    return train_score, test_score, diagnostics


def build_quantum_kernel_from(source: FidelityQuantumKernel) -> FidelityQuantumKernel:
    """Clone a kernel's hyper-parameters for an inner split."""
    return FidelityQuantumKernel(
        n_qubits=source.feature_map.n_qubits,
        feature_map=source.feature_map.feature_map,
        n_layers=source.feature_map.n_layers,
        reupload=source.feature_map.reupload,
        data_scale=source.feature_map.data_scale,
        add_ry_rotation=source.feature_map.add_ry_rotation,
        normalisation=source.normalisation,
    )


def _positive_score(model, block: np.ndarray) -> np.ndarray:
    """Decision value rescaled to ``[0, 1]`` for the advanced-stage class."""
    if hasattr(model.estimator, "predict_proba"):
        proba = model.estimator.predict_proba(block)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
    scores = np.asarray(model.estimator.decision_function(block), dtype=float).ravel()
    return 1.0 / (1.0 + np.exp(-scores))


# --------------------------------------------------------------------------
# Transformer training
# --------------------------------------------------------------------------


@dataclass
class FoldOutcome:
    """Everything one split produces."""

    name: str
    test_scores: np.ndarray
    test_index: np.ndarray
    metrics: dict
    seconds: float
    importance: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def _to_tensors(
    dense: dict[str, np.ndarray], names: dict[str, list[str]], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        block: torch.tensor(np.nan_to_num(matrix, nan=0.0), dtype=torch.float32, device=device)
        for block, matrix in dense.items()
    }


def train_transformer_fold(
    dense_train: dict[str, np.ndarray],
    dense_test: dict[str, np.ndarray],
    block_names: dict[str, list[str]],
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    fusion_train: np.ndarray | None,
    fusion_test: np.ndarray | None,
    config: Config,
    device: torch.device,
    seed: int,
    collect_importance: bool = True,
    importance_chunk: int = 16,
) -> FoldOutcome:
    """Train one transformer on one split and score the held-out rows."""
    start = time.perf_counter()
    set_seed(seed)

    block_sizes = {name: dense_train[name].shape[1] for name in dense_train}
    n_fusion = 0 if fusion_train is None else fusion_train.shape[1]

    model = MultiOmicsTransformer(
        block_sizes=block_sizes,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        n_classes=config.model.n_classes,
        n_fusion=n_fusion,
        max_tokens=config.model.max_tokens,
    ).to(device)

    X_tr = _to_tensors(dense_train, block_names, device)
    X_te = _to_tensors(dense_test, block_names, device)

    y_tr = torch.tensor(np.asarray(y_train, dtype=np.int64), device=device)

    n_train = len(y_tr)
    batch_size = min(config.train.batch_size, max(8, n_train // 4))
    steps_per_epoch = max(1, math.ceil(n_train / batch_size))
    total_steps = max(1, config.train.epochs * steps_per_epoch)
    warmup = max(1, int(total_steps * config.train.warmup_frac))

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: (step + 1) / warmup
        if step < warmup
        else 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))),
    )

    # Held-out slice of the training split for early stopping.
    n_val = max(16, int(0.15 * n_train))
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(n_train)
    val_idx, tr_idx = permutation[:n_val], permutation[n_val:]

    class_weight = count_weights(y_train).to(device) if config.model.class_weighting else None
    best_state: dict[str, torch.Tensor] | None = None
    best_auc, stale = -np.inf, 0
    history: list[dict] = []

    for epoch in range(config.train.epochs):
        model.train()
        order = generator.permutation(tr_idx)
        epoch_loss = 0.0
        for start_idx in range(0, len(order), batch_size):
            batch = order[start_idx : start_idx + batch_size]
            if len(batch) < 2:
                continue
            idx = torch.tensor(batch, device=device)

            fusion_batch = None
            if fusion_train is not None:
                fusion_batch = torch.tensor(
                    fusion_train[batch], dtype=torch.float32, device=device
                )

            out = model(
                {k: v[idx] for k, v in X_tr.items()}, fusion_batch
            )
            loss = focal_loss(
                out["logits"], y_tr[idx],
                label_smoothing=config.train.label_smoothing,
            )
            if class_weight is not None:
                loss = loss * (class_weight[y_tr[idx]].mean())

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimiser.step()
            scheduler.step()
            epoch_loss += float(loss.detach()) * len(batch)

        val_auc = _evaluate_auc(model, X_tr, y_tr, val_idx, fusion_train, device)
        history.append({"epoch": epoch, "loss": epoch_loss / max(1, len(order)), "val_auc": val_auc})
        if val_auc > best_auc + 1e-4:
            best_auc, stale = val_auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.train.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    importance = None
    with torch.no_grad():
        fusion_te = (
            torch.tensor(fusion_test, dtype=torch.float32, device=device)
            if fusion_test is not None
            else None
        )
        probs = np.empty(len(y_test), dtype=float)
        importance_chunks: list[np.ndarray] = []

        # Score and attribute in chunks.  A full-test-set forward pass
        # materialises an attention tensor of
        # (n_test, layers, heads, T, T) - for 93 samples and 193 tokens that
        # is over 100 MB per layer, which dominates peak memory for no
        # benefit.  Chunking keeps the peak at a few MB and produces
        # bit-identical results, since the rollout is computed per sample.
        n_chunks = len(y_test)
        for start in range(0, n_chunks, importance_chunk):
            stop = min(start + importance_chunk, n_chunks)
            slice_blocks = {k: v[start:stop] for k, v in X_te.items()}
            slice_fusion = fusion_te[start:stop] if fusion_te is not None else None
            out = model(
                slice_blocks, slice_fusion, need_attention=collect_importance
            )
            probs[start:stop] = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
            if collect_importance and "attention" in out:
                rollout = _rollout(out["attention"])  # (chunk, tokens)
                importance_chunks.append(rollout[:, 1:].cpu().numpy())

    if collect_importance and importance_chunks:
        importance = np.concatenate(importance_chunks, axis=0)

    metrics = classification_metrics(y_test, probs)
    return FoldOutcome(
        name="hybrid_transformer" if n_fusion else "classical_transformer",
        test_scores=probs,
        test_index=np.arange(len(y_test)),
        metrics=metrics,
        seconds=time.perf_counter() - start,
        importance=importance,
        extra={"best_val_auc": best_auc, "epochs_run": len(history), "parameters": parameter_count(model)},
    )


def _rollout(attention: torch.Tensor) -> torch.Tensor:
    """Attention rollout over layers, averaging heads.

    ``attention`` is ``(batch, layers, heads, tokens, tokens)``.
    """
    result = None
    n_tokens = attention.shape[-1]
    eye = torch.eye(n_tokens, device=attention.device, dtype=attention.dtype)
    for layer in range(attention.shape[1]):
        a = attention[:, layer].mean(dim=1)  # (B, T, T)
        a = a + eye.unsqueeze(0)
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        result = a if result is None else torch.bmm(a, result)
    return result[:, 0, :]


@torch.no_grad()
def _evaluate_auc(
    model: MultiOmicsTransformer,
    tensors: dict[str, torch.Tensor],
    y: torch.Tensor,
    index: np.ndarray,
    fusion: np.ndarray | None,
    device: torch.device,
) -> float:
    model.eval()
    idx = torch.tensor(index, device=device)
    fusion_batch = (
        torch.tensor(fusion[index], dtype=torch.float32, device=device) if fusion is not None else None
    )
    out = model({k: v[idx] for k, v in tensors.items()}, fusion_batch)
    scores = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
    labels = y[idx].cpu().numpy()
    if len(np.unique(labels)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


__all__ = [
    "set_seed",
    "resolve_device",
    "build_quantum_kernel",
    "quantum_fusion_feature",
    "train_transformer_fold",
    "FoldOutcome",
]
