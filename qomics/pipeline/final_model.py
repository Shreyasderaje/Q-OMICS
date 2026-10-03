"""Final model fitting and artefact persistence.

Cross-validation answers "how well does this generalise".  The serving layer
additionally needs a *deployed* model: one fitted on every available sample,
with the exact preprocessor, quantum kernel and encoder weights saved
together so that a prediction at inference time sees the same feature
geometry the model was trained on.

Keeping the three in a single bundle is not optional.  A quantum kernel
depends on the PCA basis it was fitted against, and the encoder depends on
the variance-filtered feature order; desynchronise any one of them and the
model silently returns confident nonsense.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..config import CONFIG, Config
from ..models.baselines import fit_quantum_svm
from ..models.transformer import MultiOmicsTransformer, count_weights, focal_loss
from .preprocess import MultiOmicsPreprocessor
from .train import _positive_score, build_quantum_kernel, resolve_device, set_seed

log = logging.getLogger(__name__)

BUNDLE_NAME = "model_bundle.pt"
RUN_NAME = "run.json"
BIOMARKERS_NAME = "biomarkers.csv"
COHORT_NAME = "cohort_clinical.csv"


@dataclass
class ModelBundle:
    """Everything required to score a new sample."""

    preprocessor: MultiOmicsPreprocessor
    transformer: MultiOmicsTransformer
    kernel: Any
    svm: Any
    feature_meta: pd.DataFrame
    training_medians: dict[str, float]
    training_means: dict[str, float]
    training_stds: dict[str, float]
    feature_order: list[str] = field(default_factory=list)
    block_order: list[str] = field(default_factory=list)
    block_slices: dict[str, tuple[int, int]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    #: Fused training vectors, saved with the bundle.  A test-time kernel row
    #: is ``K[x, X_train]``, so these are what the SVM was fitted against.
    _fused_reference: np.ndarray | None = field(default=None, repr=False)

    # -- scoring -----------------------------------------------------------

    def score(self, sample: dict[str, float]) -> dict[str, Any]:
        """Score a sparse feature dictionary.

        Unknown feature names are ignored and missing ones are imputed from
        the training median, so a caller supplying a handful of values still
        receives a defined prediction rather than an error.
        """
        per_block = self._dense_from_sparse(sample)
        fused = self.preprocessor.fuse_from_dense(per_block)
        if fused.shape[0] != 1:
            fused = fused[:1]

        k_test = self.kernel.cross(fused, self._fused_reference)
        q_score = float(_positive_score(self.svm, k_test)[0])

        self.transformer.eval()
        with torch.no_grad():
            device = next(self.transformer.parameters()).device
            blocks = {
                name: torch.tensor(per_block[name], dtype=torch.float32, device=device)
                for name in self.block_order
            }
            fusion = torch.tensor([[q_score]], dtype=torch.float32, device=device)
            out = self.transformer(blocks, fusion, need_attention=True)
            probs = torch.softmax(out["logits"], dim=-1)[0].cpu().numpy()
            importance = self.transformer.token_importance(out["attention"])[0].cpu().numpy()

        contributors = self._contributors(sample, importance)
        return {
            "probabilities": {
                "stage_I_early": float(probs[0]),
                "stage_II_IV_advanced": float(probs[1]),
            },
            "confidence": float(probs[1]),
            "prediction": "advanced_stage" if probs[1] >= 0.5 else "early_stage",
            "label": "Stage II-IV (advanced)" if probs[1] >= 0.5 else "Stage I (early)",
            "risk_tier": _risk_tier(float(probs[1])),
            "quantum_score": q_score,
            "top_contributors": contributors,
        }

    # -- helpers -----------------------------------------------------------

    def _dense_from_sparse(self, sample: dict[str, float]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name in self.block_order:
            width = len(self.preprocessor.selected[name])
            row = np.zeros(width, dtype=float)
            for j, column in enumerate(self.preprocessor.selected[name]):
                for key in (column, f"{name}::{column}", column.split("::")[-1]):
                    if key in sample:
                        value = sample[key]
                        if value is None or (isinstance(value, float) and np.isnan(value)):
                            continue
                        row[j] = float(value)
                        break
                else:
                    row[j] = self.training_medians.get(column, 0.0)
            out[name] = row[None, :]
        return out

    def _contributors(self, sample: dict[str, float], importance: np.ndarray) -> list[dict]:
        supplied = {
            key.split("::")[-1]: value for key, value in sample.items()
            if isinstance(value, (int, float)) and np.isfinite(value)
        }
        rows: list[dict] = []
        for block in self.block_order:
            start, stop = self.block_slices.get(block, (0, 0))
            for j, column in enumerate(self.preprocessor.selected[block]):
                symbol = column.split("::")[-1]
                rows.append(
                    {
                        "name": column,
                        "symbol": symbol,
                        "block": block,
                        "importance": float(importance[start + j]) if start + j < len(importance) else 0.0,
                        "value": float(supplied.get(symbol, self.training_medians.get(column, 0.0))),
                        "supplied": symbol in supplied,
                    }
                )
        rows.sort(key=lambda r: r["importance"], reverse=True)
        return rows[:10]

    def save(self, path: Path) -> Path:
        """Persist the complete serving bundle.

        The quantum-kernel SVM and the kernel's training-set state norms are
        saved alongside the weights.  Omitting them is not a size
        optimisation: at serve time the fusion score is computed as
        ``SVC.decision_function(K[x, X_train])``, and without the fitted
        support vectors and the reference norms there is no way to produce
        that row, so scoring degrades to a 503.
        """
        path = Path(path)
        payload = {
            "state_dict": self.transformer.state_dict(),
            # The checkpoint must be self-describing.  Rebuilding the encoder
            # from the *current* CONFIG means a default changed between
            # training and serving turns every weight into a shape mismatch.
            "model_config": {
                "d_model": self.transformer.d_model,
                "block_sizes": dict(self.transformer.block_sizes),
                "n_fusion": self.transformer.n_fusion,
                "n_layers": len(self.transformer.layers),
                "n_heads": self.transformer.layers[0].n_heads if self.transformer.layers else 4,
                "dim_feedforward": (
                    self.transformer.layers[0].ff[0].out_features
                    if self.transformer.layers
                    else None
                ),
                "n_classes": self.transformer.head[3].out_features
                if len(self.transformer.head) > 3
                else 2,
            },
            "block_order": self.block_order,
            "block_slices": self.block_slices,
            "feature_order": self.feature_order,
            "selected": self.preprocessor.selected,
            "reducers": {
                name: {
                    "mean": r.scaler.mean_,
                    "scale": r.scaler.scale_,
                    "components": r.pca.components_,
                    # PCA centres after projecting in current scikit-learn
                    # and needs its own mean vector; omitting it makes
                    # transform() fail on a restored bundle.
                    "pca_mean": getattr(r.pca, "mean_", None),
                    "n_samples": getattr(r.pca, "n_samples_", None),
                    "explained_variance": getattr(r.pca, "explained_variance_", None),
                    "weight": r.weight,
                }
                for name, r in self.preprocessor.reducers.items()
            },
            "training_medians": self.training_medians,
            "training_means": self.training_means,
            "training_stds": self.training_stds,
            "metadata": self.metadata,
            "svm": self.svm,
            "kernel_reference_norm": getattr(self.kernel, "_reference_norm", None),
            "fused_reference": getattr(self, "_fused_reference", None),
            "quantum": {
                "n_qubits": self.kernel.feature_map.n_qubits,
                "feature_map": self.kernel.feature_map.feature_map,
                "n_layers": self.kernel.feature_map.n_layers,
                "reupload": self.kernel.feature_map.reupload,
                "data_scale": self.kernel.feature_map.data_scale,
                "add_ry_rotation": self.kernel.feature_map.add_ry_rotation,
                "normalisation": self.kernel.normalisation,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def _risk_tier(score: float) -> str:
    if score < 0.35:
        return "low"
    if score < 0.65:
        return "moderate"
    return "elevated"


def train_final_model(
    preprocessor: MultiOmicsPreprocessor,
    cohort_blocks: dict[str, pd.DataFrame],
    y: np.ndarray,
    config: Config | None = None,
    *,
    epochs: int | None = None,
) -> ModelBundle:
    """Fit the deployable model on every sample in the cohort."""
    cfg = config or CONFIG
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    dense, fused = preprocessor.transform(None)
    block_order = list(preprocessor.block_names)

    kernel = build_quantum_kernel(cfg, fused.shape[1])
    kernel.fit(fused, y)
    svm = fit_quantum_svm(
        kernel.gram(fused), y, c=cfg.quantum.c_regularisation,
        random_state=cfg.train.seed,
    )
    k_self = kernel.cross(fused, fused)
    fusion_feature = _positive_score(svm, k_self)[:, None]

    block_sizes = {name: dense[name].shape[1] for name in block_order}
    model = MultiOmicsTransformer(
        block_sizes=block_sizes,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        n_classes=cfg.model.n_classes,
        n_fusion=1,
        max_tokens=cfg.model.max_tokens,
    ).to(device)

    X = {name: torch.tensor(np.nan_to_num(dense[name], nan=0.0), dtype=torch.float32, device=device)
         for name in block_order}
    y_t = torch.tensor(np.asarray(y, dtype=np.int64), device=device)
    fusion_t = torch.tensor(fusion_feature, dtype=torch.float32, device=device)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )
    n_epochs = epochs or cfg.train.epochs
    batch_size = min(cfg.train.batch_size, len(y))
    weights = count_weights(y).to(device) if cfg.model.class_weighting else None
    generator = np.random.default_rng(cfg.train.seed)

    started = time.perf_counter()
    model.train()
    for epoch in range(n_epochs):
        order = generator.permutation(len(y))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            if len(batch) < 2:
                continue
            idx = torch.tensor(batch, device=device)
            out = model({k: v[idx] for k, v in X.items()}, fusion_t[idx])
            loss = focal_loss(out["logits"], y_t[idx], label_smoothing=cfg.train.label_smoothing)
            if weights is not None:
                loss = loss * weights[y_t[idx]].mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimiser.step()
        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            log.info("final model epoch %d/%d (%.1fs)", epoch + 1, n_epochs, time.perf_counter() - started)

    # Training-set statistics, used to impute unseen features at serve time.
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in block_order:
        frame = cohort_blocks[name][preprocessor.selected[name]]
        med = frame.median(numeric_only=True)
        for column in preprocessor.selected[name]:
            value = float(med.get(column, np.nan))
            medians[column] = value if np.isfinite(value) else 0.0
        means[name] = float(np.nanmean(dense[name]))
        stds[name] = float(np.nanstd(dense[name]))

    block_slices: dict[str, tuple[int, int]] = {}
    cursor = 1
    for name in block_order:
        width = block_sizes[name]
        block_slices[name] = (cursor, cursor + width)
        cursor += width

    bundle = ModelBundle(
        preprocessor=preprocessor,
        transformer=model,
        kernel=kernel,
        svm=svm,
        feature_meta=cohort_blocks[block_order[0]].head(0),
        training_medians=medians,
        training_means=means,
        training_stds=stds,
        feature_order=[c for name in block_order for c in preprocessor.selected[name]],
        block_order=block_order,
        block_slices=block_slices,
        metadata={
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epochs": n_epochs,
            "device": str(device),
            "target": "FIGO stage I vs stage II-IV",
            "n_samples": int(len(y)),
            "seconds": round(time.perf_counter() - started, 1),
        },
    )
    # The bundle needs the fused reference matrix to build a test-time kernel.
    bundle._fused_reference = fused
    return bundle


__all__ = ["ModelBundle", "train_final_model", "BUNDLE_NAME", "RUN_NAME", "BIOMARKERS_NAME", "COHORT_NAME"]
