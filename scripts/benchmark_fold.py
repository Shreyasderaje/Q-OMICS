"""Single-fold benchmark: verify the pipeline runs and measure per-fold cost."""
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

from qomics.config import CONFIG
from qomics.data.cohort import build_cohort
from qomics.models.baselines import fit_baseline, fit_quantum_svm
from qomics.models.transformer import parameter_count
from qomics.pipeline.evaluate import classification_metrics
from qomics.pipeline.experiment import _make_preprocessor
from qomics.pipeline.preprocess import grouped_splits
from qomics.pipeline.train import (
    _positive_score,
    build_quantum_kernel,
    quantum_fusion_feature,
    resolve_device,
    set_seed,
    train_transformer_fold,
)

cohort = build_cohort()
print(f"cohort {cohort.n_samples} samples | blocks {cohort.n_features} | balance {cohort.y.mean():.3f}")

y, groups = cohort.y, np.asarray(cohort.patient_ids)
splits = grouped_splits(y, groups, 5, 1, CONFIG.train.seed)
repeat, train_idx, test_idx, fold = splits[0]
print(f"fold: train={len(train_idx)} test={len(test_idx)} "
      f"pos_rate_train={y[train_idx].mean():.3f} test={y[test_idx].mean():.3f}")

device = resolve_device("auto")
set_seed(CONFIG.train.seed)
print("device:", device, "| torch threads:", torch.get_num_threads())

# --- preprocessing
t0 = time.perf_counter()
pre = _make_preprocessor(cohort.blocks, CONFIG)
dense_tr, fused_tr = pre.fit_transform(train_idx)
dense_te, fused_te = pre.transform(test_idx)
print(f"preprocess {time.perf_counter()-t0:.2f}s | token counts "
      f"{ {k: v.shape[1] for k, v in dense_tr.items()} } | fused {fused_tr.shape}")
print("  total tokens:", sum(v.shape[1] for v in dense_tr.values()), "/ max", CONFIG.model.max_tokens)

# --- quantum
t0 = time.perf_counter()
kernel = build_quantum_kernel(CONFIG, fused_tr.shape[1])
ftr, fte, diag = quantum_fusion_feature(
    kernel, fused_tr, y[train_idx], fused_te,
    c=CONFIG.quantum.c_regularisation, inner_splits=3, seed=0,
)
print(f"quantum branch {time.perf_counter()-t0:.2f}s | fusion train std {ftr.std():.3f} test std {fte.std():.3f}")
q_model = fit_quantum_svm(kernel.gram(fused_tr), y[train_idx], c=1.0)
q_scores = _positive_score(q_model, kernel.cross(fused_te, fused_tr))
m = classification_metrics(y[test_idx], q_scores)
print(f"  Quantum SVM AUC {m['auc']:.4f} | sens {m['sensitivity']:.3f} spec {m['specificity']:.3f}")

# --- baselines
Xtr = np.concatenate(list(dense_tr.values()), axis=1)
Xte = np.concatenate(list(dense_te.values()), axis=1)
print(f"  dense features: {Xtr.shape}")
for name in ("logreg", "rbf_svm", "random_forest", "hist_gb"):
    t0 = time.perf_counter()
    b = fit_baseline(name, Xtr, y[train_idx], Xte, CONFIG.baseline)
    s = b.predict_proba(Xte)[:, 1]
    mm = classification_metrics(y[test_idx], s)
    print(f"  {name:14} {time.perf_counter()-t0:5.2f}s  AUC {mm['auc']:.4f}")

# --- transformers
dense_names = {k: pre.selected[k] for k in pre.block_names}
from qomics.models.transformer import MultiOmicsTransformer

probe = MultiOmicsTransformer(
    {k: v.shape[1] for k, v in dense_tr.items()}, d_model=CONFIG.model.d_model,
    n_heads=CONFIG.model.n_heads, n_layers=CONFIG.model.n_layers,
    dim_feedforward=CONFIG.model.dim_feedforward, dropout=CONFIG.model.dropout,
    n_fusion=1, max_tokens=CONFIG.model.max_tokens,
)
print(f"  transformer params: {parameter_count(probe):,}")

for label, ftr_arg, fte_arg in (
    ("hybrid", ftr[:, None], fte[:, None]),
    ("classical", None, None),
):
    t0 = time.perf_counter()
    out = train_transformer_fold(
        dense_tr, dense_te, dense_names, y[train_idx], y[test_idx],
        fusion_train=ftr_arg, fusion_test=fte_arg,
        config=CONFIG, device=device, seed=7,
    )
    mm = out.metrics
    print(f"  {label:10} transformer {time.perf_counter()-t0:6.2f}s  AUC {mm['auc']:.4f}  "
          f"epochs {out.extra['epochs_run']} best_val {out.extra['best_val_auc']:.4f}  "
          f"importance {None if out.importance is None else out.importance.shape}")
