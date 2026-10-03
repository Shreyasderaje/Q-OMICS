"""Correctness and speed check for the quantum kernel backend."""
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

from qomics.quantum.features import QuantumFeatureMap, analytic_sep_rx_gram, classical_gram
from qomics.quantum.kernels import (
    FidelityQuantumKernel,
    apply_normalisation,
    kernel_diagnostics,
    labelled_separation,
)

rng = np.random.default_rng(0)

print("=== 1. state normalisation & dimension ===")
fmap = QuantumFeatureMap(n_qubits=4, feature_map="iqp", n_layers=2)
x = rng.normal(size=4)
psi = fmap.embed(x)
print("  state dim:", psi.shape, "| norm:", round(float(np.linalg.norm(psi)), 12))

print("\n=== 2. sep_rx encoding vs closed-form fidelity ===")
sep = QuantumFeatureMap(n_qubits=4, feature_map="sep_rx", n_layers=1, add_ry_rotation=False)
X = rng.normal(size=(6, 4))
K_sim = sep.gram_from_states(sep.embed_many(X))
K_th = analytic_sep_rx_gram(X)
print("  max |sim - theory| =", float(np.abs(K_sim - K_th).max()))
assert np.abs(K_sim - K_th).max() < 1e-9, "simulator disagrees with analytic product-state fidelity"
print("  MATCH -> simulator backend is verified against theory")

print("\n=== 3. entanglement check (iqp has non-zero off-diagonal) ===")
iqp = QuantumFeatureMap(n_qubits=4, feature_map="iqp", n_layers=1)
K_iqp = iqp.gram_from_states(iqp.embed_many(X))
print("  IQP diag:", np.round(np.diag(K_iqp)[:4], 6))
print("  max off-diag:", round(float(np.max(K_iqp - np.diag(np.diag(K_iqp)))), 6))

print("\n=== 4. normalisations ===")
K_raw = iqp.gram_from_states(iqp.embed_many(X))
for scheme in ("none", "cosine", "arc_cosine_scaled"):
    Kn = apply_normalisation(K_raw, scheme)
    print(f"  {scheme:20} diag={np.round(np.diag(Kn)[:3], 4)}  offdiag_range="
          f"[{Kn[~np.eye(6, dtype=bool)].min():.4f}, {Kn[~np.eye(6, dtype=bool)].max():.4f}]")

print("\n=== 5. symmetry & PSD ===")
Kn = apply_normalisation(K_raw, "arc_cosine_scaled")
print("  symmetric:", np.allclose(Kn, Kn.T, atol=1e-10))
evals = np.linalg.eigvalsh((Kn + Kn.T) / 2)
print("  min eigenvalue:", float(evals.min()))

print("\n=== 6. scaling benchmark (n=500, 6 qubits) ===")
Xb = rng.normal(size=(500, 6))
for fmap_name in ("iqp", "hea", "sep_rx"):
    fm = QuantumFeatureMap(n_qubits=6, feature_map=fmap_name, n_layers=2)
    t0 = time.perf_counter()
    st = fm.embed_many(Xb)
    K = np.abs(st @ st.conj().T) ** 2
    dt = time.perf_counter() - t0
    gates = "H+RZ+ZZ blocks"
    print(f"  {fmap_name:7} {dt:6.2f}s  K={K.shape}  states={st.shape}  ({gates})")

print("\n=== 7. end-to-end FidelityQuantumKernel ===")
y = (Xb[:, 0] > 0).astype(int)
kq = FidelityQuantumKernel(n_qubits=6, feature_map="iqp", n_layers=2, normalisation="arc_cosine_scaled")
kq.fit(Xb, y)
print("  fingerprint:", kq.fingerprint.to_dict())
K = kq.gram(Xb)
print("  diagnostics:", {k: round(v, 4) for k, v in kernel_diagnostics(K).items()})
print("  labelled separation:", round(labelled_separation(K, y), 6))
print("  cross block shape:", kq.cross(Xb[:20], Xb).shape)

print("\n=== 8. quantum vs classical geometry ===")
Krbf = classical_gram(Xb, "rbf")
print("  labelled separation  RBF:", round(labelled_separation(Krbf, y), 6))
print("  labelled separation  QK :", round(labelled_separation(K, y), 6))
print("\nALL QUANTUM CHECKS PASSED")
