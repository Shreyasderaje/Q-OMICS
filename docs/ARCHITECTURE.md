# Architecture

How a cohort of multi-omics measurements becomes a ranked biomarker panel,
and where each component lives in the repository.

```
                      ┌──────────────────────────────────────────┐
                      │  Public archives (open access, no auth)  │
                      │  cBioPortal · UCSC Xena · TCGA           │
                      └───────────────────┬──────────────────────┘
                                          │
                          qomics/data/{cbioportal,xena,cohort}.py
                                          │
        ┌───────────────┬─────────────────┼─────────────────┬──────────────────┐
        ▼               ▼                 ▼                 ▼                  │
   GENOME 741     TRANSCRIPTOME      PROTEOME 189      CLINICAL               │
   MUT:: / CNA::  gene z-scores      RPPA antibodies    FIGO stage            │
   TMB · FGA      (375-gene panel)   (variance-filtered) age · histology      │
        │               │                 │                 │                  │
        └───────────────┴─────────────────┴─────────────────┘                  │
                                          │                                   │
                     qomics/pipeline/preprocess.py                            │
                     · variance filter      · per-fold scaler                 │
                     · per-modality PCA     · modality weighting             │
                                          │                                   │
                    ┌─────────────────────┴─────────────────────┐             │
                    ▼                                           ▼             │
        dense tokens (192)                          fused vector (12)           │
                    │                                           │             │
                    │                    qomics/quantum/{features,kernels}.py │
                    │                    · IQP encoding, 6 qubits            │
                    │                    · fidelity kernel K = |ΨΨ†|²        │
                    │                    · arc-cosine normalisation          │
                    │                                           │             │
                    │                          Quantum SVM ────┤             │
                    │                            (fusion score) │             │
                    ▼                                           ▼             │
        ┌───────────────────────────────────────────────────────────┐        │
        │  qomics/models/transformer.py                             │        │
        │  · per-modality scalar → d_model lift                    │        │
        │  · [CLS] token + modality embeddings + fusion token      │        │
        │  · 2 encoder layers × 4 heads                            │        │
        │  · head([CLS] ‖ mean-pool) → stage class                 │        │
        └───────────────┬───────────────────────┬───────────────────┘        │
                        │                       │                            │
       attention rollout │                       │ classification           │
                        ▼                       ▼                            │
        ┌───────────────────────────┐   ┌────────────────────┐              │
        │ qomics/pipeline/          │   │ qomics/pipeline/   │              │
        │ biomarkers.py             │   │ evaluate.py        │              │
        │ · rollout importance      │   │ · grouped 5×2 CV   │              │
        │ · stability across folds  │   │ · paired t-test    │              │
        │ · Cohen's d, perm. p      │   │ · Holm-Bonferroni  │              │
        │ · Benjamini-Hochberg FDR  │   │ · bootstrap CI      │              │
        └───────────────┬───────────┘   └─────────┬──────────┘              │
                        └────────────┬─────────────┘                        │
                                     ▼                                      │
                    qomics/pipeline/final_model.py                         │
                    single bundle: preprocessor + kernel + SVM + encoder    │
                                     │                                      │
                                     ▼                                      │
                    qomics/serving/{app,store,schemas}.py                  │
                    FastAPI → JSON → qomics/web (static SPA)                │
                     └─────────────────────────────────────────────────────┘
```

---

## Module map

| Module | Responsibility | Depends on |
|---|---|---|
| `qomics/config.py` | Every tunable, in one dataclass tree | — |
| `qomics/genes.py` | Curated 375-symbol panel, organised by pathway axis | — |
| `qomics/data/cbioportal.py` | cBioPortal REST client, disk cache, concurrent fetch | `config` |
| `qomics/data/xena.py` | Xena download, RPPA loading, sample-type mapping | `config` |
| `qomics/data/cohort.py` | Joins layers on the sample barcode, parses FIGO stage | both clients, `genes` |
| `qomics/quantum/features.py` | IQP / HEA / product-state feature maps | PennyLane |
| `qomics/quantum/kernels.py` | Fidelity kernel, normalisation, diagnostics, persistence | `features` |
| `qomics/models/transformer.py` | Encoder with attention capture, focal loss | torch |
| `qomics/models/baselines.py` | Classical reference models | scikit-learn |
| `qomics/pipeline/preprocess.py` | Fold-safe selection, scaling, PCA, fused vector | scikit-learn |
| `qomics/pipeline/geometry.py` | Geometric difference, CKA, entanglement ablation | `quantum` |
| `qomics/pipeline/train.py` | Single-split quantum + transformer training | above |
| `qomics/pipeline/evaluate.py` | Metrics, bootstrap, paired t-test, multiplicity | scikit-learn |
| `qomics/pipeline/biomarkers.py` | Panel assembly, effect sizes, FDR | above |
| `qomics/pipeline/experiment.py` | Orchestrates the full protocol, writes artefacts | all of the above |
| `qomics/pipeline/final_model.py` | Deployable bundle fitted on the whole cohort | above |
| `qomics/serving/store.py` | Lazy, failure-tolerant artefact loading | `final_model` |
| `qomics/serving/app.py` | FastAPI endpoints, JSON sanitisation, static mount | `store` |
| `qomics/web/` | Static single-page frontend, no build step | API contract only |

---

## Data flow invariants

Four properties the code actively defends, each with a test:

**Fold containment.** Scaler, variance filter, PCA and the quantum
embedding are all fitted on training rows only. `tests/test_data.py::
test_fold_safe_preprocessing_keeps_train_and_test_aligned` covers the failure
where the fused vector is computed over the whole cohort while the token
matrices come from a fold.

**No aliasing between splits.** `transform()` returns fresh arrays on every
call. Returning the cached dict silently made the test matrix overwrite the
train matrix, which surfaced as an unrelated length error inside a baseline
that never touched the quantum code.

**Stacking without leakage.** The fusion token fed to the transformer during
training is out-of-fold with respect to an inner split of the training set.
The test-time score comes from an SVM refitted on the full training fold.

**Cross-kernel scale.** A test kernel row is normalised by the *training*
state norms, so it lives in the same geometry as the training Gram matrix.
Asserted by duplicating a training point and requiring it to reproduce its
own row to 1e-8.

---

## The quantum backend, precisely

```
inputs        x in R^12  (6 per transcriptome PCA, 4 genome, 2 proteome)
encoding      |psi(x)> = U_IQP(x)^{repeats} RY(x/2) |0...0>
register      6 qubits -> 64-dimensional complex state
kernel        K_ij = |<psi(x_i)|psi(x_j)>|^2
evaluation    Psi = embed(X) in C^(n x 64);  K = |Psi Psi^dagger|^2
normalise     K' = (2/pi) arcsin( K / diag(K)_train )
consumer      SVC(kernel="precomputed") -> fusion score -> transformer token
```

**Why the matrix product is legitimate.** Running the compute-uncompute
circuit `U(x)` then `U(x')†` and reading the all-zero outcome probability
yields exactly `|<psi(x')|psi(x)>|^2`. When the simulator exposes the full
wave-function, that same quantity is the squared modulus of a matrix product.
The two agree to machine precision; the test suite asserts the special case
against closed-form product-state fidelity to 1e-9.

**Cost.** Two independent optimisations, both verified to leave the output
unchanged:

| Optimisation | Before | After | Speed-up |
|---|---|---|---|
| State-vector caching instead of `n^2` circuits | `O(n^2)` sims | `O(n)` sims | ~2 orders of magnitude |
| Broadcast batching instead of a Python loop | 14.5 s (n=500) | 0.06 s | ~240× |

---

## Configuration

Every number lives in `qomics/config.py`. The settings that matter most, and
the reasoning:

| Setting | Value | Why |
|---|---|---|
| `quantum.n_qubits` | 6 | 64-dim Hilbert space; the regime where fidelity kernels are expected to diverge from classical ones. Doubling width is the first experiment worth running next. |
| `quantum.feature_map` | `iqp` | Degree-2 polynomial features; all-to-all `ZZ` entanglement. `sep_rx` is kept as a non-entangled control. |
| `quantum.kernel_normalisation` | `arc_cosine_scaled` | Raw fidelity Gram matrices go diagonally dominated with depth, which starves the SVM of contrast. |
| `preprocess.fit_within_fold` | `True` | Leakage control. |
| `model.d_model` | 64 | 433k parameters on 545 samples overfits; 76k does not, and trains 6× faster. |
| `model.max_tokens` | 256 | Attention is quadratic in sequence length. The realised sequence is 192 tokens (80 transcriptome / 72 genome / 40 proteome) plus the fusion token. |
| `train.n_repeats` | 2 | 10 paired folds. Enough for the t-test, and a full sweep completes in minutes rather than hours. |
| `train.epochs` / `patience` | 40 / 12 | Early stopping on validation AUC. |

---

## Extending

**A new cohort.** Change `CohortConfig`. If the target ontology is not FIGO,
add a parser alongside `parse_figo_stage`.

**A new feature map.** Implement `_block_*` in `QuantumFeatureMap` and add the
name to `SUPPORTED`. Include a closed-form check if the map factorises, so
the simulator can be verified.

**A new modality.** Add a block to `Cohort.blocks`; the preprocessor, the
transformer's per-modality lift and the biomarker annotation all pick it up
automatically. Budget tokens for it in `TOKEN_BUDGET`.

**Hardware backends.** `FidelityQuantumKernel` takes a PennyLane device. Swap
`default.qubit` for a noisy or hardware backend and the interface is unchanged
— but expect the kernel quality to drop, and the geometry report will show it.
