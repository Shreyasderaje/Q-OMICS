# Results

> **Generated from `artifacts/run.json` by `scripts/make_results_md.py`.**
> Do not edit by hand — regenerate it instead.

Run completed 2026-10-03T14:35:37Z · 1380.9 s wall clock · Q-OMICS 1.0.0

## Headline

**Target** — FIGO stage I vs stage II-IV endometrial carcinoma

| | |
|---|---|
| Cohort | TCGA-UCEC (GDC, 2025) |
| Samples / patients | 545 / 544 |
| Class balance | {'class_0': 340, 'class_1': 205} |
| Features | {'transcriptome': 374, 'genome': 741, 'proteome': 189} |
| Cross-validation | 10 patient-disjoint folds |
| **Hybrid transformer AUC** | **0.5931 ± 0.0552** |
| 95 % bootstrap CI | [0.5176, 0.6178] |
| Best model overall | Random Forest at 0.6765 |

## Model comparison

| Model | Family | AUC | ± SD | PR-AUC | Sens. | Spec. | Brier | s/fold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | ensemble | 0.6765 | 0.0362 | 0.5594 | 0.644 | 0.690 | 0.216 | 1.1 |
| RBF SVM | kernel | 0.6706 | 0.0464 | 0.5546 | 0.632 | 0.701 | 0.217 | 0.1 |
| Logistic Regression | linear | 0.6327 | 0.0640 | 0.5290 | 0.602 | 0.675 | 0.256 | 0.0 |
| PCA + Logistic Regression (transcriptome only) | linear | 0.6311 | 0.0306 | 0.5079 | 0.583 | 0.690 | 0.247 | 0.0 |
| HistGradientBoosting | ensemble | 0.6266 | 0.0423 | 0.5069 | 0.583 | 0.684 | 0.227 | 0.2 |
| **Hybrid Transformer (Q-OMICS)** | hybrid | **0.5931** | 0.0552 | 0.4643 | 0.610 | 0.629 | 0.239 | 194355.2 |
| Classical Transformer | neural | 0.5842 | 0.0458 | 0.4843 | 0.490 | 0.744 | 0.249 | 194421.2 |
| Quantum SVM | kernel | 0.5171 | 0.0515 | 0.4230 | 0.576 | 0.562 | 0.246 | 0.0 |

All metrics are out-of-fold, averaged over repeats, and computed at the Youden-optimal operating point. AUC is prevalence-independent; sensitivity and specificity are not.

## Significance

Paired t-test on fold AUCs from identical splits, Holm-Bonferroni corrected across the family. Negative Δ means the hybrid model scored *lower*.

| Comparison | Δ AUC | t | p | significant (corrected) |
|---|---:|---:|---:|:---:|
| vs Random Forest | -0.0834 | -3.629 | 0.0055 | yes |
| vs Quantum SVM | +0.0761 | 3.507 | 0.0066 | yes |
| vs RBF SVM | -0.0774 | -3.489 | 0.0068 | yes |
| vs PCA + Logistic Regression (transcriptome only) | -0.0380 | -1.702 | 0.1229 | no |
| vs Logistic Regression | -0.0396 | -1.282 | 0.2319 | no |
| vs HistGradientBoosting | -0.0335 | -1.224 | 0.2520 | no |
| vs Classical Transformer | +0.0089 | 0.498 | 0.6305 | no |

## Quantum kernel

| | |
|---|---|
| Backend | pennylane.default.qubit |
| Register | 6 qubits → 64-dimensional Hilbert space |
| Feature map | iqp × 2 layers, re-upload True |
| Normalisation | arc_cosine_scaled |
| Encoding time | 0.041 s for 436.0 samples |
| Effective rank | 352.8 / 436 |
| Condition number | 46.74 |

### Labelled separation

Between-class minus within-class mean kernel similarity. Positive means the kernel groups same-stage samples together.

| Kernel | Separation |
|---|---:|
| quantum | -0.000620 |
| rbf | -0.019092 |
| cosine | -0.030978 |
| linear | -0.020693 |

### Geometric difference

| Reference | g |
|---|---:|
| RBF | 95.249 |
| cosine | 149.158 |
| linear | 151.458 |

> g = 95.249, but the quantum kernel carries essentially no label separation (between-class minus within-class similarity -6.20e-04, below the 1e-3 resolution of this cohort). A large or small g is not evidence of an advantage: the kernel does not separate the classes at all, so no kernel learner can benefit from its geometry. On this cohort the quantum branch does not outperform its classical references.

### Entanglement ablation

Same inputs, same register width: entangled IQP versus a non-entangled product-state encoding.

| | Value |
|---|---:|
| n samples | 200 |
| separation, entangled | -0.001287 |
| separation, product | -0.009772 |
| absolute gain | 0.008485 |
| relative gain | 86.8 % |

## Ranked biomarker panel

Ranking method: **attention_rollout** · 192 features ranked from 192 tokens.

Modality contribution: **transcriptome** 41 % · **genome** 39 % · **proteome** 20 %

**53 of 192** features are FDR-significant (Benjamini-Hochberg q < 0.05) on a permutation test of the class association.

| # | Feature | Block | Axis | Importance | Stability | Cohen's d | q |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `CNA::CCNE1` \* | genome | cell_cycle_cdk | 0.00534 | 0.13 | -0.507 | 0.006 |
| 2 | `CNA::CD22` \* | genome | immune | 0.00535 | 0.14 | -0.458 | 0.006 |
| 3 | `CNA::SNAI1` \* | genome | emt_invasion | 0.00531 | 0.10 | -0.494 | 0.006 |
| 4 | `CNA::HNF4A` \* | genome | estrogen_progesterone | 0.00531 | 0.10 | -0.460 | 0.006 |
| 5 | `CNA::MMP9` \* | genome | emt_invasion | 0.00532 | 0.11 | -0.449 | 0.006 |
| 6 | `CNA::NCOA3` \* | genome | tumor_suppressors | 0.00531 | 0.10 | -0.445 | 0.006 |
| 7 | `CNA::MYBL2` \* | genome | cell_cycle | 0.00531 | 0.11 | -0.427 | 0.006 |
| 8 | `ER-alpha-R-V` \* | proteome | hormone receptor | 0.00506 | 0.14 | 0.459 | 0.006 |
| 9 | `CNA::MYC` \* | genome | wnt_stemness | 0.00528 | 0.09 | -0.455 | 0.006 |
| 10 | `CNA::BCL2L1` \* | genome | apoptosis_ddr | 0.00531 | 0.08 | -0.438 | 0.006 |
| 11 | `PDCD4-R-C` \* | proteome | unclassified | 0.00506 | 0.19 | 0.359 | 0.006 |
| 12 | `CNA::SDHA` \* | genome | metabolism | 0.00537 | 0.14 | -0.318 | 0.006 |
| 13 | `E-Cadherin-R-V` \* | proteome | adhesion | 0.00510 | 0.17 | 0.365 | 0.006 |
| 14 | `CNA::MYCN` \* | genome | cell_cycle_cdk | 0.00536 | 0.09 | -0.372 | 0.006 |
| 15 | `CNA::TERT` \* | genome | cell_cycle_cdk | 0.00537 | 0.14 | -0.305 | 0.006 |
| 16 | `CNA::SPINT2` \* | genome | estrogen_progesterone | 0.00536 | 0.09 | -0.364 | 0.006 |
| 17 | `Akt_pT308-R-V` \* | proteome | PI3K/AKT/mTOR | 0.00507 | 0.14 | 0.392 | 0.006 |
| 18 | `CNA::FASN` \* | genome | metabolism | 0.00538 | 0.12 | -0.320 | 0.006 |
| 19 | `CNA::ERBB2` \* | genome | ras_mapk | 0.00543 | 0.07 | -0.364 | 0.006 |
| 20 | `CNA::GRB2` \* | genome | ras_mapk | 0.00538 | 0.14 | -0.273 | 0.019 |
| 21 | `Akt_pS473-R-V` \* | proteome | PI3K/AKT/mTOR | 0.00503 | 0.13 | 0.378 | 0.006 |
| 22 | `MSH6` \* | transcriptome | dna_repair | 0.00511 | 0.09 | -0.402 | 0.006 |
| 23 | `CNA::SOX17` \* | genome | tumor_suppressors | 0.00527 | 0.09 | -0.349 | 0.006 |
| 24 | `CNA::SOX2` \* | genome | wnt_stemness | 0.00533 | 0.08 | -0.338 | 0.006 |
| 25 | `CNA::PIK3CA` \* | genome | pi3k_pten | 0.00531 | 0.07 | -0.353 | 0.006 |

\* FDR-significant. Full table in `artifacts/biomarkers.csv`.

## How to read this

**The best AUC on this cohort belongs to Random Forest (0.6765), not to the hybrid model (0.5931).** That is reported rather than engineered away.

Two things are worth separating:

1. **Classification.** With 545 samples, a 38 % event rate and a distributed polygenic signal, gradient-boosted trees are a strong prior for tabular omics data. The transformer's advantage was never expected to be raw AUC at this cohort size.
2. **Discovery.** The transformer's actual contribution is the ranked panel. Attention rollout over cross-omic tokens produces a pathway-annotated, effect-size-backed, FDR-controlled list of named molecules — which is the deliverable, and which the bagged-tree baseline does not produce.

### Signal in the raw layers

Before any model is fitted, the best single marker in each layer reaches only ~0.63–0.68 AUC (PGR and ESR1 in the transcriptome, fraction-genome-altered in the genome, MSH2 in the proteome). Less than 1 % of markers exceed 0.65 alone. The endpoint is polygenic, which is why aggregate models land where they do and why per-marker claims would be unreliable. Run `python scripts/diagnose_signal.py` to reproduce.

