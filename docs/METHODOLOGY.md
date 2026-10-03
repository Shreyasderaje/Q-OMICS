# Methodology

This document states exactly what Q-OMICS computes, why each choice was
made, and where the result could be wrong. It is written to be checked.

---

## 1. The task

**FIGO stage I versus stage II–IV endometrial carcinoma.**

Stage I means the tumour is confined to the uterus. Stage II–IV means it has
spread to the cervix, adnexa, lymph nodes or beyond. The clinical stake is
the standard one: stage at diagnosis is the dominant determinant of
endometrial cancer survival, and the earlier a case is caught the more
options the patient has.

The label is parsed from the cBioPortal `FIGO_GRADE` patient attribute, which
carries strings such as `Stage IA`, `Stage IIIC2`, `Stage IVB`. Parsing
extracts the Roman numeral only — see `parse_figo_stage` — because a naive
prefix test maps `Stage II` to stage `I` and silently inverts the label.

Cohort: 545 primary-tumour samples, 544 patients, 340 stage I / 205 stage
II–IV. The 1.7 : 1 imbalance is real clinical prevalence, not an artefact, and
it drives several design decisions below.

---

## 2. Features

Three omics layers, joined on the TCGA sample barcode:

- **Genome** (741 features) — somatic mutation indicator per gene, GISTIC
  copy number per gene, plus tumour mutational burden, total mutation count
  and fraction of genome altered.
- **Transcriptome** (374 features) — RNA-seq TPM z-scores on a 375-gene
  curated panel spanning seventeen pathway axes.
- **Proteome** (189 features) — RPPA antibody abundance, variance-filtered.

A fourth representation is derived by reducing each modality with PCA and
concatenating, weighted so no modality dominates the Euclidean geometry. This
12-dimensional vector is what the quantum register encodes.

**Every transform is fitted on training rows only.** Scaler, variance filter
and PCA are all re-fitted per fold. Fitting them once on the full cohort would
let test-fold variance leak into training and inflate every number reported.

---

## 3. The quantum branch

### 3.1 Definition

A **fidelity quantum kernel** embeds each sample into a quantum state and
takes the squared overlap as an inner product:

```
k(x, x') = |<psi(x) | psi(x')>|^2
```

The state `|psi>` lives in a 2^n-dimensional Hilbert space. For six qubits
that is 64 dimensions, already larger than the 12 features encoded.

### 3.2 Encoding

The default map is **IQP**: `H^n`, then data-controlled `RZ` rotations and
all-to-all `ZZ` entanglers, repeated for two layers with data re-uploading, then
a small `RY` rotation per qubit. Gates are written out explicitly rather than
taken from PennyLane templates — template signatures have changed across
releases (`reps` → `n_repeats`, `rotations` dropped from `AngleEmbedding`),
and the primitive gates are stable API surface.

`hea` (Hamiltonian evolution) and `sep_rx` (product state) are also
implemented. `sep_rx` is a deliberate **control**: its Gram matrix is
analytically `prod_j cos((x_j - x'_j)/2)^2`, which lets the simulator be
verified against closed-form theory to machine precision. If the simulator
and the theory disagree, nothing downstream means anything.

### 3.3 Two implementation decisions that make it tractable

**State-vector caching.** The textbook kernel runs the compute-unconfigure
circuit `U(x)`, then `U(x')†`, and reads the all-zero probability — `n^2`
circuit executions for an `n × n` Gram matrix. Because the simulator exposes
the full wave-function, Q-OMICS evaluates each embedding once and forms every
pairwise overlap as one matrix product:

```
K = |Psi Psi^dagger|^2,    Psi in C^(n x 2^q)
```

Algebraically identical, roughly two orders of magnitude cheaper.

**Broadcast batching.** A Python loop over samples is dominated by QNode tape
construction, not by the simulation. Issuing the batch as a single broadcasted
call produces bit-identical output (maximum absolute difference exactly 0) and
is **~240× faster** on this circuit: 500 samples × 6 qubits × 2-layer IQP went
from 14.5 s to 0.06 s.

### 3.4 Normalisation

Raw fidelity Gram matrices become diagonally dominated as embedding depth
grows, which starves a downstream SVM of usable contrast. The default
`arc_cosine_scaled` scheme projects each embedding onto the unit sphere and
applies the first-order arc-cosine (neural-network) kernel:

```
k'(x, x') = (2/pi) * arcsin( <psi(x), psi(x')> )
```

This preserves positive semi-definiteness while expanding contrast in the
near-diagonal regime.

**Cross-blocks are normalised by the training-set state norms**, not by their
own diagonal. A test row divided by its own norm is systematically scaled
relative to the training rows, and the SVM computes its margin in the wrong
geometry. A unit test asserts that duplicating a training point reproduces its
own training row exactly.

### 3.5 Fusion without leakage

The quantum-kernel SVM's output becomes a *fusion token* in the transformer
sequence. For training rows that score must be **out-of-fold**: an inner
stratified split refits the kernel and the SVM on part of the training set and
scores the rest. Feeding the SVM's own training-set scores to the transformer
would make the fusion token near-perfect in training and worthless at test
time — the single most likely way to fake a hybrid-model result.

---

## 4. The classical branch

A biomarker panel is not a set of independent scores: PTEN loss drives PI3K
signalling, which is visible in the phosphoproteome, which feeds back on
transcription. So the sample is treated as a **sequence of molecular tokens**,
one per feature:

1. Each scalar feature is lifted to `d_model` by a per-modality linear map, so
   a token carries its value plus a modality identity.
2. A learnable `[CLS]` token aggregates the sequence.
3. The quantum fusion score enters as an additional token.
4. Three pre-norm encoder layers, four heads, mean attention.
5. The head concatenates `[CLS]` with a mean pool and classifies.

Token budget: 192 transcriptome + 192 genome + 96 proteome = **480 tokens**.

### 4.1 Biomarker readout

Per-feature importance comes from **attention rollout** (Abnar & Zuidema,
2020): at each layer the attention map is added to the identity, row-normalised
and composed across depth. Plain per-layer attention ignores the residual
stream, so a token's effective influence decays geometrically and vanishes by
layer three; rollout fixes that. The `[CLS]` row of the rollout is the
importance of each token.

Rollouts are comparable only *within* a fold, so each is normalised to unit
sum before pooling across folds.

### 4.2 Optimisation

Focal loss, not plain cross-entropy. Stage II–IV is the minority (37.6 %) and
the clinically dangerous class; cross-entropy collapses onto stage I, and
inverse-frequency class weights alone over-correct and destabilise training.
Focal loss down-weights easy examples so capacity is spent on the decision
boundary that matters.

AdamW, cosine schedule with warmup, gradient clipping at 1.0, label smoothing
0.05, early stopping on a held-out 15 % slice of the training split by
validation AUC.

---

## 5. Evaluation

**Repeated stratified group K-fold**, 5 folds × 3 repeats = 15 splits, built on
the *patient* identifier. Several TCGA aliquots can come from one patient;
splitting on samples would place near-duplicate profiles on both sides of a
split and quietly inflate the score.

Metrics: ROC-AUC, PR-AUC, F1, sensitivity, specificity and Brier score, all at
the Youden-optimal operating point (threshold maximising sensitivity +
specificity − 1). Out-of-fold scores are averaged per sample across repeats
and pooled, giving one honest score per sample rather than a mean of
small-sample scores dominated by fold-size variance. Confidence intervals are
percentile bootstrap over the pooled scores.

**Significance** is a paired t-test on fold AUCs from identical splits,
Holm-Bonferroni corrected across the baseline family. Folds are paired by
construction, so an unpaired test would inflate the variance and hide a real
effect; with eight models compared, an uncorrected test would produce a
spurious "improvement" often enough to matter.

### 5.1 Baselines

A quantum claim is only meaningful against strong classical references, all
trained on identical folds and features:

| Baseline | Role |
|---|---|
| Logistic regression | Linear floor |
| **RBF SVM** | The model a fidelity kernel is *meant* to beat — swapping the Gram matrix is the only difference |
| Random forest | Bagged trees, scaling-immune |
| Histogram gradient boosting | Usual tabular leader |
| PCA + logistic regression, transcriptome only | What most published UCEC stage models use; quantifies what genome and proteome add |
| Quantum SVM | The kernel alone |
| Classical transformer | The same network without the fusion token — the hybrid ablation |

### 5.2 Does the quantum kernel actually add geometry?

The sharpest available test, from Huang et al. (2024), is the **geometric
difference**

```
g(K_q, K_c) = sqrt( || sqrt(K_q) K_c^-1 sqrt(K_q) ||_inf )
```

`g = 1` means the two kernels induce the same geometry and *no* kernel learner
can do better with one than the other. `g > 1` means the quantum geometry is
not reproducible by that classical family.

Q-OMICS reports `g` against RBF and cosine references, plus centred kernel
alignment and the labelled separation ratio. **This is reported whatever it
says.** If `g ≈ 1`, the honest conclusion is that the quantum kernel adds no
structural advantage on this cohort, and the report states that.

An **entanglement ablation** separates the contribution of entanglement from
the contribution of the encoding style: the entangled IQP Gram is compared
against the non-entangled `sep_rx` Gram on the *same* inputs and the *same*
register width. Any difference isolates entanglement itself.

---

## 6. What could be wrong

Stated plainly, because a report that cannot fail is not a report.

- **The quantum branch runs on a noiseless state-vector simulator.** No shot
  noise, no gate error, no finite connectivity. Fidelity kernels are known to
  degrade sharply under noise, particularly as depth grows. Every number here
  is an upper bound on what the same circuit would deliver on hardware.
- **The cohort is retrospective and single-cohort.** 545 tumours from one
  project, all FIGO-staged. There is no external validation set. The
  cross-validated AUC is an internal estimate.
- **The proteome is partial.** 437 of 545 samples carry RPPA data; the rest
  are median-imputed. RPPA also measures a few hundred proteins, not the
  whole proteome.
- **The panel is curated, not exhaustive.** That is a deliberate trade for
  interpretability and register width, and it means a strong marker outside
  the panel is invisible to the model.
- **Attention importance is not causal.** Rollout attributes the model's
  decision; it does not establish that perturbing the marker changes the
  outcome. Effect sizes, permutation p-values and FDR are reported alongside
  precisely because attention alone is insufficient evidence.
- **Focal loss and class weighting are tuned for the observed 37.6 %
  prevalence.** A screening population with a different prevalence needs the
  operating point re-derived; the AUC is prevalence-independent, the
  sensitivity and specificity are not.
- **The quantum register is six qubits.** Results describe that width.
  Whether the geometry stays distinct at 12 or 20 qubits is untested here and
  is the first experiment worth running next.
