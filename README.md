# Q-OMICS

**An open hybrid quantum-classical foundation model for multi-omics biomarker discovery.**

Q-OMICS learns which molecules separate early-stage from advanced disease by
combining three real omics layers — somatic mutations and copy number, RNA-seq
expression, and reverse-phase protein array abundance — with a **fidelity
quantum kernel** and a **cross-omic transformer** that reads its own attention
to rank the biomarkers it relied on.

It runs on public data. It ships the artefacts it produced. And it reports
whether the quantum branch actually earned its place.

---

## Why this exists

Endometrial carcinoma is the most common gynaecological malignancy in high-income
countries and stage at diagnosis is the dominant determinant of survival. Stage I
disease is confined to the uterus; stage II–IV has spread beyond it. The clinical
problem is to identify stage I from a molecular profile, because that is the
window where the intervention is most effective.

Doing that well is hard for a specific reason: the biology is **multi-layered**.
PTEN loss drives PI3K signalling, which is visible in the phosphoproteome, which
feeds back on transcription, and none of the three layers alone is sufficient.
Classical multi-omics models typically score each layer separately and miss those
cross-layer couplings. Q-OMICS treats the sample as a *sequence of molecular
tokens* and lets attention learn which markers co-vary across layers.

---

## What it does

| Capability | Detail |
|---|---|
| **Cohort** | TCGA-UCEC, 463 primary tumours, 462 patients |
| **Layers** | Genome (mutations, GISTIC copy number, TMB) · Transcriptome (RNA-seq TPM z-scores) · Proteome (189 RPPA antibodies) |
| **Target** | FIGO stage I vs stage II–IV — 340 / 123 |
| **Quantum branch** | Fidelity kernel `k(x,x') = |⟨ψ(x)|ψ(x')⟩|²` on a 6-qubit register, 64-dimensional Hilbert space |
| **Classical branch** | 2-layer, 4-head transformer over 192 molecular tokens, ~76k parameters |
| **Hybrid fusion** | Quantum-kernel SVM score enters the sequence as a fusion token, stacked out-of-fold |
| **Output** | Ranked, pathway-annotated biomarker panel with effect sizes and FDR control |
| **Validation** | Repeated stratified group 5-fold × 2 repeats, patient-disjoint, with paired significance testing |

---

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/Q-OMICS.git
cd Q-OMICS

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/fetch_data.py     # download the cohort (~1 min, cached afterwards)
python scripts/train.py          # cross-validate and fit the model
python scripts/serve.py          # http://127.0.0.1:8000
```

`train.py` writes every artefact the interface serves into `artifacts/`. The
web interface, the API docs and the ranked panel are all available once
training finishes:

- **Web interface** — <http://127.0.0.1:8000>
- **OpenAPI docs** — <http://127.0.0.1:8000/docs>
- **Ranked panel** — `python scripts/train.py --report`, or `cat artifacts/biomarkers.txt`

### Command line

```bash
python -m qomics.cli fetch                        # assemble the cohort
python -m qomics.cli train                        # full protocol
python -m qomics.cli train --repeats 1 --splits 5 # quick pass
python -m qomics.cli train --skip-transformer     # baselines + quantum only
python -m qomics.cli serve --port 9000            # API + web interface
python -m qomics.cli report                       # print the biomarker panel
```

---

## The quantum branch, concretely

A classical kernel is an inner product in a feature space you have to choose by
hand. A fidelity quantum kernel embeds each sample into a quantum state and
takes the squared overlap:

```
k(x, x') = |⟨ψ(x) | ψ(x')⟩|²
```

The state lives in a 2ⁿ-dimensional Hilbert space, so the kernel evaluates an
inner product in a far larger space than the input admits directly.

**Two implementation decisions make a real cohort tractable.**

*State-vector caching.* The textbook kernel runs the compute-unconfigure circuit
`U(x)`, then `U(x')†`, and reads the all-zero probability — `n²` circuit
executions for an `n × n` Gram matrix. Since PennyLane exposes the full
wave-function, Q-OMICS evaluates each embedding once and forms every pairwise
overlap as a single matrix product. Algebraically identical, about two orders of
magnitude cheaper.

*Broadcast batching.* A Python loop over samples is dominated by QNode tape
construction, not by the simulation. Issuing the batch as one broadcasted call
produces **bit-identical** output and is ~240× faster: 500 samples × 6 qubits ×
2-layer IQP went from 14.5 s to 0.06 s.

**Correctness is asserted, not assumed.** The non-entangled `sep_rx` embedding
has a closed-form fidelity, `∏ cos((x_j − x'_j)/2)²`. The test suite requires
the simulator to agree with that closed form to 1e-9. If the backend were wrong,
the suite would fail rather than quietly produce plausible numbers.

---

## Honest reporting

A quantum advantage claim should be falsifiable. Q-OMICS reports three things
that could contradict it, and does so whether or not they come out favourably.

**Strong classical baselines.** An RBF SVM, logistic regression, random forest
and histogram gradient boosting all train on identical folds and features. The
RBF SVM is the one that matters — swapping its Gram matrix for the quantum one is
the *only* difference, so it is the model a fidelity kernel is meant to beat.

**Geometric difference.** Following Huang et al. (2024),

```
g(K_q, K_c) = sqrt( ‖ sqrt(K_q) · K_c⁻¹ · sqrt(K_q) ‖_∞ )
```

`g = 1` means the two kernels induce the *same* geometry, and no kernel learner
can do better with one than the other. Q-OMICS reports `g` against RBF and
cosine references, plus centred kernel alignment.

**An entanglement ablation.** The entangled IQP Gram is compared against the
non-entangled `sep_rx` Gram on the same inputs and the same register width, so
the difference isolates entanglement itself rather than encoding style or qubit
count.

**Multiplicity.** The hybrid model is compared against every baseline on
identical folds with a paired t-test, Holm-Bonferroni corrected across the
family. Folds are paired by construction, so an unpaired test would inflate
variance and hide a real effect.

If `g` comes out near 1, the conclusion is that the quantum kernel adds no
structural advantage on this cohort — and the report says so.

---

## Layout

```
qomics/
  config.py            every tunable, in one dataclass tree
  genes.py             curated 375-symbol panel, organised by pathway axis
  cli.py               fetch / train / serve / report
  data/                cBioPortal + Xena clients, cohort assembly
  quantum/             feature maps, fidelity kernel, normalisation, diagnostics
  models/              transformer encoder, classical baselines
  pipeline/            preprocessing, training, evaluation, geometry, biomarkers
  serving/             FastAPI app, artefact store, schemas
  web/                 static single-page frontend, no build step
docs/
  RESULTS.md         generated from artifacts/run.json - never edited by hand
  METHODOLOGY.md     what is computed, why, and where it could be wrong
  ARCHITECTURE.md    module map, data flow, invariants
  DATA_SOURCES.md    provenance for every layer
  API_CONTRACT.md    the wire format
  CONTRIBUTING.md    how to propose a change
tests/                 pytest suite
scripts/               fetch_data · train · serve · diagnose_signal · make_figures · make_results_md
artifacts/             written by train.py: run.json, biomarkers.csv, model_bundle.pt
```

---

## Results

Full numbers, tables and the biomarker panel are in
**[`docs/RESULTS.md`](docs/RESULTS.md)**, generated directly from
`artifacts/run.json` so they cannot drift from the run that produced them.

The short, honest version: on this cohort the **best classification AUC belongs
to a classical baseline, not to the hybrid model**, and Q-OMICS reports that
rather than engineering it away. Two things are worth separating.

- **Classification.** 463 samples, a 27 % event rate, and a distributed
  polygenic signal — the best single marker in any layer reaches only ~0.63–0.68
  AUC on its own, and under 1 % of markers exceed 0.65. Gradient-boosted trees
  are a strong prior for tabular omics at this sample size, and the hybrid
  transformer's advantage was never expected to be raw AUC.
- **Discovery.** The transformer's actual contribution is the ranked panel.
  Attention rollout over cross-omic tokens produces a pathway-annotated,
  effect-size-backed, FDR-controlled list of named molecules, led by hormone
  receptor and PI3K-pathway markers. The bagged-tree baseline does not produce
  that; it produces a score.

The **quantum branch does not outperform its classical references on this
cohort**, and the geometric-difference report says so explicitly rather than
quoting a large `g` as an advantage. See
[`docs/RESULTS.md`](docs/RESULTS.md#quantum-kernel) and
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md#52-does-the-quantum-kernel-actually-add-geometry).

---

## Reproducibility

Every number comes from one deterministic protocol. Seeds are set for Python,
NumPy and PyTorch; the quantum embedding is parameter-free, so there is no
quantum optimiser and nothing to converge stochastically. The single run writes
`artifacts/run.json` containing the full configuration, the complete per-fold
metrics and the geometry report, so a result can be traced to the exact settings
that produced it.

```bash
pytest -q          # test suite
```

---

## Extending

**Another cohort.** Change `CohortConfig` in `qomics/config.py`. If the target
ontology is not FIGO, add a parser next to `parse_figo_stage`.

**Another feature map.** Implement a `_block_*` method in `QuantumFeatureMap`.
If the map factorises, add a closed-form check so the simulator stays verifiable.

**Another modality.** Add a block to `Cohort.blocks`. The preprocessor, the
per-modality token lift and the biomarker annotation pick it up automatically.

---

## Limitations

Stated in full in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md#6-what-could-be-wrong).
The short version:

- The quantum branch runs on a **noiseless state-vector simulator**. Every number
  is an upper bound on what the same circuit would deliver on hardware; fidelity
  kernels are known to degrade sharply under noise.
- **One retrospective cohort, no external validation.** The cross-validated AUC
  is an internal estimate.
- **Attention importance is attribution, not causation.** Effect sizes,
  permutation p-values and FDR are reported alongside for that reason.
- **Six qubits.** Results describe that register width. Whether the geometry
  stays distinct at 12 or 20 qubits is the first experiment worth running next.

---

## Data and licence

Code is MIT licensed — see [`LICENSE`](LICENSE).

Q-OMICS downloads data from **cBioPortal** and **UCSC Xena**, both of which serve
TCGA openly and require no credentials. No patient data is collected, stored or
redistributed by this project; downloads are cached locally under `data/` at run
time and the repository ships only derived aggregate results. The underlying
datasets remain subject to their providers' terms — see
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

---

## Citation

If you use Q-OMICS, please cite the project and the underlying data. See
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md#citation) for the TCGA, cBioPortal
and Xena citations.

---

**Research use only. Not a diagnostic device and not for clinical decision-making.**
