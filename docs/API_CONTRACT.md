# Q-OMICS HTTP API Contract

Base URL: `http://127.0.0.1:8000`
All endpoints return JSON. Errors return `{"detail": "..."}` with a 4xx/5xx status.

The frontend is a static single-page app served from `/`. It must work against
exactly the shapes below.

---

## GET `/api/health`

```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true,
  "trained_at": "2026-02-11T09:31:04Z",
  "cohort": "TCGA-UCEC (GDC, 2025)",
  "runtime_seconds": 12.4
}
```

---

## GET `/api/overview`

Everything the landing page needs in one request.

```json
{
  "product": {
    "name": "Q-OMICS",
    "tagline": "Hybrid quantum-classical foundation model for multi-omics biomarker discovery",
    "version": "1.0.0"
  },
  "cohort": {
    "study": "TCGA-UCEC (GDC, 2025)",
    "study_id": "ucec_tcga_gdc",
    "target": "early_stage_detection",
    "target_description": "FIGO stage I vs stage II-IV endometrial carcinoma",
    "n_samples": 545,
    "n_patients": 544,
    "block_shapes": {"transcriptome": 374, "genome": 741, "proteome": 189},
    "class_counts": {"class_0": 340, "class_1": 205},
    "class_balance": 0.3761,
    "feature_annotations": 1304
  },
  "headline_metrics": {
    "hybrid_auc_mean": 0.5931,
    "hybrid_auc_std": 0.0552,
    "hybrid_auc_ci_low": 0.5176,
    "hybrid_auc_ci_high": 0.6178,
    "best_baseline_auc_mean": 0.6765,
    "best_baseline_name": "Random Forest",
    "improvement_over_best_classical": -0.0834,
    "n_splits": 10
  },
  "modality_contribution": [
    {"block": "transcriptome", "importance": 0.4098},
    {"block": "genome", "importance": 0.3888},
    {"block": "proteome", "importance": 0.2015}
  ]
}
```

`improvement_over_best_classical` is signed and is **negative** on the
recorded run: the hybrid model does not lead on AUC. The field is reported as
measured rather than flipped to a positive framing.

---

## GET `/api/metrics`

Full model-comparison table plus per-fold detail.

```json
{
  "target": "FIGO stage I vs stage II-IV endometrial carcinoma",
  "n_splits": 10,
  "primary": "hybrid_transformer",
  "models": [
    {
      "name": "Hybrid Transformer (Q-OMICS)",
      "family": "hybrid",
      "auc_mean": 0.5931, "auc_std": 0.0552,
      "auc_pooled": 0.5674, "auc_ci_low": 0.5176, "auc_ci_high": 0.6178,
      "pr_auc_mean": 0.4643, "pr_auc_std": 0.0630,
      "f1_mean": 0.5455, "f1_std": 0.0461,
      "sensitivity_mean": 0.6098, "sensitivity_std": 0.1331,
      "specificity_mean": 0.6294, "specificity_std": 0.1562,
      "brier_mean": 0.2393, "brier_std": 0.0167,
      "seconds_mean": 194355.2,
      "n_folds": 10
    },
    {
      "name": "Random Forest",
      "family": "ensemble",
      "auc_mean": 0.6765, "auc_std": 0.0362,
      "auc_pooled": 0.6744, "auc_ci_low": 0.6270, "auc_ci_high": 0.7201,
      "pr_auc_mean": 0.5594, "pr_auc_std": 0.0430,
      "seconds_mean": 1.07,
      "n_folds": 10
    }
  ],
  "significance": [
    {
      "comparison": "Hybrid Transformer (Q-OMICS) vs Random Forest",
      "reference": "random_forest",
      "mean_delta": -0.0834,
      "t_statistic": -3.629,
      "p_value": 0.0055,
      "n_pairs": 10,
      "significant_at_0.05": true,
      "significant_after_correction": true
    },
    {
      "comparison": "Hybrid Transformer (Q-OMICS) vs Quantum SVM",
      "reference": "quantum_svm",
      "mean_delta": 0.0761,
      "t_statistic": 3.507,
      "p_value": 0.0066,
      "n_pairs": 10,
      "significant_at_0.05": true,
      "significant_after_correction": true
    }
  ],
  "roc_curve": {
    "labels": ["class_0", "class_1"],
    "curves": [{"name": "Hybrid Transformer (Q-OMICS)", "fpr": [0.0, 0.02], "tpr": [0.0, 0.51]}]
  }
}
```

All eight models are present, each with `n_folds: 10` (5 folds × 2 repeats).
The recorded run has the hybrid model significantly **behind** Random Forest
(Δ −0.0834, p = 0.0055, Holm-Bonferroni significant) and RBF SVM
(Δ −0.0774, p = 0.0068), and significantly ahead of the Quantum SVM
(Δ +0.0761, p = 0.0066). The two `roc_curve` entries shown above are truncated
for readability; the real arrays carry one point per distinct FPR value.

**Model names that must be present** (exact strings):
`Hybrid Transformer (Q-OMICS)`, `Quantum SVM`, `Classical Transformer`,
`RBF SVM`, `Logistic Regression`, `Random Forest`, `HistGradientBoosting`,
`PCA + Logistic Regression (transcriptome only)`.

---

## GET `/api/biomarkers?limit=25&block=all`

```json
{
  "total": 192,
  "returned": 1,
  "method": "attention_rollout",
  "biomarkers": [
    {
      "rank": 1,
      "name": "CNA::CCNE1",
      "symbol": "CCNE1",
      "block": "genome",
      "axis": "cell_cycle_cdk",
      "importance": 0.005336,
      "importance_normalised": 0.005366,
      "modality": "genome",
      "stability": 0.1339,
      "direction": "depleted_in_early_stage",
      "mean_early": -0.1923,
      "mean_advanced": 0.2976,
      "effect_size": -0.5067,
      "p_value": 0.000999,
      "q_value": 0.005812,
      "fdr_significant": true,
      "n_observed": 545
    }
  ]
}
```

`block` accepts `all`, `transcriptome`, `genome`, `proteome`. `total` is the size
of the ranked panel (192 tokens), not the number of raw archive features.

---

## GET `/api/biomarkers/{symbol}`

```json
{
  "rank": 1,
  "name": "CNA::CCNE1",
  "symbol": "CCNE1",
  "block": "genome",
  "axis": "cell_cycle_cdk",
  "importance": 0.005336,
  "importance_normalised": 0.005366,
  "stability": 0.1339,
  "direction": "depleted_in_early_stage",
  "mean_early": -0.1923,
  "mean_advanced": 0.2976,
  "effect_size": -0.5067,
  "p_value": 0.000999,
  "q_value": 0.005812,
  "fdr_significant": true,
  "n_observed": 545,
  "modality": "genome",
  "present": true,
  "train_frequency_early": -0.1923,
  "train_frequency_advanced": 0.2976
}
```

For genome features, `train_frequency_*` carries the **mean z-score per class**
rather than a mutation rate — for a CNA token the quantity of interest is the
copy-number level, not how often the gene is mutated. `p_value`, `q_value` and
`fdr_significant` (Benjamini-Hochberg) are present on every entry.

Returns `{"present": false, "symbol": "..."}` with HTTP 200 when the symbol is
not in the ranked panel (a miss is not an error).

---

## GET `/api/quantum`

Quantum-kernel provenance and the geometric-difference analysis.

```json
{
  "backend": "pennylane.default.qubit",
  "n_qubits": 6,
  "hilbert_dimension": 64,
  "feature_map": "iqp",
  "n_layers": 2,
  "reupload": true,
  "normalisation": "arc_cosine_scaled",
  "encoding_seconds": 0.041,
  "inner_splits": 3,
  "kernel_diagnostics": {
    "n": 436, "mean": 0.0362, "std": 0.0307, "min": 1.66e-06, "max": 0.5343,
    "mean_diagonal": 1.0, "condition_number": 46.74, "effective_rank": 352.8
  },
  "labelled_separation": {
    "quantum": -0.00062, "rbf": -0.0191, "cosine": -0.0310, "linear": -0.0207
  },
  "entanglement_ablation": {
    "separation_entangled": -0.00129,
    "separation_product": -0.00977,
    "relative_gain": 0.868,
    "n_samples": 200
  },
  "kernel_similarity": {
    "rbf": {"cka": 0.2012, "mean_abs_difference": 0.5663},
    "cosine": {"cka": 0.1726, "mean_abs_difference": 0.3363},
    "linear": {"cka": 0.1649, "mean_abs_difference": 0.3099}
  },
  "geometric_difference": {
    "g_quantum_vs_rbf": 95.25,
    "g_quantum_vs_cosine": 149.16,
    "g_quantum_vs_linear": 151.46,
    "interpretation": "g = 95.249, but the quantum kernel carries essentially no label separation (between-class minus within-class similarity -6.20e-04, below the 1e-3 resolution of this cohort). A large or small g is not evidence of an advantage: the kernel does not separate the classes at all, so no kernel learner can benefit from its geometry. On this cohort the quantum branch does not outperform its classical references."
  },
  "state_vector_visualisation": {
    "n_examples": 2,
    "labels": ["seeded state A", "seeded state B"],
    "dim": 64,
    "amplitudes": [[0.0465, 0.0848], [0.0882, 0.1862]],
    "pairwise_fidelity": 0.0066
  }
}
```

`geometric_difference` and `labelled_separation` are reported together on
purpose. On this cohort `g` is large while the labelled separation is
indistinguishable from zero — a conditioning artefact, not an advantage, and the
shipped `interpretation` string says so.

The `state_vector_visualisation` labels are **seeded, synthetic** states, not
cohort samples: they are a fixed, reproducible pair so the frontend can render
amplitude bars without leaking cohort data. Only the first 16 of 64 amplitudes
per state are returned.

---

## POST `/api/predict`

```json
{
  "sample": {
    "transcriptome": {"PTEN": -1.2, "MKI67": 0.4},
    "genome": {"CNA::PTEN": 1, "CNA::TP53": -1, "CNA::CCNE1": 2},
    "proteome": {"pAKT_pS473-R-V": 0.8}
  },
  "top_k": 3
}
```

Missing features are imputed from the training median; unknown feature names
are ignored. Response:

```json
{
  "prediction": "early_stage",
  "label": "Stage I (early)",
  "confidence": 0.1079,
  "probabilities": {
    "stage_I_early": 0.8921,
    "stage_II_IV_advanced": 0.1079
  },
  "risk_tier": "low",
  "quantum_score": 0.4821,
  "matched_features": 4,
  "total_features": 192,
  "coverage": 0.0208,
  "top_contributors": [
    {"name": "CNA::PTEN", "symbol": "PTEN", "block": "genome",
     "importance": 0.01929, "value": 1.0, "supplied": true,
     "direction": "depleted_in_early_stage", "contribution": 0.01929}
  ],
  "disclaimer": "Research use only. Not a diagnostic device and not for clinical decision-making."
}
```

`prediction` is one of `early_stage`, `advanced_stage`.
`risk_tier` is one of `low`, `moderate`, `elevated`.
`confidence` is the probability of the **predicted** class, so it is not
symmetric across classes.

`total_features` is 192 — the size of the ranked token panel, not the raw
archive block (374 / 741 / 189). `coverage` is therefore
`matched_features / 192` and is small for any small ad-hoc request; that is
expected, since a real cohort upload should supply most of the panel. Each
`top_contributors` entry carries `supplied`, which is `false` when the feature
was median-imputed rather than provided in the request — imputed contributors
are shown for completeness and are the dominant entries for sparse input.
`quantum_score` is the normalised quantum-kernel affinity of the sample, exposed
for transparency about the quantum branch's actual contribution.

---

## GET `/api/cohort`

```json
{
  "columns": ["sample_id", "patient_id", "figo_stage", "figo_label", "sample_type",
              "histological_type", "age_at_diagnosis", "prediction", "confidence", "in_sample"],
  "rows": [
    {"sample_id": "TCGA-2E-A9G8-01", "patient_id": "TCGA-2E-A9G8",
     "figo_stage": "III", "figo_label": "Stage II-IV (advanced)",
     "sample_type": "tumor",
     "histological_type": "Endometrioid endometrial adenocarcinoma",
     "age_at_diagnosis": null,
     "prediction": "early_stage", "confidence": 0.0237, "in_sample": true}
  ],
  "stage_distribution": [{"stage": "I", "count": 340}, {"stage": "II", "count": 52},
                          {"stage": "III", "count": 124}, {"stage": "IV", "count": 29}],
  "histology_distribution": [
    {"name": "Endometrioid endometrial adenocarcinoma", "count": 409},
    {"name": "Serous endometrial adenocarcinoma", "count": 114},
    {"name": "Mixed serous and endometrioid", "count": 22}
  ],
  "total": 545,
  "offset": 0,
  "limit": 60
}
```

`figo_stage` is the **Roman numeral group** (`I`, `II`, `III`, `IV`), not the
full clinical staging such as `IIIC2` — grouping is what the label is derived
from. `age_at_diagnosis` is `null` where the clinical record has no value; it is
never coerced to `0`. `in_sample` is `true` for rows that were present in the
training cohort.

Query params: `limit` (default 60, max 545), `offset` (default 0).

---

## GET `/api/cohort/download`

Query params:

| Param | Default | Behaviour |
|---|---|---|
| `format` | `csv` | `csv` returns `text/csv`; `json` returns a JSON array of row objects |

CSV is the default because a spreadsheet is what a cohort export is usually
for. The `format` query parameter — not the `Accept` header — selects the
representation, because many HTTP clients send `*/*` or advertise
`application/json` while still expecting a file download.

CSV columns: `sample_id,patient_id,figo_stage,figo_label,histological_type,age_at_diagnosis,sample_type,prediction,confidence,in_sample`.

```csv
sample_id,patient_id,figo_stage,figo_label,histological_type,age_at_diagnosis,sample_type,prediction,confidence,in_sample
TCGA-2E-A9G8-01,TCGA-2E-A9G8,III,Stage II-IV (advanced),Endometrioid endometrial adenocarcinoma,,tumor,early_stage,0.024,true
```

---

## GET `/api/reload` (POST)

Re-reads `artifacts/` from disk. Useful after a re-training run without
restarting the server.

```json
{"reloaded": true, "ready": true, "models": 8, "biomarkers": 192,
 "cohort_rows": 545, "bundle_loaded": true}
```

---

## GET `/api/architecture`

Static description of the model graph, for the architecture page.

```json
{
  "stages": [
    {"id": "ingest", "title": "Cohort ingestion",
     "detail": "cBioPortal GDC + UCSC Xena RPPA", "tokens": 0},
    {"id": "reduce", "title": "Modality reduction",
     "detail": "variance filter + per-modality PCA", "tokens": 192},
    {"id": "quantum", "title": "Fidelity quantum kernel",
     "detail": "6-qubit IQP, 64-dimensional Hilbert space", "tokens": 0},
    {"id": "encode", "title": "Multi-omics tokenisation",
     "detail": "one token per molecular feature", "tokens": 192},
    {"id": "attend", "title": "Cross-omic attention",
     "detail": "2 layers x 4 heads", "tokens": 192},
    {"id": "readout", "title": "Attention rollout",
     "detail": "biomarker attribution", "tokens": 192}
  ],
  "model": {"d_model": 64, "n_heads": 4, "n_layers": 2,
            "dim_feedforward": 128, "dropout": 0.25},
  "blocks": {"transcriptome": 80, "genome": 72, "proteome": 40},
  "total_tokens": 192,
  "training": {"epochs": 40, "batch_size": 64, "learning_rate": 0.0015,
               "weight_decay": 0.0001, "loss": "focal", "n_splits": 10,
               "device": "cpu"}
}
```

---

## GET `/api/features`

The per-modality feature inventory backing the biomarker explorer.

```json
{
  "blocks": [
    {"block": "transcriptome", "n_features": 80,
     "axes": [{"axis": "immune", "count": 11},
              {"axis": "emt_invasion", "count": 8},
              {"axis": "pi3k_pten", "count": 7}]},
    {"block": "genome", "n_features": 72,
     "axes": [{"axis": "immune", "count": 8},
              {"axis": "estrogen_progesterone", "count": 6},
              {"axis": "tumor_suppressors", "count": 6}]},
    {"block": "proteome", "n_features": 40,
     "axes": [{"axis": "unclassified", "count": 24},
              {"axis": "cell cycle", "count": 4},
              {"axis": "adhesion", "count": 3}]}
  ],
  "n_total": 192
}
```

Axis lists are truncated above for readability; the response carries every axis
for each block. Note the proteome block is dominated by `unclassified` (24 of
40) — the RPPA antibody panel is largely unannotated, which is a property of
the source data, not of the curation.

`n_features` counts the **ranked tokens** for that modality — the variance-
filtered features the transformer actually saw — not the size of the raw
archive block (374 / 741 / 189). `cohort.block_shapes` in `/api/overview`
carries the raw counts.

---

## GET `/api/config`

The full resolved configuration actually used for the run, grouped as
`cohort`, `preprocess`, `quantum`, `model`, `train`, `baseline`, `app`. This is
the reproducibility surface: it is read from the checkpoint, so it always
describes the model being served rather than the current source defaults.

```json
{
  "cohort": {"study_id": "ucec_tcga_gdc", "target_name": "early_stage_detection",
             "xena_rppa_url": "https://tcga.xenahubs.net/download/TCGA.UCEC.sampleMap%2FRPPA.gz"},
  "preprocess": {"variance_top_frac": 0.6, "n_pcs_transcriptome": 24,
                 "n_pcs_genome": 12, "n_pcs_proteome": 12, "fit_within_fold": true},
  "quantum": {"enabled": true, "n_qubits": 6, "feature_map": "iqp", "n_layers": 2,
              "n_input_features": 12, "c_regularisation": 1.0},
  "model": {"d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.25},
  "train": {"n_splits": 5, "n_repeats": 2, "epochs": 40, "patience": 12, "seed": 20240517},
  "app": {"host": "127.0.0.1", "port": 8000, "max_upload_mb": 25}
}
```

---

## Error handling

| Status | When |
|---|---|
| 404 | unknown route |
| 422 | malformed request body (FastAPI default shape) |
| 503 | model artefacts not present — `{"detail": "model not trained; run scripts/train.py"}` |
| 500 | unexpected — `{"detail": "internal error"}` |

## Static asset routes

- `GET /` → `qomics/web/index.html`
- `GET /assets/*` → `qomics/web/assets/*`
