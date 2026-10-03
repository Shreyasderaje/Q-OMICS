"""Tests for the FastAPI serving layer.

These run against a synthetic artefact directory so the suite stays fast and
deterministic; no network access and no trained model are required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from qomics.serving.app import clean

# --------------------------------------------------------------------------
# JSON hygiene
# --------------------------------------------------------------------------


def test_clean_converts_nan_and_infinity_to_null():
    """NaN is not valid JSON and would break the whole response in a browser."""
    payload = {"a": float("nan"), "b": float("inf"), "c": [1.0, float("-inf")], "d": 1.5}
    cleaned = clean(payload)
    assert cleaned["a"] is None
    assert cleaned["b"] is None
    assert cleaned["c"][1] is None
    assert cleaned["d"] == 1.5
    # The result must survive a strict serialisation round-trip.
    json.dumps(cleaned, allow_nan=False)


def test_clean_handles_numpy_scalars_and_arrays():
    cleaned = clean({
        "i": np.int64(3),
        "f": np.float32(1.25),
        "b": np.bool_(True),
        "arr": np.array([1.0, 2.0]),
    })
    assert cleaned["i"] == 3 and isinstance(cleaned["i"], int)
    assert cleaned["f"] == pytest.approx(1.25)
    assert cleaned["b"] is True
    assert cleaned["arr"] == [1.0, 2.0]


# --------------------------------------------------------------------------
# Artefact fixtures
# --------------------------------------------------------------------------


def _write_artifacts(root: Path) -> Path:
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    run = {
        "version": "1.0.0",
        "generated_at": "2026-02-11T09:31:04Z",
        "runtime_seconds": 123.4,
        "cohort": {
            "study": "TCGA-UCEC (GDC, 2025)",
            "study_id": "ucec_tcga_gdc",
            "n_samples": 463,
            "n_patients": 462,
            "target": "early_stage_detection",
            "target_description": "FIGO stage I vs stage II-IV",
            "block_shapes": {"transcriptome": 374, "genome": 740, "proteome": 189},
            "class_counts": {"class_0": 340, "class_1": 123},
            "class_balance": 0.2657,
        },
        "metrics": {
            "target": "FIGO stage I vs stage II-IV endometrial carcinoma",
            "n_splits": 10,
            "primary": "hybrid_transformer",
            "models": [
                {
                    "name": "Hybrid Transformer (Q-OMICS)", "family": "hybrid",
                    "auc_mean": 0.71, "auc_std": 0.04, "auc_ci_low": 0.66,
                    "auc_ci_high": 0.76, "pr_auc_mean": 0.70, "pr_auc_std": 0.05,
                    "f1_mean": 0.65, "f1_std": 0.06, "sensitivity_mean": 0.70,
                    "sensitivity_std": 0.07, "specificity_mean": 0.75,
                    "specificity_std": 0.06, "brier_mean": 0.22, "brier_std": 0.03,
                    "seconds_mean": 40.0,
                    "folds": [{"repeat": 0, "fold": 0, "auc": 0.70, "n_test": 92}],
                    "_oof_scores": [], "_oof_ids": [],
                },
                {
                    "name": "RBF SVM", "family": "kernel",
                    "auc_mean": 0.66, "auc_std": 0.05, "auc_ci_low": 0.61,
                    "auc_ci_high": 0.71, "pr_auc_mean": 0.65, "pr_auc_std": 0.06,
                    "f1_mean": 0.61, "f1_std": 0.07, "sensitivity_mean": 0.64,
                    "sensitivity_std": 0.08, "specificity_mean": 0.77,
                    "specificity_std": 0.05, "brier_mean": 0.23, "brier_std": 0.03,
                    "seconds_mean": 0.4,
                    "folds": [{"repeat": 0, "fold": 0, "auc": 0.65, "n_test": 92}],
                },
            ],
            "significance": [
                {
                    "comparison": "Hybrid Transformer (Q-OMICS) vs RBF SVM",
                    "mean_delta": 0.05, "t_statistic": 2.5, "p_value": 0.03,
                    "n_pairs": 10, "significant_at_0.05": True,
                    "significant_after_correction": True, "reference": "rbf_svm",
                }
            ],
            "roc_curve": {
                "labels": ["class_0", "class_1"],
                "curves": [
                    {"name": "Hybrid Transformer (Q-OMICS)", "fpr": [0.0, 0.5, 1.0],
                     "tpr": [0.0, 0.7, 1.0]},
                    {"name": "RBF SVM", "fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.6, 1.0]},
                ],
            },
        },
        "quantum": {
            "backend": "pennylane.default.qubit", "n_qubits": 6,
            "hilbert_dimension": 64, "feature_map": "iqp", "n_layers": 2,
            "reupload": True, "normalisation": "arc_cosine_scaled",
            "kernel_diagnostics": {"mean": 0.01, "effective_rank": 300.0},
            "labelled_separation": {"quantum": 0.02, "rbf": 0.01},
            "geometric_difference": {
                "g_quantum_vs_rbf": 1.03, "g_quantum_vs_cosine": 0.98,
                "interpretation": "g = 1.030 exceeds 1, so the quantum geometry is "
                                  "not reproducible by this classical kernel family.",
            },
        },
        "architecture": {
            "stages": [
                {"id": "ingest", "title": "Cohort ingestion", "detail": "x", "tokens": 0},
                {"id": "quantum", "title": "Fidelity quantum kernel", "detail": "y", "tokens": 0},
            ],
            "model": {"d_model": 64, "n_heads": 4, "n_layers": 2},
            "training": {"epochs": 20, "batch_size": 64, "loss": "focal", "n_splits": 10},
            "blocks": {"transcriptome": 80, "genome": 72, "proteome": 40},
            "total_tokens": 192,
        },
        "biomarker_summary": {
            "method": "attention_rollout", "n_features_considered": 192,
            "modality_contribution": [{"block": "genome", "importance": 0.4}],
            "top_axes": [{"axis": "pi3k_pten", "importance": 0.1}],
        },
        "config": {"quantum": {"n_qubits": 6, "feature_map": "iqp", "n_layers": 2,
                               "reupload": True, "data_scale": 1.0,
                               "add_ry_rotation": True},
                   "model": {"d_model": 64, "n_heads": 4, "n_layers": 2}},
    }
    (root / "artifacts" / "run.json").write_text(json.dumps(run), encoding="utf-8")

    pd.DataFrame([
        {"rank": 1, "feature": "MUT::PTEN", "symbol": "PTEN", "block": "genome",
         "axis": "pi3k_pten", "importance": 0.05, "importance_normalised": 0.2,
         "importance_std": 0.01, "stability": 0.9, "effect_size": -0.6,
         "p_value": 0.001, "q_value": 0.02, "fdr_significant": True,
         "composite": 0.07, "direction": "depleted_in_early_stage",
         "mean_early": -0.4, "mean_advanced": 0.1, "n_observed": 463},
        {"rank": 2, "feature": "MKI67", "symbol": "MKI67", "block": "transcriptome",
         "axis": "cell_cycle", "importance": 0.03, "importance_normalised": 0.12,
         "importance_std": 0.005, "stability": 0.7, "effect_size": 0.4,
         "p_value": 0.02, "q_value": 0.2, "fdr_significant": False,
         "composite": 0.03, "direction": "enriched_in_early_stage",
         "mean_early": 0.3, "mean_advanced": -0.1, "n_observed": 463},
    ]).to_csv(root / "artifacts" / "biomarkers.csv", index=False)

    pd.DataFrame([
        {"patient_id": "TCGA-1", "figo_stage": "IA", "figo_label": "Stage I (early)",
         "histological_type": "Endometrioid", "age_at_diagnosis": 63.0, "sample_type": "tumor"},
        {"patient_id": "TCGA-2", "figo_stage": "II", "figo_label": "Stage II-IV (advanced)",
         "histological_type": "Serous", "age_at_diagnosis": 71.0, "sample_type": "tumor"},
    ], index=["TCGA-1-01", "TCGA-2-01"]).to_csv(root / "artifacts" / "cohort_clinical.csv")
    return root / "artifacts"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient bound to a synthetic artefact directory."""
    import qomics.serving.store as store_module
    from qomics.serving.app import app

    _write_artifacts(tmp_path)
    monkeypatch.setenv("QOMICS_ARTIFACT_DIR", str(tmp_path / "artifacts"))

    fresh = store_module.ArtifactStore(root=tmp_path / "artifacts").load(with_bundle=False)
    monkeypatch.setattr(store_module, "_STORE", fresh, raising=False)
    monkeypatch.setattr("qomics.serving.app.get_store", lambda: fresh)

    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


def test_health_reports_loaded_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["cohort"] == "TCGA-UCEC (GDC, 2025)"


def test_overview_shape_and_json_safety(client):
    response = client.get("/api/overview")
    assert response.status_code == 200
    body = response.json()
    assert body["product"]["name"] == "Q-OMICS"
    assert body["cohort"]["n_samples"] == 463
    assert body["headline_metrics"]["hybrid_auc_mean"] == pytest.approx(0.71)
    assert body["headline_metrics"]["n_splits"] == 10
    # Strict parse: no NaN tokens anywhere.
    json.dumps(body, allow_nan=False)


def test_metrics_lists_models_and_significance(client):
    body = client.get("/api/metrics").json()
    names = [m["name"] for m in body["models"]]
    assert "Hybrid Transformer (Q-OMICS)" in names
    assert "RBF SVM" in names
    assert body["significance"][0]["significant_at_0.05"] is True
    assert body["roc_curve"]["curves"][0]["fpr"] == [0.0, 0.5, 1.0]


def test_biomarkers_sorted_and_annotated(client):
    body = client.get("/api/biomarkers?limit=10").json()
    assert body["returned"] == 2
    first = body["biomarkers"][0]
    assert first["rank"] == 1
    assert first["symbol"] == "PTEN"
    assert first["direction"] == "depleted_in_early_stage"
    assert first["modality"] == first["block"] == "genome"
    assert first["importance_normalised"] == pytest.approx(0.2)
    assert first["fdr_significant"] is True


def test_biomarkers_block_filter(client):
    body = client.get("/api/biomarkers?block=genome").json()
    assert body["returned"] == 1
    assert body["biomarkers"][0]["block"] == "genome"


def test_biomarker_detail_hit_and_miss(client):
    hit = client.get("/api/biomarkers/PTEN").json()
    assert hit["present"] is True
    assert hit["block"] == "genome"
    # A miss is a legitimate result, not a 404.
    miss = client.get("/api/biomarkers/NOT_A_GENE")
    assert miss.status_code == 200
    assert miss.json()["present"] is False


def test_quantum_payload_includes_geometry(client):
    body = client.get("/api/quantum").json()
    assert body["n_qubits"] == 6
    assert body["hilbert_dimension"] == 64
    assert body["geometric_difference"]["g_quantum_vs_rbf"] == pytest.approx(1.03)
    # The endpoint computes a live state-vector preview.
    assert body["state_vector_visualisation"]["dim"] == 64
    assert len(body["state_vector_visualisation"]["amplitudes"]) == 2


def test_architecture_and_features(client):
    arch = client.get("/api/architecture").json()
    assert len(arch["stages"]) == 2
    assert arch["model"]["d_model"] == 64

    features = client.get("/api/features").json()
    assert features["n_total"] == 2
    assert {b["block"] for b in features["blocks"]} == {"genome", "transcriptome"}


def test_cohort_paging_and_distributions(client):
    body = client.get("/api/cohort?limit=1&offset=0").json()
    assert body["total"] == 2
    assert len(body["rows"]) == 1
    assert body["rows"][0]["sample_id"] == "TCGA-1-01"
    assert {d["stage"] for d in body["stage_distribution"]} == {"IA", "II"}

    second = client.get("/api/cohort?limit=1&offset=1").json()
    assert second["rows"][0]["sample_id"] == "TCGA-2-01"


def test_cohort_csv_download(client):
    response = client.get("/api/cohort/download")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "attachment" in response.headers.get("content-disposition", "")
    assert "sample_id" in response.text
    assert "TCGA-1-01" in response.text


def test_cohort_download_as_json(client):
    response = client.get("/api/cohort/download?format=json")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload[0]["sample_id"] == "TCGA-1-01"
    json.dumps(payload, allow_nan=False)


def test_unknown_api_endpoint_returns_json_404(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_index_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_without_bundle_returns_503(client):
    """A missing checkpoint must be a clear 503, not a 500 traceback."""
    response = client.post("/api/predict", json={"features": {"PTEN": 1.0}})
    assert response.status_code == 503
    assert "checkpoint" in response.json()["detail"]


def test_predict_rejects_an_empty_sample(client):
    response = client.post("/api/predict", json={"features": {}})
    assert response.status_code in (422, 503)


# --------------------------------------------------------------------------
# Degraded mode
# --------------------------------------------------------------------------


def test_missing_artifacts_degrade_to_503(tmp_path, monkeypatch):
    import qomics.serving.store as store_module
    from qomics.serving.app import app

    empty = store_module.ArtifactStore(root=tmp_path / "nothing").load()
    assert empty.ready is False
    monkeypatch.setattr("qomics.serving.app.get_store", lambda: empty)

    with TestClient(app) as test_client:
        assert test_client.get("/api/health").json()["status"] == "degraded"
        response = test_client.get("/api/metrics")
        assert response.status_code == 503
        assert "train" in response.json()["detail"]
