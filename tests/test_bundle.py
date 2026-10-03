"""Tests for the deployable model bundle: save, reload, and score.

The serving path is where a partial artefact becomes a silent failure.  If
the SVM or the kernel's reference norms are not persisted, the endpoint
returns 503 and nothing in the training run looks wrong.  These tests fit a
small bundle, round-trip it through disk, and require the restored model to
reproduce the original prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from qomics.pipeline.final_model import train_final_model  # noqa: E402
from qomics.pipeline.geometry import (  # noqa: E402
    geometric_difference,
    interpret_geometric_difference,
)
from qomics.pipeline.preprocess import MultiOmicsPreprocessor  # noqa: E402


def _synthetic_cohort(n: int = 90, seed: int = 0) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    """A small three-block cohort with a real but weak signal."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(-0.8, 1.0, n)
    y = (signal > 0).astype(int)

    transcriptome = pd.DataFrame(
        signal[:, None] + rng.normal(0, 1.0, (n, 10)), columns=[f"tx{i}" for i in range(10)]
    )
    genome = pd.DataFrame(
        (rng.random((n, 8)) < (0.2 + 0.3 * y)[:, None]).astype(float),
        columns=[f"MUT::G{i}" for i in range(8)],
    )
    proteome = pd.DataFrame(
        rng.normal(0, 1.0, (n, 6)) + 0.5 * y[:, None], columns=[f"p{i}" for i in range(6)]
    )
    return {"transcriptome": transcriptome, "genome": genome, "proteome": proteome}, y


def _small_config() -> object:
    from dataclasses import replace

    from qomics.config import CONFIG

    return replace(
        CONFIG,
        quantum=replace(CONFIG.quantum, n_qubits=3, n_layers=1, n_input_features=6),
        model=replace(CONFIG.model, d_model=16, n_heads=2, n_layers=1, dim_feedforward=32,
                      max_tokens=64),
        train=replace(CONFIG.train, epochs=3, batch_size=16, patience=3),
    )


def _fit(tmp_path, cfg):
    blocks, y = _synthetic_cohort()
    pre = MultiOmicsPreprocessor(
        blocks,
        max_features_per_block={"transcriptome": 8, "genome": 6, "proteome": 4},
        fused_components={"transcriptome": 3, "genome": 2, "proteome": 1},
    )
    pre.fit(None)
    bundle = train_final_model(pre, blocks, y, cfg, epochs=3)
    return bundle, blocks, y


def test_bundle_scores_a_partial_sample(tmp_path):
    bundle, _blocks, _y = _fit(tmp_path, _small_config())
    # Only three features supplied out of eighteen; the rest are imputed.
    result = bundle.score({"tx0": 2.0, "MUT::G0": 1.0, "p0": 0.5})
    assert result["prediction"] in {"early_stage", "advanced_stage"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["risk_tier"] in {"low", "moderate", "elevated"}
    assert abs(result["probabilities"]["stage_I_early"]
               + result["probabilities"]["stage_II_IV_advanced"] - 1.0) < 1e-6
    assert len(result["top_contributors"]) == 10


def test_bundle_persists_the_svm_and_reference(tmp_path):
    """Regression test for the 503 at serve time.

    Without the fitted SVM and the fused reference matrix the scoring path
    cannot be reconstructed, and the API degrades to 503 even though the
    training run reported success.
    """
    bundle, _blocks, _y = _fit(tmp_path, _small_config())
    path = bundle.save(tmp_path / "bundle.pt")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["svm"] is not None, "SVM must be persisted for serve-time scoring"
    assert payload["fused_reference"] is not None, "fused reference matrix must be persisted"
    assert payload["kernel_reference_norm"] is not None, "kernel state norms must be persisted"
    assert payload["quantum"]["n_qubits"] == 3


def test_bundle_round_trip_reproduces_the_prediction(tmp_path, monkeypatch):
    import qomics.serving.store as store_module

    bundle, _blocks, _y = _fit(tmp_path, _small_config())
    # The store looks for `model_bundle.pt` by name, so the test must save
    # under the same filename the training CLI writes.
    bundle.save(tmp_path / "model_bundle.pt")
    original = bundle.score({"tx0": 1.5, "p0": 1.0})

    store = store_module.ArtifactStore(root=tmp_path)
    store._load_bundle()
    assert store.bundle is not None
    assert store.bundle.svm is not None, "reloaded bundle must carry a usable SVM"

    restored = store.bundle.score({"tx0": 1.5, "p0": 1.0})
    assert restored["prediction"] == original["prediction"]
    assert restored["confidence"] == pytest.approx(original["confidence"], abs=1e-5)


def test_bundle_ignores_unknown_features_and_imputes_missing(tmp_path):
    bundle, _blocks, _y = _fit(tmp_path, _small_config())
    result = bundle.score({"TOTALLY_UNKNOWN_GENE": 99.0, "tx0": 1.0})
    assert result["prediction"] in {"early_stage", "advanced_stage"}
    # The unknown key must not appear as a contributor.
    assert all(c["name"] != "TOTALLY_UNKNOWN_GENE" for c in result["top_contributors"])


def test_bundle_reports_contributors_within_known_features(tmp_path):
    bundle, _blocks, _y = _fit(tmp_path, _small_config())
    result = bundle.score({"tx0": 1.0})
    known = set(bundle.feature_order)
    assert all(c["name"] in known for c in result["top_contributors"])


# --------------------------------------------------------------------------
# Geometry interpretation
# --------------------------------------------------------------------------


def test_geometric_difference_is_bounded_for_matched_kernels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    K = X @ X.T + 0.5 * np.eye(30)
    g = geometric_difference(K, K)
    # A relative eigenvalue floor keeps g near 1 for identical kernels; an
    # absolute floor previously sent this to 181 for near-singular inputs.
    assert 0.5 < g < 2.0


def test_interpretation_refuses_to_claim_advantage_without_separation():
    """A large g with no label signal must not be reported as an advantage."""
    text = interpret_geometric_difference(180.0, quantum_separation=1e-9,
                                         classical_separation=-1e-4)
    assert "no label separation" in text
    assert "not evidence of an advantage" in text
    assert "does not outperform" in text


def test_interpretation_flags_large_g_as_conditioning_artefact():
    text = interpret_geometric_difference(180.0, quantum_separation=0.05,
                                         classical_separation=0.02)
    assert "artefact" in text
    assert "not evidence" in text


def test_interpretation_reports_equivalence_near_one():
    assert "within 5% of 1" in interpret_geometric_difference(1.02, quantum_separation=0.05,
                                                             classical_separation=0.05)


def test_interpretation_reports_structural_difference_for_modest_g():
    text = interpret_geometric_difference(1.4, quantum_separation=0.05,
                                         classical_separation=0.02)
    assert "not reproducible" in text
