"""Tests for the transformer encoder, evaluation maths and biomarker ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from qomics.models.baselines import DISPLAY_NAMES, FAMILY  # noqa: E402
from qomics.models.transformer import (  # noqa: E402
    MultiOmicsTransformer,
    count_weights,
    focal_loss,
    parameter_count,
)
from qomics.pipeline.biomarkers import (  # noqa: E402
    BiomarkerPanel,
    build_panel,
    cohens_d,
    fdr_bh,
    top_decile_membership,
)
from qomics.pipeline.evaluate import (  # noqa: E402
    bootstrap_ci,
    classification_metrics,
    holm_bonferroni,
    paired_t_test,
    youden_threshold,
)
from qomics.pipeline.geometry import (  # noqa: E402
    geometric_difference,
    interpret_geometric_difference,
)

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def _model(**kwargs) -> MultiOmicsTransformer:
    defaults = {
        "block_sizes": {"transcriptome": 12, "genome": 10, "proteome": 6},
        "d_model": 32, "n_heads": 4, "n_layers": 2, "dim_feedforward": 64, "dropout": 0.1,
        "n_fusion": 0, "max_tokens": 256,
    }
    defaults.update(kwargs)
    return MultiOmicsTransformer(**defaults)


def test_forward_shapes_and_token_budget():
    model = _model()
    blocks = {k: torch.randn(5, n) for k, n in model.block_sizes.items()}
    out = model(blocks)
    assert out["logits"].shape == (5, 2)
    assert out["token_embeddings"].shape == (5, 1 + 12 + 10 + 6, 32)


def test_fusion_token_is_appended_when_requested():
    model = _model(n_fusion=1)
    blocks = {k: torch.randn(4, n) for k, n in model.block_sizes.items()}
    fusion = torch.randn(4, 1)
    out = model(blocks, fusion, need_attention=True)
    assert out["logits"].shape == (4, 2)
    assert out["attention"].shape[0] == 4
    assert out["attention"].shape[2] == 4  # heads
    assert out["attention"].shape[-1] == model.n_tokens


def test_missing_fusion_vector_raises_clearly():
    model = _model(n_fusion=1)
    blocks = {k: torch.randn(2, n) for k, n in model.block_sizes.items()}
    with pytest.raises(ValueError, match="fusion"):
        model(blocks, None)


def test_sequence_limit_is_enforced():
    with pytest.raises(ValueError, match="max_tokens"):
        _model(max_tokens=4)


def test_attention_rollout_yields_per_token_importance():
    model = _model()
    blocks = {k: torch.randn(3, n) for k, n in model.block_sizes.items()}
    out = model(blocks, need_attention=True)
    importance = model.token_importance(out["attention"])
    assert importance.shape == (3, model.n_tokens)
    assert torch.isfinite(importance).all()
    assert (importance >= -1e-6).all()  # attention weights are non-negative


def test_block_importance_covers_every_modality():
    model = _model()
    blocks = {k: torch.randn(2, n) for k, n in model.block_sizes.items()}
    out = model(blocks, need_attention=True)
    importance = model.token_importance(out["attention"])
    per_block = model.block_importance(importance)
    assert set(per_block) == set(model.block_sizes)


def test_focal_loss_downweights_easy_examples():
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    targets = torch.tensor([0, 1])
    confident = float(focal_loss(logits, targets))
    uncertain = float(focal_loss(torch.zeros(2, 2), targets))
    assert confident < uncertain


def test_count_weights_favour_minority_class():
    """Inverse-frequency weights must up-weight the rare class.

    scikit-learn's "balanced" weighting ``n / (k * n_i)`` is a ratio, not a
    mean-one normalisation, so the assertion is on the ratio between the
    classes rather than on the absolute scale.
    """
    weights = count_weights(np.array([0] * 8 + [1] * 2))
    assert weights[1] > weights[0]
    assert float(weights[1] / weights[0]) == pytest.approx(8 / 2, rel=1e-5)


def test_count_weights_are_balanced_for_equal_classes():
    weights = count_weights(np.array([0, 0, 1, 1]))
    assert weights[0] == pytest.approx(weights[1])


def test_parameter_count_is_positive_and_finite():
    assert parameter_count(_model()) > 1000


# --------------------------------------------------------------------------
# Evaluation maths
# --------------------------------------------------------------------------


def test_metrics_on_a_perfect_ranker():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.10, 0.15, 0.85, 0.90, 0.95])
    m = classification_metrics(y, scores)
    assert m["auc"] == pytest.approx(1.0)
    assert m["sensitivity"] == pytest.approx(1.0)
    assert m["specificity"] == pytest.approx(1.0)
    assert m["brier"] < 0.05


def test_metrics_on_an_inverted_ranker():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))
    assert m["auc"] == pytest.approx(0.0)


def test_single_class_input_is_reported_not_crashed():
    m = classification_metrics(np.zeros(5, dtype=int), np.random.default_rng(0).random(5))
    assert np.isnan(m["auc"])


def test_youden_threshold_separates_a_clean_split():
    y = np.array([0, 0, 1, 1])
    t = youden_threshold(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.2 < t <= 0.8


def test_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 120)
    scores = y * 0.7 + rng.random(120) * 0.3
    mean, low, high = bootstrap_ci(y, scores, n_boot=200, seed=1)
    assert low <= mean <= high


def test_paired_t_test_detects_a_consistent_shift():
    a = np.array([0.80, 0.82, 0.81, 0.83, 0.79, 0.84, 0.82, 0.81])
    b = a - 0.05
    result = paired_t_test(a, b)
    assert result["mean_delta"] == pytest.approx(0.05)
    assert result["p_value"] < 0.05
    assert result["significant_at_0.05"] is True


def test_paired_t_test_reports_no_difference():
    a = np.array([0.80, 0.82, 0.79, 0.83, 0.81, 0.80])
    result = paired_t_test(a, a.copy())
    assert result["mean_delta"] == pytest.approx(0.0)
    assert result["p_value"] > 0.5


def test_paired_t_test_needs_three_pairs():
    result = paired_t_test(np.array([0.8, 0.9]), np.array([0.7, 0.8]))
    assert result["n_pairs"] == 2
    assert np.isnan(result["p_value"])


def test_holm_bonferroni_controls_family_wise_error():
    flags = holm_bonferroni([0.001, 0.04, 0.30, 0.9])
    assert flags[0] is True
    assert flags[3] is False


def test_student_t_survival_function_matches_scipy():
    """The p-value drives every significance claim, so it is checked directly."""
    from scipy import stats

    from qomics.pipeline.evaluate import _t_sf_two_sided

    for t, df in [(0.0, 10), (1.0, 10), (2.228, 10), (2.5, 9), (3.0, 25), (1.96, 100), (4.0, 50)]:
        assert _t_sf_two_sided(t, df) == pytest.approx(2 * stats.t.sf(t, df), abs=1e-10)
    # The canonical textbook value: t = 2.228, df = 10 is the 5 % two-sided point.
    assert _t_sf_two_sided(2.228, 10) == pytest.approx(0.05, abs=1e-3)
    assert np.isnan(_t_sf_two_sided(1.0, 0))


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_geometric_difference_is_one_for_identical_kernels():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(20, 4))
    K = X @ X.T + 1e-3 * np.eye(20)
    g = geometric_difference(K, K)
    assert g == pytest.approx(1.0, abs=0.05)


def test_geometric_difference_exceeds_one_for_disparate_kernels():
    rng = np.random.default_rng(5)
    K_q = np.eye(20) * 0.9 + 0.1
    X = rng.normal(size=(20, 3))
    K_c = X @ X.T + 1e-2 * np.eye(20)
    assert geometric_difference(K_q, K_c) > 1.0


def test_interpretation_is_honest_below_one():
    text = interpret_geometric_difference(0.90)
    assert "below 1" in text
    assert interpret_geometric_difference(1.5).startswith("g = 1.500")


# --------------------------------------------------------------------------
# Biomarkers
# --------------------------------------------------------------------------


def test_cohens_d_sign_and_magnitude():
    a = np.array([1.0, 1.1, 0.9, 1.05])
    b = np.array([3.0, 3.1, 2.9, 3.05])
    assert cohens_d(a, b) < -1.0
    assert cohens_d(b, a) > 1.0
    assert cohens_d(a, a) == pytest.approx(0.0, abs=1e-9)


def test_cohens_d_tolerates_short_input():
    assert cohens_d(np.array([1.0]), np.array([2.0, 3.0])) == 0.0


def test_fdr_bh_rejects_the_real_association():
    p = np.array([1e-8, 1e-6, 0.4, 0.8, 0.99])
    reject, q = fdr_bh(p, alpha=0.05)
    assert reject[0] and reject[1]
    assert not reject[-1]
    # q-values are monotone in p and never below the raw p-value.
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)
    assert (q >= p - 1e-12).all()
    assert (q <= 1.0).all()


def test_fdr_bh_rejects_nothing_when_all_p_values_are_large():
    reject, q = fdr_bh(np.array([0.5, 0.6, 0.7, 0.8]), alpha=0.05)
    assert not reject.any()
    assert (q <= 1.0).all()


def test_top_decile_membership_counts_repeats():
    importance = [
        np.array([0.9, 0.8, 0.1, 0.2, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]),
        np.array([0.1, 0.2, 0.9, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0, 0.0]),
    ]
    stability = top_decile_membership(importance, fraction=0.1)
    assert stability[0] == pytest.approx(0.5)
    assert stability[2] == pytest.approx(0.5)


def test_build_panel_ranks_and_annotates():
    n, p = 40, 6
    rng = np.random.default_rng(6)
    y = np.array([0] * 20 + [1] * 20)
    strong = np.concatenate([rng.normal(-1.5, 0.2, 20), rng.normal(1.5, 0.2, 20)])
    matrix = np.column_stack([strong] + [rng.normal(size=n) for _ in range(p - 1)])

    importance = np.zeros((n, p))
    importance[:, 0] = 1.0

    panel = build_panel(
        importance, [f"f{i}" for i in range(p)], ["transcriptome"] * p,
        pd.DataFrame({"feature": [f"f{i}" for i in range(p)],
                      "axis": ["cell_cycle"] * p}),
        y, {"transcriptome": matrix}, {"transcriptome": [f"f{i}" for i in range(p)]},
        stability=np.array([1.0, 0.2, 0.1, 0.0, 0.0, 0.0]),
    )
    assert isinstance(panel, BiomarkerPanel)
    assert panel.table.iloc[0]["feature"] == "f0"
    assert panel.table.iloc[0]["effect_size"] < 0  # low in early stage
    assert panel.table.iloc[0]["direction"] == "depleted_in_early_stage"
    assert panel.table["rank"].tolist() == list(range(1, p + 1))
    assert panel.block_summary()[0]["block"] == "transcriptome"


def test_panel_normalised_importance_sums_to_one():
    n, p = 30, 5
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, n)
    matrix = rng.normal(size=(n, p))
    panel = build_panel(
        np.abs(rng.normal(size=(n, p))), [f"g{i}" for i in range(p)], ["genome"] * p,
        pd.DataFrame(columns=["feature", "axis"]), y,
        {"genome": matrix}, {"genome": [f"g{i}" for i in range(p)]},
    )
    assert panel.table["importance_normalised"].sum() == pytest.approx(1.0, abs=1e-6)


def test_panel_importance_is_a_unit_sum_per_sample_distribution():
    """Regression test: normalising per extraction chunk rescales the values.

    The trainer scores the test set in chunks to bound attention memory.  If
    the pooled importance is normalised over the whole block instead of per
    sample, every reported value is divided by the chunk size and the numbers
    silently depend on an implementation detail.
    """
    from qomics.pipeline.experiment import _build_biomarker_panel

    rng = np.random.default_rng(0)
    n, p = 50, 6
    names = [f"f{i}" for i in range(p)]
    # Three folds, each delivered as two 16-row chunks, as the trainer does.
    runs = [rng.random((16, p)) for _ in range(6)]
    y = rng.integers(0, 2, n)
    dense = {"transcriptome": rng.normal(size=(n, p))}

    panel = _build_biomarker_panel(
        runs, names, ["transcriptome"] * p, dense, {"transcriptome": names},
        y, pd.DataFrame({"feature": names, "axis": ["x"] * p}),
    )
    total = float(panel.table["importance"].sum())
    assert total == pytest.approx(1.0, abs=1e-6)
    # A unit-sum distribution over 6 tokens puts the mean near 1/6.
    assert 1.0 / p * 0.5 < float(panel.table["importance"].mean()) < 1.0 / p * 1.5
    assert float(panel.table["importance_normalised"].sum()) == pytest.approx(1.0, abs=1e-9)


def test_panel_falls_back_when_no_attention_was_collected():
    from qomics.pipeline.experiment import _build_biomarker_panel

    rng = np.random.default_rng(1)
    n, p = 30, 4
    names = [f"g{i}" for i in range(p)]
    y = rng.integers(0, 2, n)
    matrix = rng.normal(size=(n, p)) * np.arange(1, p + 1)
    panel = _build_biomarker_panel(
        [], names, ["genome"] * p, {"genome": matrix}, {"genome": names},
        y, pd.DataFrame({"feature": names, "axis": ["x"] * p}),
    )
    assert panel.method == "variance_fallback"
    # The highest-variance column must lead.
    assert panel.table.iloc[0]["feature"] == names[-1]


def test_display_names_cover_every_reported_model():
    for key in ("hybrid_transformer", "classical_transformer", "quantum_svm",
                "rbf_svm", "logreg", "random_forest", "hist_gb", "pca_logreg_tx"):
        assert key in DISPLAY_NAMES
        assert key in FAMILY
