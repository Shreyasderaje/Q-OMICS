"""Tests for data ingestion, FIGO parsing and cohort assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qomics.data.cohort import parse_figo_stage, stage_to_label
from qomics.data.xena import sample_type_of
from qomics.genes import build_symbol_panels, flatten, symbol_to_axis

# --------------------------------------------------------------------------
# FIGO stage parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Stage I", "I"),
        ("Stage IA", "I"),
        ("Stage IB", "I"),
        ("Stage IC", "I"),
        ("Stage IC1", "I"),
        ("Stage II", "II"),
        ("Stage IIA", "II"),
        ("Stage IIIA", "III"),
        ("Stage IIIB", "III"),
        # IIIC is stage III with a nodal subcategory; the base stage is III.
        ("Stage IIIC1", "III"),
        ("Stage IIIC2", "III"),
        ("Stage IIIC", "III"),
        ("Stage IV", "IV"),
        ("Stage IVB", "IV"),
        ("stage iva", "IV"),
    ],
)
def test_parse_figo_stage(raw, expected):
    assert parse_figo_stage(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "unknown", "Indeterminate", "NA", float("nan")])
def test_parse_figo_stage_rejects_unknown(raw):
    assert parse_figo_stage(raw) is None


def test_parse_figo_stage_only_returns_base_stages():
    for raw in ("Stage IA", "Stage IIIB", "Stage IIIC2", "Stage IVB", "Stage II"):
        assert parse_figo_stage(raw) in {"I", "II", "III", "IV"}


def test_stage_to_label_boundaries():
    assert stage_to_label("I") == 0
    assert stage_to_label("II") == 1
    assert stage_to_label("III") == 1
    assert stage_to_label("IV") == 1
    assert stage_to_label(None) is None
    assert stage_to_label("V") is None


@pytest.mark.parametrize("raw", ["Stage IIIC1", "Stage IIIC2", "Stage IIIC", "Stage IIIB"])
def test_stage_subcategories_are_not_dropped(raw):
    """Stage IIIC/IIIB must be classified advanced, not discarded.

    Regression test: an unrecognised stage is silently excluded from the
    cohort, which would quietly shrink the advanced class.
    """
    assert stage_to_label(parse_figo_stage(raw)) == 1


def test_label_boundary_is_not_a_string_prefix_match():
    """'Stage II' must not be parsed as stage I by a naive prefix test."""
    assert parse_figo_stage("Stage II") == "II"
    assert stage_to_label(parse_figo_stage("Stage II")) == 1
    assert parse_figo_stage("Stage IIIA") == "III"
    assert stage_to_label(parse_figo_stage("Stage IIIA")) == 1


# --------------------------------------------------------------------------
# Xena helpers
# --------------------------------------------------------------------------


def test_sample_type_mapping():
    clinical = pd.DataFrame(
        {
            "sample_type": [
                "Primary Tumor", "Solid Tissue Normal", "Recurrent Tumor", "Other",
            ]
        },
        index=["A-01", "A-11", "A-02", "A-99"],
    )
    result = sample_type_of(clinical)
    assert result["A-01"] == "tumor"
    assert result["A-11"] == "normal"
    assert result["A-02"] == "tumor"
    assert result["A-99"] == "other"


def test_sample_type_from_numeric_codes():
    clinical = pd.DataFrame(
        {"sample_type_id": ["01", "11", "02"]}, index=["A", "B", "C"]
    )
    result = sample_type_of(clinical)
    assert list(result) == ["tumor", "normal", "tumor"]


# --------------------------------------------------------------------------
# Gene panel
# --------------------------------------------------------------------------

EXPECTED_DRIVERS = {"PTEN", "PIK3CA", "ARID1A", "TP53", "CTNNB1", "POLE", "MLH1", "KRAS"}


def test_panel_covers_known_ucec_drivers():
    from qomics.genes import GENOME_PANEL, TRANSCRIPTOME_PANEL

    symbols = set(flatten(GENOME_PANEL)) | set(flatten(TRANSCRIPTOME_PANEL))
    missing = EXPECTED_DRIVERS - symbols
    assert not missing, f"panel is missing known UCEC drivers: {missing}"


def test_build_symbol_panels_returns_flat_lists():
    panels = build_symbol_panels()
    assert set(panels) == {"genome", "transcriptome"}
    for name, symbols in panels.items():
        assert isinstance(symbols, list), f"{name} should be a flat list"
        assert len(symbols) == len(set(symbols)), f"{name} contains duplicates"
        assert len(symbols) > 100, f"{name} has an unexpected size: {len(symbols)}"
    # The transcriptomic panel is the one the model tokenises, and it is
    # deliberately larger than the genomic panel.
    assert len(panels["transcriptome"]) > len(panels["genome"])


def test_flatten_is_deduplicated_and_order_preserving():
    flat = flatten({"a": ("X", "Y"), "b": ("Y", "Z")})
    assert flat == ["X", "Y", "Z"]


def test_symbol_to_axis_resolves():
    assert symbol_to_axis("PTEN") == "pi3k_pten"
    assert symbol_to_axis("MKI67") == "cell_cycle"
    assert symbol_to_axis("NOT_A_REAL_GENE_XYZ") == "unclassified"


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def test_response_cache_roundtrip(tmp_path):
    from qomics.data.cbioportal import ResponseCache

    cache = ResponseCache(root=tmp_path)
    assert cache.get("missing") is None
    cache.put("key", {"a": 1, "b": [1, 2, 3]})
    assert cache.get("key") == {"a": 1, "b": [1, 2, 3]}


def test_response_cache_survives_corrupt_entry(tmp_path):
    from qomics.data.cbioportal import ResponseCache

    cache = ResponseCache(root=tmp_path)
    cache.put("key", {"a": 1})
    path = cache._path("key")
    path.write_bytes(b"not gzip")
    assert cache.get("key") is None


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


def test_variance_filter_keeps_highest_variance_columns():
    from qomics.pipeline.preprocess import select_features

    frame = pd.DataFrame(
        {
            "flat": np.zeros(50),
            "noisy": np.random.default_rng(0).normal(size=50),
            "medium": np.arange(50, dtype=float) % 7,
        }
    )
    chosen = select_features(frame, 2)
    assert "noisy" in chosen.columns
    assert "flat" not in chosen.columns


def test_modality_reducer_imputes_and_preserves_rank():
    from qomics.pipeline.preprocess import ModalityReducer

    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(size=(60, 8)), columns=[f"f{i}" for i in range(8)])
    frame.iloc[0, 0] = np.nan
    reducer = ModalityReducer("test", n_components=3).fit(frame)
    out = reducer.transform(frame)
    assert out.shape == (60, 3)
    assert np.isfinite(out).all()


def test_fold_safe_preprocessing_keeps_train_and_test_aligned():
    """The fused vector must describe the requested rows, not the whole block.

    Regression test: an earlier version fused PCA components computed over
    the entire cohort while the token matrices came from a single fold, which
    silently misaligned the two model branches.
    """
    from qomics.pipeline.preprocess import MultiOmicsPreprocessor

    rng = np.random.default_rng(2)
    blocks = {
        "transcriptome": pd.DataFrame(rng.normal(size=(40, 12)),
                                      columns=[f"t{i}" for i in range(12)]),
        "genome": pd.DataFrame(rng.normal(size=(40, 9)),
                               columns=[f"g{i}" for i in range(9)]),
    }
    pre = MultiOmicsPreprocessor(
        blocks,
        max_features_per_block={"transcriptome": 8, "genome": 6},
        fused_components={"transcriptome": 3, "genome": 2},
    )
    train = np.arange(30)
    test = np.arange(30, 40)

    dense_tr, fused_tr = pre.fit_transform(train)
    dense_te, fused_te = pre.transform(test)

    assert fused_tr.shape[0] == len(train)
    assert fused_te.shape[0] == len(test)
    assert dense_tr["transcriptome"].shape[0] == len(train)
    assert dense_te["transcriptome"].shape[0] == len(test)
    assert fused_tr.shape[1] == 5
