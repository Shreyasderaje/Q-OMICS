"""Central configuration for the Q-OMICS platform.

Every tunable in the system is declared here so that a run is fully
reproducible from a single dataclass tree.  Values are intentionally
conservative defaults that run to completion on a laptop CPU in minutes.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_DIR = Path(os.environ.get("QOMICS_DATA_DIR", PROJECT_ROOT / "data"))
ARTIFACT_DIR = Path(os.environ.get("QOMICS_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))
WEB_DIR = PACKAGE_ROOT / "web"


def ensure_dirs() -> None:
    """Create the writable directories used for caching and artifacts."""
    for d in (DATA_DIR, ARTIFACT_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Cohort definition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortConfig:
    """TCGA uterine corpus endometrial carcinoma (UCEC) cohort.

    Layers are pulled from two independent public archives and joined on the
    TCGA sample barcode (``TCGA-XX-XXXX-NN``):

    * cBioPortal  - genome (somatic mutations, GISTIC copy number) and
      transcriptome (RNA-seq TPM z-scores), plus FIGO stage and survival.
    * UCSC Xena   - proteome (RPPA antibody panel, standardised abundance).

    The primary predictive task is *early-stage detection*: FIGO stage I
    (confined to the uterus) versus stage II-IV (extrauterine spread).
    """

    study_id: str = "ucec_tcga_gdc"
    study_label: str = "TCGA-UCEC (GDC, 2025)"

    # ---- cBioPortal molecular profiles -----------------------------------
    profile_mrna: str = "ucec_tcga_gdc_mrna_seq_tpm_Zscores"
    profile_mut: str = "ucec_tcga_gdc_mutations"
    profile_cna: str = "ucec_tcga_gdc_cna"

    # ---- cBioPortal sample lists ------------------------------------------
    sl_mrna: str = "ucec_tcga_gdc_tpm"
    sl_mut: str = "ucec_tcga_gdc_sequenced"
    sl_cna: str = "ucec_tcga_gdc_cna"

    # ---- UCSC Xena resources ----------------------------------------------
    xena_rppa_url: str = (
        "https://tcga.xenahubs.net/download/TCGA.UCEC.sampleMap%2FRPPA.gz"
    )
    xena_clinical_url: str = (
        "https://tcga.xenahubs.net/download/TCGA.UCEC.sampleMap%2FUCEC_clinicalMatrix"
    )

    # ---- Task definition ---------------------------------------------------
    target_name: str = "early_stage_detection"
    target_description: str = "FIGO stage I vs stage II-IV endometrial carcinoma"
    early_stage_prefixes: tuple[str, ...] = ("stage i",)
    advanced_stage_prefixes: tuple[str, ...] = ("stage ii", "stage iii", "stage iv")


@dataclass(frozen=True)
class PreprocessConfig:
    """Modality-wise preprocessing and dimensionality reduction."""

    # Fraction of features retained by the per-modality variance filter.
    variance_top_frac: float = 0.60

    # PCA components per modality (multi-omics fusion block).
    n_pcs_transcriptome: int = 24
    n_pcs_genome: int = 12
    n_pcs_proteome: int = 12

    # Relative weight of each modality inside the fused classical block.
    modality_weights: dict[str, float] = field(
        default_factory=lambda: {"transcriptome": 1.0, "genome": 0.8, "proteome": 0.8}
    )

    # Fit PCA on training folds only to avoid label leakage.
    fit_within_fold: bool = True


@dataclass(frozen=True)
class QuantumConfig:
    """Quantum kernel settings.

    The kernel is a *fidelity* kernel ``k(x, x') = |<psi(x)|psi(x')>|^2``
    evaluated on a PennyLane state-vector simulator.  Because the simulator
    exposes the full wave-function we evaluate every embedding once and form
    the Gram matrix by a single matrix product, which is algebraically
    identical to running the compute-uncompute circuit ``n^2`` times but
    roughly two orders of magnitude cheaper.
    """

    enabled: bool = True
    n_qubits: int = 6
    feature_map: str = "iqp"  # {"iqp", "hea", "sep_rx"}
    n_layers: int = 2
    reupload: bool = True
    data_scale: float = 1.0
    add_ry_rotation: bool = True
    kernel_normalisation: str = "arc_cosine_scaled"  # {"none", "cosine", "arc_cosine_scaled"}
    random_state: int = 20240517

    # Feature-space reduction applied *before* encoding (qubit budget).
    n_input_features: int = 12
    c_regularisation: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    """Hybrid transformer encoder settings.

    Sized for the cohort, not for the architecture.  On 463 samples with a
    27 % event rate, a 433k-parameter encoder is not a stronger model, it is
    a larger one that will overfit and cost an hour per cross-validation
    sweep on CPU.  Measured epoch cost at d_model=128 / 480 tokens was
    45.8 s; the settings below trade a little capacity for a 6x faster,
    still-expressive fit.
    """

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.25
    activation: str = "gelu"
    n_classes: int = 2
    max_tokens: int = 256
    use_fusion_token: bool = True
    class_weighting: bool = True


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation and cross-validation schedule."""

    n_splits: int = 5
    n_repeats: int = 2
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1.5e-3
    weight_decay: float = 1.0e-4
    warmup_frac: float = 0.10
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    patience: int = 12
    monitor: str = "val_auc"
    seed: int = 20240517
    device: str = "auto"  # {"auto", "cpu", "cuda"}
    num_threads: int = 0  # 0 -> leave to torch defaults


@dataclass(frozen=True)
class BaselineConfig:
    """Classical reference models used for honest comparison."""

    random_state: int = 20240517
    rbf_svm_c: float = 1.0
    rbf_svm_gamma: str = "scale"
    logreg_c: float = 0.1
    rf_trees: int = 500
    hgb_learning_rate: float = 0.06
    hgb_max_iter: int = 300


@dataclass(frozen=True)
class AppConfig:
    """Serving configuration."""

    host: str = os.environ.get("QOMICS_HOST", "127.0.0.1")
    port: int = int(os.environ.get("QOMICS_PORT", "8000"))
    cors_origins: tuple[str, ...] = ("*",)
    max_upload_mb: int = 25


@dataclass(frozen=True)
class Config:
    cohort: CohortConfig = field(default_factory=CohortConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    app: AppConfig = field(default_factory=AppConfig)

    def to_dict(self) -> dict:
        return asdict(self)


CONFIG = Config()

__all__ = [
    "CONFIG",
    "Config",
    "CohortConfig",
    "PreprocessConfig",
    "QuantumConfig",
    "ModelConfig",
    "TrainConfig",
    "BaselineConfig",
    "AppConfig",
    "DATA_DIR",
    "ARTIFACT_DIR",
    "WEB_DIR",
    "PROJECT_ROOT",
    "ensure_dirs",
]
