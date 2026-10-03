"""Pydantic request/response models for the Q-OMICS API.

Kept deliberately small: the wire format is the contract in
``docs/API_CONTRACT.md`` and every model here mirrors it exactly, so a change
to one is a change to both.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    trained_at: str | None = None
    cohort: str | None = None
    runtime_seconds: float = 0.0


class SamplePayload(BaseModel):
    """Sparse feature dictionary for one sample.

    Keys may be bare feature names (``PTEN``), block-qualified
    (``genome::MUT::PTEN``) or antibody names.  Unknown keys are ignored and
    missing ones are imputed, so a partial submission is valid.
    """

    transcriptome: dict[str, float] = Field(default_factory=dict)
    genome: dict[str, float] = Field(default_factory=dict)
    proteome: dict[str, float] = Field(default_factory=dict)
    flat: dict[str, float] = Field(
        default_factory=dict, description="Unqualified features merged across blocks"
    )

    def merged(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for block in (self.transcriptome, self.genome, self.proteome):
            out.update(block)
        out.update(self.flat)
        return out


class PredictRequest(BaseModel):
    sample: SamplePayload | None = None
    features: dict[str, float] = Field(
        default_factory=dict, description="Direct alternative to `sample`"
    )
    top_k: int = Field(default=10, ge=1, le=50)

    def resolved(self) -> dict[str, float]:
        if self.sample is not None:
            merged = self.sample.merged()
            if merged:
                return merged
        return self.features


class Contributor(BaseModel):
    name: str
    symbol: str
    block: str
    importance: float
    value: float
    supplied: bool = False
    direction: str | None = None
    contribution: float | None = None


class PredictResponse(BaseModel):
    prediction: Literal["early_stage", "advanced_stage"]
    label: str
    confidence: float
    probabilities: dict[str, float]
    risk_tier: Literal["low", "moderate", "elevated"]
    quantum_score: float | None = None
    matched_features: int = 0
    total_features: int = 0
    coverage: float = 0.0
    top_contributors: list[Contributor] = Field(default_factory=list)
    disclaimer: str = (
        "Research use only. Not a diagnostic device and not for clinical decision-making."
    )


class BiomarkerRecord(BaseModel):
    rank: int
    name: str
    symbol: str
    block: str
    axis: str
    importance: float
    importance_normalised: float
    stability: float
    effect_size: float
    p_value: float | None = None
    q_value: float | None = None
    direction: str
    mean_early: float | None = None
    mean_advanced: float | None = None
    n_observed: int = 0
    mutation_frequency_early: float | None = None
    mutation_frequency_advanced: float | None = None


class BiomarkerListResponse(BaseModel):
    total: int
    returned: int
    method: str
    biomarkers: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    detail: str


__all__ = [
    "HealthResponse",
    "SamplePayload",
    "PredictRequest",
    "PredictResponse",
    "Contributor",
    "BiomarkerRecord",
    "BiomarkerListResponse",
    "ErrorResponse",
]
