"""FastAPI application exposing the trained Q-OMICS artefacts.

Design notes
------------
* The server is **read-only with respect to the model**.  It never trains,
  refits or mutates artefacts; ``POST /api/reload`` re-reads disk, which is
  enough to pick up a new run.
* Every endpoint degrades independently.  A missing biomarker table blanks
  the biomarker panel but leaves metrics, quantum provenance and the cohort
  explorer fully functional, so a partial run still yields a usable product.
* Numeric output is sanitised before serialisation.  NaN is not valid JSON,
  and a single NaN inside ``metrics.models`` would make the entire response
  fail to parse in the browser - so it is converted to ``null`` at the
  boundary rather than allowed to escape.
"""

from __future__ import annotations

import io
import logging
import math
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config import CONFIG, WEB_DIR
from .schemas import (
    Contributor,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from .store import get_store, reload_store

log = logging.getLogger(__name__)

_STARTED_AT = time.time()

DISCLAIMER = (
    "Research use only. Not a diagnostic device and not for clinical decision-making."
)


# --------------------------------------------------------------------------
# JSON hygiene
# --------------------------------------------------------------------------


def clean(value: Any) -> Any:
    """Recursively make a payload JSON-safe.

    ``NaN`` and infinities become ``None``.  Without this, ``json.dumps``
    emits bare ``NaN`` tokens that ``JSON.parse`` rejects, so a single
    degenerate fold would take down the whole page.
    """
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _num(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    store = get_store()
    log.info(
        "Q-OMICS API ready | artefacts=%s | %d models | %d biomarkers",
        "loaded" if store.ready else "absent",
        len(store.run.get("metrics", {}).get("models", [])),
        len(store.biomarkers),
    )
    yield


app = FastAPI(
    title="Q-OMICS",
    version="1.0.0",
    description=(
        "Hybrid quantum-classical foundation model for multi-omics biomarker "
        "discovery. Research use only."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CONFIG.app.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _store_or_503():
    store = get_store()
    if not store.ready:
        raise HTTPException(
            status_code=503,
            detail=store.load_error or "model not trained; run `python scripts/train.py`",
        )
    return store


# --------------------------------------------------------------------------
# Health and overview
# --------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    store = get_store()
    return HealthResponse(
        status="ok" if store.ready else "degraded",
        version="1.0.0",
        model_loaded=store.ready,
        trained_at=store.run.get("generated_at"),
        cohort=store.run.get("cohort", {}).get("study"),
        runtime_seconds=round(time.time() - _STARTED_AT, 1),
    )


@app.get("/api/overview", tags=["report"])
def overview() -> dict:
    store = _store_or_503()
    run = store.run
    metrics = run.get("metrics", {})
    models = metrics.get("models", [])
    by_name = {m.get("name"): m for m in models}

    hybrid = by_name.get("Hybrid Transformer (Q-OMICS)", {})
    classical = [
        m for m in models
        if m.get("family") in {"linear", "kernel", "ensemble", "neural"}
        and m.get("family") != "quantum"
        and m.get("name") != "Classical Transformer"
    ]
    best_classical = max(classical, key=lambda m: _num(m.get("auc_mean")), default={})

    return clean(
        {
            "product": {
                "name": "Q-OMICS",
                "tagline": "Hybrid quantum-classical foundation model for multi-omics biomarker discovery",
                "version": run.get("version", "1.0.0"),
            },
            "cohort": run.get("cohort", {}),
            "headline_metrics": {
                "hybrid_auc_mean": _num(hybrid.get("auc_mean")),
                "hybrid_auc_std": _num(hybrid.get("auc_std")),
                "hybrid_auc_ci_low": hybrid.get("auc_ci_low"),
                "hybrid_auc_ci_high": hybrid.get("auc_ci_high"),
                "best_baseline_auc_mean": _num(best_classical.get("auc_mean")),
                "best_baseline_name": best_classical.get("name"),
                "improvement_over_best_classical": _num(hybrid.get("auc_mean"))
                - _num(best_classical.get("auc_mean")),
                "n_splits": metrics.get("n_splits", 0),
            },
            "modality_contribution": run.get("biomarker_summary", {}).get(
                "modality_contribution", []
            ),
            "runtime_seconds": run.get("runtime_seconds"),
            "generated_at": run.get("generated_at"),
        }
    )


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@app.get("/api/metrics", tags=["report"])
def metrics() -> dict:
    store = _store_or_503()
    return clean(store.run.get("metrics", {}))


@app.get("/api/architecture", tags=["report"])
def architecture() -> dict:
    store = _store_or_503()
    payload = dict(store.run.get("architecture", {}))
    payload.setdefault("model", {})
    payload.setdefault("training", {})
    payload.setdefault("stages", [])
    return clean(payload)


@app.get("/api/quantum", tags=["report"])
def quantum() -> dict:
    store = _store_or_503()
    payload = dict(store.run.get("quantum", {}))

    # Recompute a live state-vector example so the panel shows a real
    # measurement from the configured circuit rather than a stored blob.
    try:
        from ..quantum.features import QuantumFeatureMap

        qc = store.run.get("config", {}).get("quantum", {})
        feature_map = QuantumFeatureMap(
            n_qubits=int(qc.get("n_qubits", 6)),
            feature_map=qc.get("feature_map", "iqp"),
            n_layers=int(qc.get("n_layers", 2)),
            reupload=bool(qc.get("reupload", True)),
            data_scale=float(qc.get("data_scale", 1.0)),
            add_ry_rotation=bool(qc.get("add_ry_rotation", True)),
        )
        rng = np.random.default_rng(CONFIG.quantum.random_state)
        examples = rng.normal(size=(2, feature_map.n_qubits))
        states = feature_map.embed_many(examples)
        labels = ["seeded state A", "seeded state B"]
        payload["state_vector_visualisation"] = {
            "n_examples": 2,
            "labels": labels,
            "dim": int(states.shape[1]),
            "amplitudes": [
                [float(v.real) for v in np.abs(row)[:16]] for row in states
            ],
        }
        # Fidelity between the two example states, for context.
        overlap = float(abs(np.vdot(states[0], states[1])) ** 2)
        payload["state_vector_visualisation"]["pairwise_fidelity"] = overlap
    except Exception as exc:  # pragma: no cover - never break the endpoint
        log.warning("state-vector preview unavailable: %s", exc)

    return clean(payload)


# --------------------------------------------------------------------------
# Biomarkers
# --------------------------------------------------------------------------


@app.get("/api/biomarkers", tags=["discovery"])
def biomarkers(
    limit: int = Query(default=25, ge=1, le=500),
    block: str = Query(default="all"),
) -> dict:
    store = _store_or_503()
    frame = store.biomarkers
    if frame.empty:
        return {"total": 0, "returned": 0, "method": "attention_rollout", "biomarkers": []}

    if block != "all":
        frame = frame[frame["block"] == block]
    frame = frame.sort_values("rank") if "rank" in frame.columns else frame

    total = int(len(store.biomarkers))
    rows = frame.head(limit).to_dict(orient="records")
    return clean(
        {
            "total": total,
            "returned": len(rows),
            "method": store.run.get("biomarker_summary", {}).get(
                "method", "attention_rollout"
            ),
            "biomarkers": [_decorate_biomarker(r) for r in rows],
        }
    )


def _decorate_biomarker(row: dict) -> dict:
    out = {
        "rank": int(row.get("rank", 0)),
        "name": row.get("feature"),
        "symbol": row.get("symbol"),
        "block": row.get("block"),
        "axis": row.get("axis"),
        "importance": _num(row.get("importance")),
        "importance_normalised": _num(row.get("importance_normalised")),
        "stability": _num(row.get("stability")),
        "effect_size": _num(row.get("effect_size")),
        "p_value": row.get("p_value"),
        "q_value": row.get("q_value"),
        "fdr_significant": bool(row.get("fdr_significant", False)),
        "direction": row.get("direction"),
        "mean_early": row.get("mean_early"),
        "mean_advanced": row.get("mean_advanced"),
        "n_observed": int(_num(row.get("n_observed"))),
    }
    out["modality"] = out["block"]
    return out


@app.get("/api/biomarkers/{symbol}", tags=["discovery"])
def biomarker_detail(symbol: str) -> dict:
    store = _store_or_503()
    hit = store.biomarker_lookup(symbol)
    if hit is None:
        # A symbol outside the ranked panel is a legitimate query result, not
        # an error: the frontend wants to render "not in panel" state.
        return {"symbol": symbol, "present": False}
    payload = _decorate_biomarker(hit)
    payload["present"] = True
    payload["train_frequency_early"] = _num(hit.get("mean_early"), float("nan"))
    payload["train_frequency_advanced"] = _num(hit.get("mean_advanced"), float("nan"))
    return clean(payload)


@app.get("/api/features", tags=["discovery"])
def features() -> dict:
    store = _store_or_503()
    blocks: list[dict] = []
    n_total = 0
    if not store.biomarkers.empty:
        for block, group in store.biomarkers.groupby("block"):
            counts = group["axis"].value_counts().to_dict() if "axis" in group else {}
            blocks.append(
                {
                    "block": str(block),
                    "n_features": int(len(group)),
                    "axes": [
                        {"axis": str(axis), "count": int(count)}
                        for axis, count in counts.items()
                    ],
                }
            )
            n_total += int(len(group))
    blocks.sort(key=lambda b: b["block"])
    return {"blocks": blocks, "n_total": n_total}


# --------------------------------------------------------------------------
# Cohort explorer
# --------------------------------------------------------------------------


def _cohort_frame() -> pd.DataFrame:
    store = get_store()
    frame = store.cohort.copy()
    if frame.empty:
        return frame
    # The sample barcode is the index.  A CSV round-trip can drop its name,
    # in which case reset_index() would produce "Unnamed: 0" and every
    # downstream consumer of `sample_id` would fail.
    if frame.index.name is None:
        frame.index.name = "sample_id"
    predictions = store.cohort_predictions()
    if predictions:
        frame["prediction"] = [predictions.get(str(s), {}).get("prediction") for s in frame.index]
        frame["confidence"] = [predictions.get(str(s), {}).get("confidence") for s in frame.index]
        frame["in_sample"] = frame["prediction"].notna()
    return frame.reset_index()


@app.get("/api/cohort", tags=["cohort"])
def cohort(
    limit: int = Query(default=60, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    frame = _cohort_frame()
    if frame.empty:
        return {"columns": [], "rows": [], "total": 0,
                "stage_distribution": [], "histology_distribution": []}

    stage_col = "figo_stage" if "figo_stage" in frame.columns else None
    hist_col = "histological_type" if "histological_type" in frame.columns else None

    # ``value_counts().to_dict()`` maps value -> count, so iterating yields
    # (value, count) pairs rather than records.
    stage_counts = (
        frame[stage_col].fillna("unknown").astype(str).value_counts()
        if stage_col
        else {}
    )
    hist_counts = (
        frame[hist_col].fillna("unknown").astype(str).value_counts()
        if hist_col
        else {}
    )
    stage_dist = [{"stage": str(k), "count": int(v)} for k, v in stage_counts.items()]
    hist_dist = [{"name": str(k), "count": int(v)} for k, v in hist_counts.items()]

    page = frame.iloc[offset : offset + limit]
    columns = list(page.columns)
    return clean(
        {
            "columns": columns,
            "rows": page.to_dict(orient="records"),
            "stage_distribution": stage_dist,
            "histology_distribution": hist_dist,
            "total": int(len(frame)),
            "offset": offset,
            "limit": limit,
        }
    )


@app.get("/api/cohort/download", tags=["cohort"])
def cohort_download(request: Request, format: str = Query(default="csv")) -> Response:
    """Export the cohort table.

    CSV is the default because that is what a spreadsheet user wants.  JSON
    is opt-in via ``?format=json`` - inferring it from the ``Accept`` header
    is unreliable, since many HTTP clients send ``*/*`` or advertise
    ``application/json`` while still expecting a download.
    """
    frame = _cohort_frame()
    if frame.empty:
        raise HTTPException(status_code=503, detail="cohort table unavailable")

    preferred = [
        "sample_id", "patient_id", "figo_stage", "figo_label", "histological_type",
        "age_at_diagnosis", "sample_type", "prediction", "confidence", "in_sample",
    ]
    columns = [c for c in preferred if c in frame.columns] + [
        c for c in frame.columns if c not in preferred
    ]
    table = frame[columns]

    if format.lower() == "json":
        accept = request.headers.get("accept", "")
        records = clean(table.to_dict(orient="records"))
        if "text/csv" in accept:
            buffer = io.StringIO()
            table.to_csv(buffer, index=False)
            return PlainTextResponse(
                buffer.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": 'attachment; filename="qomics_cohort.csv"'},
            )
        return JSONResponse(content=records)

    buffer = io.StringIO()
    table.to_csv(buffer, index=False)
    return PlainTextResponse(
        buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="qomics_cohort.csv"'},
    )


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------


@app.post("/api/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest) -> PredictResponse:
    store = _store_or_503()
    if store.bundle is None:
        raise HTTPException(
            status_code=503,
            detail="model checkpoint absent; re-run `python scripts/train.py` to build it",
        )

    features = request.resolved()
    if not features:
        raise HTTPException(status_code=422, detail="no features supplied")

    bundle = store.bundle
    if getattr(bundle, "svm", None) is None:
        raise HTTPException(status_code=503, detail="serving bundle incomplete")

    try:
        result = bundle.score(features)
    except Exception as exc:  # pragma: no cover
        log.exception("scoring failed")
        raise HTTPException(status_code=500, detail=f"scoring failed: {exc}") from exc

    top = result["top_contributors"][: request.top_k]
    direction_lookup = {
        row.get("symbol"): row.get("direction")
        for row in (
            store.biomarkers.to_dict(orient="records")
            if not store.biomarkers.empty
            else []
        )
    }

    contributors = [
        Contributor(
            name=item["name"],
            symbol=item["symbol"],
            block=item["block"],
            importance=item["importance"],
            value=item["value"],
            supplied=item.get("supplied", False),
            direction=direction_lookup.get(item["symbol"]),
            contribution=item["importance"] * item["value"],
        )
        for item in top
    ]

    total_features = len(bundle.feature_order) or 1
    # Coverage is how much of the *vocabulary* the caller actually supplied,
    # not how many of the top contributors happened to be among them.  Counting
    # the latter reports 0% for a request that named real features, purely
    # because the ranked panel is dominated by features the caller omitted.
    vocabulary = set(bundle.feature_order)
    matched = sum(
        1 for key in features
        if key in vocabulary or key.split("::")[-1] in {f.split("::")[-1] for f in vocabulary}
    )

    return PredictResponse(
        prediction=result["prediction"],
        label=result["label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        risk_tier=result["risk_tier"],
        quantum_score=result.get("quantum_score"),
        matched_features=matched,
        total_features=total_features,
        coverage=matched / total_features,
        top_contributors=contributors,
        disclaimer=DISCLAIMER,
    )


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------


@app.post("/api/reload", tags=["system"])
def reload() -> dict:
    store = reload_store()
    return {
        "reloaded": True,
        "ready": store.ready,
        "models": len(store.run.get("metrics", {}).get("models", [])),
        "biomarkers": len(store.biomarkers),
        "cohort_rows": len(store.cohort),
        "bundle_loaded": store.bundle is not None,
    }


@app.get("/api/config", tags=["system"])
def configuration() -> dict:
    store = get_store()
    return clean(store.run.get("config", CONFIG.to_dict()))


# --------------------------------------------------------------------------
# Static frontend
# --------------------------------------------------------------------------

if (WEB_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> Response:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            "<html><body style='font-family:system-ui;padding:48px'>"
            "<h1>Q-OMICS API</h1>"
            "<p>The frontend bundle is not present. See <code>/docs</code> for the API.</p>"
            "</body></html>",
            status_code=200,
        )
    return FileResponse(str(index_path), media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    path = WEB_DIR / "assets" / "favicon.svg"
    if path.exists():
        return FileResponse(str(path), media_type="image/svg+xml")
    return Response(status_code=204)


@app.exception_handler(404)
async def not_found(request: Request, exc: Exception) -> JSONResponse:
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": f"no such endpoint: {request.url.path}"}, status_code=404)
    return JSONResponse({"detail": "not found"}, status_code=404)


__all__ = ["app", "clean"]
