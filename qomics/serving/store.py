"""Artifact store: loads the run output and serves it to the API layer.

The store is deliberately lazy and failure-tolerant.  A missing or partially
written artefact directory degrades individual endpoints rather than
preventing the server from starting, so the frontend can render what exists
and show a precise instruction for what does not.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import ARTIFACT_DIR
from ..pipeline.final_model import BIOMARKERS_NAME, COHORT_NAME, RUN_NAME

log = logging.getLogger(__name__)


class ArtifactStore:
    """In-memory view of everything ``scripts/train.py`` produced."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or ARTIFACT_DIR)
        self._lock = threading.Lock()
        self._loaded_at: float | None = None
        self.run: dict[str, Any] = {}
        self.biomarkers: pd.DataFrame = pd.DataFrame()
        self.cohort: pd.DataFrame = pd.DataFrame()
        self.bundle: Any = None
        self.load_error: str | None = None

    # -- loading -----------------------------------------------------------

    def load(self, with_bundle: bool = True) -> ArtifactStore:
        with self._lock:
            self._loaded_at = time.time()
            self.run = {}
            self.biomarkers = pd.DataFrame()
            self.cohort = pd.DataFrame()
            self.bundle = None
            self.load_error = None

            run_path = self.root / RUN_NAME
            if not run_path.exists():
                self.load_error = (
                    "no trained artefacts found; run `python scripts/train.py` first"
                )
                log.warning(self.load_error)
                return self

            try:
                self.run = json.loads(run_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self.load_error = f"could not read {run_path.name}: {exc}"
                return self

            markers = self.root / BIOMARKERS_NAME
            if markers.exists():
                try:
                    self.biomarkers = pd.read_csv(markers)
                except (OSError, pd.errors.ParserError) as exc:
                    log.warning("could not read biomarkers: %s", exc)

            cohort_path = self.root / COHORT_NAME
            if cohort_path.exists():
                try:
                    self.cohort = pd.read_csv(cohort_path, index_col=0)
                except (OSError, pd.errors.ParserError) as exc:
                    log.warning("could not read cohort: %s", exc)

            if with_bundle:
                self._load_bundle()

            log.info(
                "artefacts loaded: %d models, %d biomarkers, %d cohort rows%s",
                len(self.run.get("metrics", {}).get("models", [])),
                len(self.biomarkers),
                len(self.cohort),
                ", bundle ok" if self.bundle is not None else ", no bundle",
            )
        return self

    def _load_bundle(self) -> None:
        """Rebuild the deployable bundle from the persisted checkpoint."""
        import torch
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        from ..config import CONFIG
        from ..models.transformer import MultiOmicsTransformer
        from ..pipeline.final_model import ModelBundle
        from ..pipeline.preprocess import ModalityReducer, MultiOmicsPreprocessor
        from ..quantum.kernels import FidelityQuantumKernel

        path = self.root / "model_bundle.pt"
        if not path.exists():
            log.warning("model_bundle.pt absent; /api/predict will return 503")
            return

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover
            log.warning("could not load bundle: %s", exc)
            return

        selected: dict[str, list[str]] = {
            block: list(columns) for block, columns in payload["selected"].items()
        }
        pre = MultiOmicsPreprocessor(
            {name: pd.DataFrame(columns=cols) for name, cols in selected.items()},
            max_features_per_block={name: len(cols) for name, cols in selected.items()},
            fused_components={name: len(spec["components"]) for name, spec in payload["reducers"].items()},
        )
        pre.selected = selected
        pre.block_names = list(selected)
        for name, spec in payload["reducers"].items():
            reducer = ModalityReducer(name=name, n_components=len(spec["components"]), weight=spec["weight"])
            reducer.scaler = StandardScaler()
            reducer.scaler.mean_ = np.asarray(spec["mean"], dtype=float)
            reducer.scaler.scale_ = np.asarray(spec["scale"], dtype=float)
            # ``n_features_in_`` is the width of the *input*, not the number
            # of components.  Setting it to len(components) makes scikit-learn
            # reject a correctly-shaped row on the very first transform.
            reducer.scaler.n_features_in_ = len(spec["mean"])
            reducer.pca = PCA(n_components=len(spec["components"]))
            reducer.pca.components_ = np.asarray(spec["components"], dtype=float)
            reducer.pca.n_features_in_ = len(spec["mean"])
            pca_mean = spec.get("pca_mean")
            reducer.pca.mean_ = (
                np.asarray(pca_mean, dtype=float) if pca_mean is not None else np.zeros(len(spec["mean"]))
            )
            reducer.pca.n_samples_ = int(
                spec.get("n_samples") or payload.get("metadata", {}).get("n_samples", 1) or 1
            )
            explained = spec.get("explained_variance")
            if explained is not None:
                reducer.pca.explained_variance_ = np.asarray(explained, dtype=float)
            pre.reducers[name] = reducer

        qc = payload.get("quantum") or {
            "n_qubits": CONFIG.quantum.n_qubits,
            "feature_map": CONFIG.quantum.feature_map,
            "n_layers": CONFIG.quantum.n_layers,
            "reupload": CONFIG.quantum.reupload,
            "data_scale": CONFIG.quantum.data_scale,
            "add_ry_rotation": CONFIG.quantum.add_ry_rotation,
            "normalisation": CONFIG.quantum.kernel_normalisation,
        }
        kernel = FidelityQuantumKernel(
            n_qubits=int(qc.get("n_qubits", 6)),
            feature_map=qc.get("feature_map", "iqp"),
            n_layers=int(qc.get("n_layers", 2)),
            reupload=bool(qc.get("reupload", True)),
            data_scale=float(qc.get("data_scale", 1.0)),
            add_ry_rotation=bool(qc.get("add_ry_rotation", True)),
            normalisation=qc.get("normalisation", "arc_cosine_scaled"),
        )
        # The test-time kernel row is normalised by the *training* state
        # norms; without them the row lands in a different scale from every
        # row the SVM was fitted on.
        reference = payload.get("kernel_reference_norm")
        if reference is not None:
            kernel._reference_norm = np.asarray(reference, dtype=float)

        # Rebuild the encoder from the checkpoint's own hyperparameters.
        # Using the current CONFIG instead would fail on any default that
        # changed between the training run and the server start.
        model_cfg = payload.get("model_config", {})
        block_sizes = model_cfg.get("block_sizes") or {k: len(v) for k, v in selected.items()}
        n_layers = int(model_cfg.get("n_layers", CONFIG.model.n_layers))
        model = MultiOmicsTransformer(
            block_sizes=block_sizes,
            d_model=int(model_cfg.get("d_model", CONFIG.model.d_model)),
            n_heads=int(model_cfg.get("n_heads", CONFIG.model.n_heads)),
            n_layers=n_layers,
            dim_feedforward=int(
                model_cfg.get("dim_feedforward") or CONFIG.model.dim_feedforward
            ),
            dropout=CONFIG.model.dropout,
            n_classes=int(model_cfg.get("n_classes", CONFIG.model.n_classes)),
            n_fusion=int(model_cfg.get("n_fusion", 1)),
            max_tokens=sum(block_sizes.values()) + 2,
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()

        bundle = ModelBundle(
            preprocessor=pre, transformer=model, kernel=kernel,
            svm=payload.get("svm"),
            feature_meta=pd.DataFrame(),
            training_medians=payload.get("training_medians", {}),
            training_means=payload.get("training_means", {}),
            training_stds=payload.get("training_stds", {}),
            feature_order=payload.get("feature_order", []),
            block_order=payload.get("block_order", list(selected)),
            block_slices={k: tuple(v) for k, v in payload.get("block_slices", {}).items()},
            metadata=payload.get("metadata", {}),
        )
        fused_reference = payload.get("fused_reference")
        if fused_reference is not None:
            fused = np.asarray(fused_reference, dtype=float)
            bundle._fused_reference = fused
        else:
            # Without the stored reference matrix the test row can only be
            # built against an empty training set, so scoring is impossible.
            bundle._fused_reference = np.zeros((0, 1), dtype=float)

        if bundle.svm is None:
            log.warning("bundle has no fitted SVM; /api/predict will return 503")
        self.bundle = bundle
        log.info(
            "bundle ready: %d blocks, %d tokens, svm=%s, reference=%s",
            len(block_sizes), sum(block_sizes.values()),
            bundle.svm is not None,
            "ok" if fused_reference is not None else "missing",
        )

    # -- accessors ---------------------------------------------------------

    @property
    def ready(self) -> bool:
        return bool(self.run)

    @property
    def uptime(self) -> float:
        return time.time() - self._loaded_at if self._loaded_at else 0.0

    def require(self) -> None:
        if not self.run:
            raise FileNotFoundError(self.load_error or "model not trained")

    def biomarker_lookup(self, symbol: str) -> dict | None:
        if self.biomarkers.empty:
            return None
        frame = self.biomarkers
        if "symbol" in frame.columns:
            hit = frame[frame["symbol"] == symbol]
            if not hit.empty:
                return hit.iloc[0].to_dict()
        if symbol in frame.index:
            return frame.loc[symbol].to_dict()
        direct = frame[frame["feature"] == symbol]
        return direct.iloc[0].to_dict() if not direct.empty else None

    def cohort_predictions(self) -> dict[str, dict]:
        """In-sample predictions for the cohort explorer.

        Cached to disk at training time by ``scripts/train.py``; when absent
        the explorer still lists samples without a prediction column.
        """
        path = self.root / "cohort_predictions.csv"
        if not path.exists():
            return {}
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            return {}
        return {
            str(row["sample_id"]): {
                "prediction": row.get("prediction"),
                "confidence": _safe_float(row.get("confidence")),
            }
            for _, row in frame.iterrows()
        }


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
        return out if np.isfinite(out) else None
    except (TypeError, ValueError):
        return None


_STORE: ArtifactStore | None = None


def get_store() -> ArtifactStore:
    """Process-wide singleton store, loaded on first use."""
    global _STORE
    if _STORE is None:
        _STORE = ArtifactStore().load()
    return _STORE


def reload_store() -> ArtifactStore:
    global _STORE
    _STORE = ArtifactStore().load()
    return _STORE


__all__ = ["ArtifactStore", "get_store", "reload_store"]
