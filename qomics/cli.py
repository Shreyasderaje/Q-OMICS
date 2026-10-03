"""Command-line interface.

    python -m qomics.cli fetch     # download and assemble the cohort
    python -m qomics.cli train     # run cross-validation and fit the model
    python -m qomics.cli all       # fetch + train
    python -m qomics.cli serve     # start the API and web interface
    python -m qomics.cli report    # print the ranked biomarker panel
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow `python scripts/train.py` as well as `python -m qomics.cli`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qomics.config import ARTIFACT_DIR, CONFIG, ensure_dirs  # noqa: E402

log = logging.getLogger("qomics.cli")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "pennylane", "matplotlib", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _build_cohort():
    from qomics.data.cohort import build_cohort, save_manifest

    started = time.perf_counter()
    cohort = build_cohort()
    manifest = save_manifest(cohort)

    log.info("cohort ready in %.1fs -> %s", time.perf_counter() - started, manifest)
    summary = cohort.summary()
    log.info(
        "  %d samples / %d patients | %s | class balance %.3f",
        summary["n_samples"], summary["n_patients"],
        summary["block_shapes"], summary["class_balance"],
    )
    return cohort


def cmd_fetch(args: argparse.Namespace) -> int:
    _build_cohort()
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from qomics.pipeline.experiment import _make_preprocessor, run_experiment
    from qomics.pipeline.final_model import BUNDLE_NAME, train_final_model

    ensure_dirs()
    cohort = _build_cohort()
    started = time.perf_counter()

    payload = run_experiment(
        cohort,
        CONFIG,
        artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        n_splits=args.splits,
        n_repeats=args.repeats,
        epochs=args.epochs,
        skip_transformer=args.skip_transformer,
    )

    metrics = payload["metrics"]["models"]
    print("\n" + "=" * 78)
    print("CROSS-VALIDATED RESULTS  ({})".format(payload["metrics"]["target"]))
    print("=" * 78)
    print(f"{'model':<48} {'AUC':>8} {'+/-':>7} {'PR-AUC':>8} {'sens':>7} {'spec':>7}")
    print("-" * 78)
    for row in sorted(metrics, key=lambda r: -(r.get("auc_mean") or 0)):
        print(
            f"{str(row.get('name'))[:47]:<48} "
            f"{_fmt(row.get('auc_mean')):>8} {_fmt(row.get('auc_std')):>7} "
            f"{_fmt(row.get('pr_auc_mean')):>8} {_fmt(row.get('sensitivity_mean')):>7} "
            f"{_fmt(row.get('specificity_mean')):>7}"
        )

    significance = payload["metrics"].get("significance", [])
    if significance:
        print("\nPaired comparison vs the hybrid model (Holm-Bonferroni corrected):")
        for test in significance:
            flag = "significant" if test.get("significant_at_0.05") else "not significant"
            print(
                f"  vs {test['comparison'].split(' vs ')[-1]:<42} "
                f"delta={_fmt(test['mean_delta']):>8}  p={_fmt(test['p_value']):>7}  {flag}"
            )

    geometry = payload["quantum"].get("geometric_difference", {})
    if geometry.get("interpretation"):
        print("\nQuantum geometry:")
        print(f"  g vs RBF    = {_fmt(geometry.get('g_quantum_vs_rbf'))}")
        print(f"  g vs cosine = {_fmt(geometry.get('g_quantum_vs_cosine'))}")
        print(f"  {geometry['interpretation']}")

    if not args.skip_transformer:
        print("\nFitting deployable model on the full cohort...")
        preprocessor = _make_preprocessor(cohort.blocks, CONFIG)
        preprocessor.fit(None)
        bundle = train_final_model(preprocessor, cohort.blocks, cohort.y, CONFIG, epochs=args.epochs)
        bundle.save(Path(ARTIFACT_DIR) / BUNDLE_NAME)
        cohort_predictions = _score_cohort(bundle, cohort)
        cohort_predictions.to_csv(Path(ARTIFACT_DIR) / "cohort_predictions.csv", index=False)
        log.info("bundle saved -> %s", Path(ARTIFACT_DIR) / BUNDLE_NAME)

    panel_path = Path(ARTIFACT_DIR) / "biomarkers.txt"
    if panel_path.exists():
        print("\n" + "=" * 78)
        print(panel_path.read_text(encoding="utf-8"))

    print(f"\nTotal wall clock: {time.perf_counter() - started:.1f}s")
    print(f"Artefacts:       {ARTIFACT_DIR}")
    return 0


def _score_cohort(bundle, cohort) -> object:
    import pandas as pd


    rows = []
    dense, _ = bundle.preprocessor.transform(None)
    for i, sample_id in enumerate(cohort.sample_ids):
        payload = {b: {c: float(dense[b][i, j]) for j, c in enumerate(bundle.preprocessor.selected[b])}
                   for b in bundle.block_order}
        try:
            out = bundle.score(_flatten(payload))
        except Exception as exc:  # pragma: no cover
            log.warning("in-sample scoring failed for %s: %s", sample_id, exc)
            out = {"prediction": None, "confidence": float("nan")}
        rows.append({"sample_id": sample_id, "prediction": out.get("prediction"),
                     "confidence": out.get("confidence")})
    return pd.DataFrame(rows)


def _flatten(nested: dict) -> dict:
    out: dict = {}
    for block, values in nested.items():
        for key, value in values.items():
            out[key] = value
    return out


def _fmt(value, digits: int = 4) -> str:
    try:
        f = float(value)
        return f"{f:.{digits}f}" if f == f else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from qomics.serving.app import app

    host = args.host or CONFIG.app.host
    port = args.port or CONFIG.app.port
    print(f"\n  Q-OMICS serving on http://{host}:{port}")
    print(f"  API docs:          http://{host}:{port}/docs")
    print(f"  Artefacts:         {ARTIFACT_DIR}\n")
    uvicorn.run(app, host=host, port=port, reload=args.reload, log_level="info")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.artifact_dir or ARTIFACT_DIR) / "biomarkers.txt"
    if not path.exists():
        print("no biomarker panel found; run `python -m qomics.cli train` first", file=sys.stderr)
        return 1
    print(path.read_text(encoding="utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qomics", description="Q-OMICS: hybrid quantum-classical biomarker discovery"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download and assemble the cohort").set_defaults(func=cmd_fetch)

    train = sub.add_parser("train", help="run cross-validation and fit the model")
    train.add_argument("--splits", type=int, default=None, help="folds per repeat")
    train.add_argument("--repeats", type=int, default=None, help="repeat count")
    train.add_argument("--epochs", type=int, default=None, help="transformer epochs")
    train.add_argument("--artifact-dir", type=str, default=None)
    train.add_argument("--skip-transformer", action="store_true",
                       help="baselines and quantum branch only (fast smoke run)")
    train.set_defaults(func=cmd_train)

    serve = sub.add_parser("serve", help="start the API and web interface")
    serve.add_argument("--host", type=str, default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true", help="uvicorn auto-reload")
    serve.set_defaults(func=cmd_serve)

    report = sub.add_parser("report", help="print the ranked biomarker panel")
    report.add_argument("--artifact-dir", type=str, default=None)
    report.set_defaults(func=cmd_report)

    both = sub.add_parser("all", help="fetch then train")
    both.add_argument("--splits", type=int, default=None)
    both.add_argument("--repeats", type=int, default=None)
    both.add_argument("--epochs", type=int, default=None)
    both.add_argument("--artifact-dir", type=str, default=None)
    both.add_argument("--skip-transformer", action="store_true")
    both.set_defaults(func=cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
