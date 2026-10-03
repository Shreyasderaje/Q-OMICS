"""Print the measured results from a completed run."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qomics.config import ARTIFACT_DIR

d = json.loads((ARTIFACT_DIR / "run.json").read_text(encoding="utf-8"))
models = sorted(d["metrics"]["models"], key=lambda r: -(r.get("auc_mean") or 0))

f = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "  n/a"  # noqa: E731
print(f"{'model':<50}{'AUC':>8}{'+/-':>8}{'PR-AUC':>9}{'sens':>8}{'spec':>8}{'sec':>7}")
print("-" * 98)
for r in models:
    print(
        f"{r['name'][:49]:<50}{f(r.get('auc_mean')):>8}{f(r.get('auc_std')):>8}"
        f"{f(r.get('pr_auc_mean')):>9}{f(r.get('sensitivity_mean')):>8}"
        f"{f(r.get('specificity_mean')):>8}{f(r.get('seconds_mean')):>7}"
    )

by = {r["name"]: r for r in models}
hyb = by.get("Hybrid Transformer (Q-OMICS)", {})
print(f"\nHybrid 95% bootstrap CI: [{hyb.get('auc_ci_low')}, {hyb.get('auc_ci_high')}]"
      f"  pooled {f(hyb.get('auc_pooled'))}")

print("\nSignificance (paired t-test on 10 folds, Holm-Bonferroni corrected):")
for t in d["metrics"]["significance"]:
    print(f"  vs {t['comparison'].split(' vs ')[-1]:<46} "
          f"delta={t['mean_delta']:+.4f}  p={t['p_value']:.4f}  "
          f"{'SIGNIFICANT' if t.get('significant_at_0.05') else 'ns'}")

q = d["quantum"]
print("\nQuantum kernel:")
kd = q.get("kernel_diagnostics", {})
print("  diagnostics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in kd.items()})
print("  labelled separation:", {k: round(v, 5) for k, v in q.get("labelled_separation", {}).items()})
print("  geometric difference:", {k: round(v, 3) for k, v in q.get("geometric_difference", {}).items()
                                  if isinstance(v, float)})
print("  entanglement ablation:", {k: round(v, 5) if isinstance(v, float) else v
                                    for k, v in q.get("entanglement_ablation", {}).items()})
print("  encoding seconds:", q.get("encoding_seconds"))

print("\nModality contribution:", d["biomarker_summary"]["modality_contribution"])
print("Top axes:", [a["axis"] for a in d["biomarker_summary"]["top_axes"][:6]])

import csv

rows = list(csv.DictReader((ARTIFACT_DIR / "biomarkers.csv").open(encoding="utf-8")))
print(f"\nTop 15 biomarkers ({len(rows)} ranked):")
print(f"  {'#':>3} {'feature':<26} {'block':<14} {'axis':<20} {'imp':>7} {'stab':>6} {'d':>7} {'q':>7}")
for r in rows[:15]:
    print(f"  {r['rank']:>3} {r['feature'][:25]:<26} {r['block'][:13]:<14} "
          f"{r['axis'][:19]:<20} {float(r['importance']):>7.4f} {float(r['stability']):>6.2f} "
          f"{float(r['effect_size']):>+7.3f} {float(r['q_value']):>7.3f}")
sig = [r for r in rows if r["fdr_significant"] == "True"]
print(f"\nFDR-significant (q<0.05): {len(sig)} of {len(rows)}")
