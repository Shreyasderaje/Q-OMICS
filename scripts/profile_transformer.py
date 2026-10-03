"""Profile transformer training cost across configurations."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qomics.models.transformer import MultiOmicsTransformer, focal_loss, parameter_count

print(f"threads={torch.get_num_threads()}\n")
n_train, n_val, n_classes = 320, 50, 2
y = np.array([0] * (n_train - 90) + [1] * 90)
y_t = torch.tensor(y)

print(f"{'tokens':>7} {'d':>4} {'L':>2} {'H':>2} {'bs':>4} | {'s/epoch':>8} {'params':>9}")
print("-" * 62)

for tokens, d, layers, heads, bs in [
    (480, 128, 3, 4, 32),
    (480, 128, 3, 4, 64),
    (320, 96,  2, 4, 64),
    (256, 64,  2, 4, 64),
    (192, 64,  2, 4, 64),
    (160, 64,  2, 4, 128),
    (128, 64,  2, 4, 64),
]:
    split = {"transcriptome": tokens // 2, "genome": tokens - tokens // 2 - tokens // 6, "proteome": tokens // 6}
    model = MultiOmicsTransformer(split, d_model=d, n_heads=heads, n_layers=layers,
                                 dim_feedforward=2 * d, dropout=0.3, n_fusion=1, max_tokens=1024)
    X = {k: torch.randn(n_train, n) for k, n in split.items()}
    fusion = torch.randn(n_train, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # warmup
    idx = torch.arange(bs)
    out = model({k: v[idx] for k, v in X.items()}, fusion[idx])
    loss = focal_loss(out["logits"], y_t[idx]); opt.zero_grad(); loss.backward(); opt.step()

    t0 = time.perf_counter()
    steps = 0
    for start in range(0, n_train, bs):
        idx = torch.arange(start, min(start + bs, n_train))
        if len(idx) < 2:
            continue
        out = model({k: v[idx] for k, v in X.items()}, fusion[idx])
        loss = focal_loss(out["logits"], y_t[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        steps += 1
    per_epoch = (time.perf_counter() - t0) / max(1, steps) * steps
    print(f"{tokens:>7} {d:>4} {layers:>2} {heads:>2} {bs:>4} | "
          f"{per_epoch:>8.2f} {parameter_count(model):>9,}")

print("\nNote: 's/epoch' is train-only; add ~15% for the per-epoch validation pass.")
print("30 epochs estimate = s/epoch * 30 * 2 (hybrid + classical) * n_folds")
