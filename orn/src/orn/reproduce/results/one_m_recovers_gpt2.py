"""COU-01 — a single shared M reconstructs a large fraction of per-layer coupling.

Small-scale version: train a transformer, then solve the least-squares problem
    argmin_{M}  sum_l || M_l - M ||_F^2
(i.e. M is the mean of per-layer M_l), and report the fraction of Frobenius
energy recovered,  1 - ||M_l - M||_F^2 / ||M_l||_F^2, averaged over layers.

At real scale (GPT-2 Small, L=12) this lands near 95.8%. At our smoke scale
we expect somewhat less but the fraction should be large (>70%) if the
universality claim holds.
"""
from __future__ import annotations

import numpy as np
import torch

from orn.models.transformer import Transformer, TransformerConfig


def run(device: str | None = None, train_steps: int = 150, seed: int = 0) -> dict:
    torch.manual_seed(seed)

    cfg = TransformerConfig.from_dict(dict(
        d_model=64, n_layers=4, n_heads=4, vocab_size=128, seq_len=32,
    ))
    model = Transformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(train_steps):
        x = torch.randint(0, cfg.vocab_size, (8, cfg.seq_len))
        y = torch.roll(x, -1, dims=1)
        _, loss = model(x, targets=y)
        opt.zero_grad(); loss.backward(); opt.step()

    per_layer_M = []
    with np.errstate(all="ignore"):
        for blk in model.blocks:
            Wq = blk.Wq.weight.detach().cpu().double().numpy()
            Wk = blk.Wk.weight.detach().cpu().double().numpy()
            per_layer_M.append(Wq.T @ Wk)

    with np.errstate(all="ignore"):
        M_mean = np.mean(np.stack(per_layer_M, axis=0), axis=0)
        recovery = []
        for Ml in per_layer_M:
            num = np.linalg.norm(Ml - M_mean) ** 2
            den = np.linalg.norm(Ml) ** 2
            recovery.append(1 - num / (den + 1e-12))

    return {
        "n_layers":      len(per_layer_M),
        "mean_recovery": float(np.mean(recovery)),
        "min_recovery":  float(np.min(recovery)),
        "final_loss":    loss.item(),
        "_per_layer_recovery": recovery,
    }


def plot(result: dict, save_path=None, backend: str = "matplotlib"):
    """Per-layer bar chart of Frobenius recovery from a single shared M."""
    import matplotlib.pyplot as plt
    from orn.diagnostics.style import apply_style, PALETTE
    apply_style()

    rec = result["_per_layer_recovery"]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(rec))
    ax.bar(xs, [100 * r for r in rec], color=PALETTE[0], width=0.7)
    ax.axhline(100 * result["mean_recovery"], color="#333", linestyle="--",
               linewidth=1.2, label=f"mean {100*result['mean_recovery']:.1f}%")
    ax.set_xlabel("layer"); ax.set_ylabel("Frobenius recovery (%)")
    ax.set_title("A single shared M recovers most of per-layer W_Q^T W_K")
    ax.set_ylim(0, 100); ax.set_xticks(xs)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
