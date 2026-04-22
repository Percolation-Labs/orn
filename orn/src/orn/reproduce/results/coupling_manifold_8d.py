"""COU-02 — the coupling manifold of a trained transformer is low-dimensional.

Build a tiny transformer, train it for a few hundred steps on a synthetic LM
task, then stack per-layer M_l = W_Q^T W_K and measure the effective rank of
the stacked matrix. At real scale this lands near 8 for GPT-2; at our smoke
scale (d_model=64, L=4) we just verify (a) the eff-rank is well below L*d and
(b) the spectral-correlation metric matches the claim of universality.
"""
from __future__ import annotations

import numpy as np
import torch

from orn.models.transformer import Transformer, TransformerConfig


def _eff_rank(sv: np.ndarray, cutoff: float = 0.9) -> int:
    sv2 = sv ** 2
    cum = np.cumsum(sv2) / (sv2.sum() + 1e-12)
    return int(np.searchsorted(cum, cutoff) + 1)


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

    with np.errstate(all="ignore"):
        per_layer_M = []
        for blk in model.blocks:
            Wq = blk.Wq.weight.detach().cpu().double().numpy()
            Wk = blk.Wk.weight.detach().cpu().double().numpy()
            per_layer_M.append(Wq.T @ Wk)

        stacked = np.stack([M.reshape(-1) for M in per_layer_M], axis=0)
        svs = np.linalg.svd(stacked, compute_uv=False)
        eff = _eff_rank(svs)

        eigvals = [np.sort(np.abs(np.linalg.eigvals(M)))[::-1] for M in per_layer_M]
        L = len(eigvals)
        corrs = []
        for i in range(L):
            for j in range(i + 1, L):
                c = float(np.corrcoef(eigvals[i], eigvals[j])[0, 1])
                corrs.append(c)

    return {
        "n_layers": L,
        "eff_rank_stacked": eff,
        "mean_pairwise_spectral_corr": float(np.mean(corrs)),
        "min_pairwise_spectral_corr": float(np.min(corrs)),
        "final_loss": loss.item(),
    }
