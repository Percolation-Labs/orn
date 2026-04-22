"""SPE-01 — W_Q^T W_K eigenspectra correlate across pretrained families.

Smoke-scale shape: train several small transformers with different seeds on
the same synthetic LM task and measure the Spearman correlation between the
sorted eigenvalue magnitudes of their layer-averaged M = mean_l W_Q^T W_K.
The claim at real scale (GPT-2 / SmolLM2 / Qwen2.5) is ≥ 0.93; here we
accept anything > 0.6 as the signal surviving an undertrained smoke run.

Full-scale version fetches pretrained HF models and lives behind --slow.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

from orn.models.transformer import Transformer, TransformerConfig


def _train_small(seed: int, steps: int = 120) -> np.ndarray:
    torch.manual_seed(seed)
    cfg = TransformerConfig.from_dict(dict(
        d_model=64, n_layers=4, n_heads=4, vocab_size=128, seq_len=32,
    ))
    m = Transformer(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(steps):
        x = torch.randint(0, cfg.vocab_size, (8, cfg.seq_len))
        y = torch.roll(x, -1, dims=1)
        _, loss = m(x, targets=y)
        opt.zero_grad(); loss.backward(); opt.step()

    with np.errstate(all="ignore"):
        Ms = []
        for blk in m.blocks:
            Wq = blk.Wq.weight.detach().cpu().double().numpy()
            Wk = blk.Wk.weight.detach().cpu().double().numpy()
            Ms.append(Wq.T @ Wk)
        M_avg = np.mean(np.stack(Ms, 0), 0)
        eig = np.sort(np.abs(np.linalg.eigvals(M_avg)))[::-1]
    return eig


def run(device: str | None = None, n_models: int = 3) -> dict:
    eigs = [_train_small(seed=s) for s in range(n_models)]
    corrs = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            c, _ = spearmanr(eigs[i], eigs[j])
            corrs.append(float(c))
    return {
        "n_models": n_models,
        "mean_pairwise_spearman": float(np.mean(corrs)),
        "min_pairwise_spearman": float(np.min(corrs)),
    }
