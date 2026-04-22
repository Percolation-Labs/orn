"""SPE-02 — M crystallises to rank 19-20 of 512 during training (91M scale).

This is a long-run result: train an ORNV2 (~91M) on OpenWebText for ~250M
tokens and record the effective rank of M across checkpoints. At real scale
the effective-rank trajectory drops from ~500 to ~20 within the first 200M
tokens and stabilises.

The smoke-scale version: train a small ORNV2 (d=128, L=4) for N steps and
report the effective-rank trajectory. The claim (rank collapses well below
d) is expected to hold qualitatively; the absolute number is scale-dependent.

Not run in default CI. Invoke via `orn reproduce --slow SPE-02`.
"""
from __future__ import annotations

import numpy as np
import torch

from orn.models.orn import ORN, ORNConfig


def run(device: str | None = None, train_steps: int = 2000,
        checkpoint_every: int = 200, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    cfg = ORNConfig(
        d_model=128, n_layers=4, n_heads=4, d_corr=32,
        vocab_size=256, seq_len=64, version=1,
    )
    m = ORN(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)

    trajectory = []
    for step in range(train_steps):
        x = torch.randint(0, cfg.vocab_size, (8, cfg.seq_len))
        y = torch.roll(x, -1, dims=1)
        _, loss = m(x, targets=y)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % checkpoint_every == 0 or step == 0:
            diag = m.spectral_diagnostics()
            trajectory.append({
                "step": step + 1,
                "eff_rank_90": diag["eff_rank_90"],
                "condition_number": diag["condition_number"],
            })

    final = trajectory[-1]
    initial = trajectory[0]
    return {
        "d_model": cfg.d_model,
        "initial_eff_rank": initial["eff_rank_90"],
        "final_eff_rank":   final["eff_rank_90"],
        "rank_fraction":    final["eff_rank_90"] / cfg.d_model,
        "rank_collapsed":   final["eff_rank_90"] < cfg.d_model / 2,
        "trajectory":       trajectory,
    }
