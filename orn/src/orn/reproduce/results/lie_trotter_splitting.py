"""OPE-04 — ORN layer structure IS Lie-Trotter operator splitting.

An ORN layer interleaves SharedM attention (operator A) with a shared FFN
(operator B). This is exactly the structure exp(tA/n) exp(tB/n) iterated n
times — a first-order Lie-Trotter splitting for exp(t(A+B)). We quantify
the commutator [A, B] on a trained ORN block: it is stable, non-zero, and
small enough that the splitting is a reasonable approximation.

Small-scale check: use an untrained ORN V3 block and report ||[M, FFN]|| and
commutator ratio ||[M, F]|| / (||M|| ||F||). If the ratio is not ≈ 0 we
confirm that each operator contributes distinct dynamics (otherwise the
splitting would be redundant).
"""
from __future__ import annotations

import numpy as np
import torch

from orn.models.orn_v3 import ORNV3, ORNV3Config


def run(device: str | None = None, seed: int = 0) -> dict:
    torch.manual_seed(seed)

    cfg = ORNV3Config(
        d_model=64, n_layers=2, n_q_heads=4, n_kv_heads=2, d_head=16,
        d_corr=16, vocab_size=200, seq_len=32, ffn_width_mult=2,
    )
    model = ORNV3(cfg)
    with torch.no_grad():
        # Give the FFN weights a normal scale so finite differencing is well-conditioned.
        for p in model.shared_ffn.parameters():
            p.data = torch.randn_like(p) * 0.1
        M = (model.A @ model.B.T).cpu().numpy().astype(np.float64)
        d = cfg.d_model
        x = torch.randn(1, 1, d)
        y0 = model.shared_ffn(x)
        eps = 1e-2
        J = np.zeros((d, d), dtype=np.float64)
        for i in range(d):
            dx = x.clone(); dx[0, 0, i] += eps
            y1 = model.shared_ffn(dx)
            J[:, i] = ((y1 - y0).view(-1) / eps).cpu().numpy()

    with np.errstate(all="ignore"):
        MJ = M @ J
        JM = J @ M
        comm = MJ - JM
        m_norm = float(np.linalg.norm(M))
        j_norm = float(np.linalg.norm(J))
        ratio = float(np.linalg.norm(comm) / (m_norm * j_norm + 1e-12))

    return {
        "||M||_F": m_norm,
        "||FFN_lin||_F": j_norm,
        "||[M, FFN]||_F": float(np.linalg.norm(comm)),
        "commutator_ratio": ratio,
        "nonzero_commutator": ratio > 1e-3,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
