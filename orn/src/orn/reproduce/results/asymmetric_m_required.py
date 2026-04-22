"""COU-03 — asymmetric M = AB^T is essential; symmetric LL^T fails.

Compare two tiny SharedM models on autoregressive training: one parameterises
M asymmetrically (Q = hA, K = hB) and one symmetrically (A = B = L, so
M = LL^T). The symmetric variant should train noticeably worse because
self-attention becomes commutative in the coupling direction.

Metric: best val loss after a fixed budget (val-loss lower is better).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class _Cfg:
    d: int = 64
    L: int = 2
    heads: int = 4
    V: int = 128
    S: int = 32


class _Block(nn.Module):
    def __init__(self, d, heads, A, B):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.A, self.B = A, B
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, S, D = x.shape
        h = self.ln1(x)
        q = (h @ self.A).view(B, S, self.heads, self.dh).transpose(1, 2)
        k = (h @ self.B).view(B, S, self.heads, self.dh).transpose(1, 2)
        v = self.Wv(h).view(B, S, self.heads, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.Wo(o.transpose(1, 2).contiguous().view(B, S, D))
        return x + self.ff(self.ln2(x))


class _SharedM(nn.Module):
    def __init__(self, cfg: _Cfg, symmetric: bool):
        super().__init__()
        self.cfg = cfg
        self.symmetric = symmetric
        self.tok = nn.Embedding(cfg.V, cfg.d)
        self.pos = nn.Embedding(cfg.S, cfg.d)
        if symmetric:
            self.L = nn.Parameter(torch.randn(cfg.d, cfg.d) * 0.02)
            self.A = self.L        # Q uses L
            self.B = self.L        # K uses L too → M = LL^T
        else:
            self.A = nn.Parameter(torch.randn(cfg.d, cfg.d) * 0.02)
            self.B = nn.Parameter(torch.randn(cfg.d, cfg.d) * 0.02)
        self.blocks = nn.ModuleList([_Block(cfg.d, cfg.heads, self.A, self.B) for _ in range(cfg.L)])
        self.ln = nn.LayerNorm(cfg.d); self.head = nn.Linear(cfg.d, cfg.V, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x, targets=None):
        B, S = x.shape
        h = self.tok(x) + self.pos(torch.arange(S, device=x.device))
        for b in self.blocks:
            h = b(h)
        logits = self.head(self.ln(h))
        if targets is not None:
            return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits


def _train(symmetric: bool, steps: int = 400, seed: int = 0) -> float:
    torch.manual_seed(seed)
    cfg = _Cfg()
    m = _SharedM(cfg, symmetric=symmetric)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    best = float("inf")
    for step in range(steps):
        x = torch.randint(0, cfg.V, (16, cfg.S))
        y = torch.roll(x, -1, dims=1)
        _, loss = m(x, targets=y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step >= steps - 20:
            best = min(best, loss.item())
    return best


def run(device: str | None = None, steps: int = 400) -> dict:
    asym = _train(symmetric=False, steps=steps, seed=0)
    symm = _train(symmetric=True,  steps=steps, seed=0)
    return {
        "asymmetric_best_loss": asym,
        "symmetric_best_loss":  symm,
        "asymmetric_wins":      asym < symm,
        "gap":                  symm - asym,
    }
