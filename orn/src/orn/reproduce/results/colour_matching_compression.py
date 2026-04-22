"""COM-01 — SharedM matches per-layer coupling on a provably-invariant task.

Colour-matching task: each sequence has a random palette of K colours, and
the model must predict the next colour in a periodic sequence. The task is
invariant across "layers" by construction — the coupling rule does not depend
on depth — so a model with shared M should match a model with per-layer M
at a fraction of the coupling-parameter count.

Metric: best-val loss ratio MEMOISE / STORE. Claim: ≤ 1.02 (i.e. within 2%).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB = 32
SEQ = 32
D = 64
HEADS = 4
LAYERS = 4


def _palette_batch(batch: int = 32, K: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row is a random permutation of K colours, repeated to fill SEQ."""
    x = torch.zeros(batch, SEQ, dtype=torch.long)
    for b in range(batch):
        palette = torch.randperm(VOCAB - 1)[:K] + 1  # reserve 0 for padding
        reps = (SEQ + K - 1) // K
        seq = palette.repeat(reps)[:SEQ]
        x[b] = seq
    y = torch.roll(x, -1, dims=1)
    return x, y


class _Block(nn.Module):
    def __init__(self, A, B):
        super().__init__()
        self.A, self.B = A, B
        self.dh = D // HEADS
        self.Wv = nn.Linear(D, D, bias=False)
        self.Wo = nn.Linear(D, D, bias=False)
        self.ln = nn.LayerNorm(D)

    def forward(self, x):
        B_, S, _ = x.shape
        h = self.ln(x)
        q = (h @ self.A).view(B_, S, HEADS, self.dh).transpose(1, 2)
        k = (h @ self.B).view(B_, S, HEADS, self.dh).transpose(1, 2)
        v = self.Wv(h).view(B_, S, HEADS, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return x + self.Wo(o.transpose(1, 2).contiguous().view(B_, S, D))


class _Model(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(SEQ, D)
        if mode == "memoise":
            A = nn.Parameter(torch.randn(D, D) * 0.02)
            B = nn.Parameter(torch.randn(D, D) * 0.02)
            self.A, self.B = A, B
            self.blocks = nn.ModuleList([_Block(A, B) for _ in range(LAYERS)])
        elif mode == "store":
            self.As = nn.ParameterList(nn.Parameter(torch.randn(D, D) * 0.02) for _ in range(LAYERS))
            self.Bs = nn.ParameterList(nn.Parameter(torch.randn(D, D) * 0.02) for _ in range(LAYERS))
            self.blocks = nn.ModuleList([_Block(self.As[i], self.Bs[i]) for i in range(LAYERS)])
        else:
            raise ValueError(mode)
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x, targets=None):
        B_, S = x.shape
        h = self.tok(x) + self.pos(torch.arange(S, device=x.device))
        for b in self.blocks:
            h = b(h)
        logits = self.head(self.ln(h))
        if targets is not None:
            return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits


def _train(mode: str, steps: int = 400, seed: int = 0) -> tuple[float, int]:
    torch.manual_seed(seed)
    m = _Model(mode)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    best = float("inf")
    for step in range(steps):
        x, y = _palette_batch()
        _, loss = m(x, targets=y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step >= steps - 20:
            best = min(best, loss.item())
    coupling = 0
    if mode == "memoise":
        coupling = m.A.numel() + m.B.numel()
    else:
        coupling = sum(p.numel() for p in m.As) + sum(p.numel() for p in m.Bs)
    return best, coupling


def run(device: str | None = None, steps: int = 400) -> dict:
    mem_loss, mem_params = _train("memoise", steps=steps)
    sto_loss, sto_params = _train("store",   steps=steps)
    return {
        "memoise_best_loss":   mem_loss,
        "store_best_loss":     sto_loss,
        "loss_ratio":          mem_loss / max(sto_loss, 1e-8),
        "coupling_params_memoise": mem_params,
        "coupling_params_store":   sto_params,
        "compression_factor":  sto_params / max(mem_params, 1),
    }
