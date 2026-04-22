"""OPE-01 — function tokens receive consistent attention; content tokens do not.

On a trained ORN, token types that *couple* (function words like "the", "of")
draw stable attention masses across contexts, while content tokens (names,
numbers) vary with context. We approximate this with a small SharedM model
trained on randomly-generated "function + content" sequences where a special
"function" token is planted at deterministic positions and content tokens are
randomly chosen. After training, the attention over function-token positions
should have low variance across prompts; content-token attention variance
should be higher.

Metric: ratio var_content / var_function. Expected > 1 (content is noisier).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB = 64
SEQ = 32
D = 64
HEADS = 4
LAYERS = 2
FUNC_TOKEN = 1     # planted at positions 0, 8, 16, 24


def _batch(batch_size: int = 16):
    x = torch.randint(2, VOCAB, (batch_size, SEQ))
    x[:, ::8] = FUNC_TOKEN
    y = torch.roll(x, -1, dims=1)
    return x, y


class _Attn(nn.Module):
    def __init__(self, A, B):
        super().__init__()
        self.A, self.B = A, B
        self.dh = D // HEADS
        self.Wv = nn.Linear(D, D, bias=False)
        self.Wo = nn.Linear(D, D, bias=False)

    def forward(self, x, return_attn: bool = False):
        B_, S, _ = x.shape
        q = (x @ self.A).view(B_, S, HEADS, self.dh).transpose(1, 2)
        k = (x @ self.B).view(B_, S, HEADS, self.dh).transpose(1, 2)
        v = self.Wv(x).view(B_, S, HEADS, self.dh).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / (self.dh ** 0.5)
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        o = attn @ v
        out = self.Wo(o.transpose(1, 2).contiguous().view(B_, S, D))
        if return_attn:
            return out, attn
        return out


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(SEQ, D)
        self.A = nn.Parameter(torch.randn(D, D) * 0.02)
        self.B = nn.Parameter(torch.randn(D, D) * 0.02)
        self.blocks = nn.ModuleList([_Attn(self.A, self.B) for _ in range(LAYERS)])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False); self.head.weight = self.tok.weight

    def forward(self, x, targets=None, return_attn: bool = False):
        B_, S = x.shape
        h = self.tok(x) + self.pos(torch.arange(S))
        attn0 = None
        for i, b in enumerate(self.blocks):
            if return_attn and i == 0:
                out, attn0 = b(h, return_attn=True); h = h + out
            else:
                h = h + b(h)
        logits = self.head(self.ln(h))
        if targets is not None:
            return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return (logits, attn0) if return_attn else logits


def run(device: str | None = None, train_steps: int = 300, n_probe: int = 16,
        seed: int = 0) -> dict:
    torch.manual_seed(seed)
    m = _Model()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(train_steps):
        x, y = _batch()
        _, loss = m(x, targets=y)
        opt.zero_grad(); loss.backward(); opt.step()

    # Probe: collect the attention rows AT positions 7, 15, 23 (just after a
    # function token) across many prompts. Measure variance of the column
    # attending to the function-token positions vs. to content positions.
    m.eval()
    func_cols = list(range(0, SEQ, 8))
    content_cols = [c for c in range(SEQ) if c not in func_cols]
    row_idx = 15  # mid-sequence probe row

    func_masses = []
    content_masses = []
    with torch.no_grad():
        for _ in range(n_probe):
            x, _ = _batch(batch_size=8)
            _, attn = m(x, return_attn=True)
            # attn: (B, H, S, S)
            mass = attn.mean(dim=1)[:, row_idx, :]  # (B, S)
            func_masses.append(mass[:, func_cols].sum(-1).cpu().numpy())
            content_masses.append(mass[:, content_cols].sum(-1).cpu().numpy())

    func = np.concatenate(func_masses)
    cont = np.concatenate(content_masses)
    return {
        "function_attn_mean": float(func.mean()),
        "function_attn_std":  float(func.std()),
        "content_attn_mean":  float(cont.mean()),
        "content_attn_std":   float(cont.std()),
        "std_ratio_content_over_function": float(cont.std() / (func.std() + 1e-8)),
    }
