from __future__ import annotations
"""
Standard Transformer Baseline
===============================

A conventional transformer for controlled comparison with ORN and CORN.
Uses the same building blocks (RMSNorm, SwiGLU) but with PER-LAYER W_Q and W_K
instead of SharedM.

This is the STORE model in our terminology: it stores a separate coupling rule
at each layer, while ORN memoises a single shared rule.

When comparing, we match TOTAL parameter count (not architecture shape). The
transformer gets 2Ld² parameters for coupling (L layers × W_Q + W_K), while
ORN gets 2d² (one A + B pair) and redistributes the savings to model width.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from orn.models.layers import RMSNorm, SwiGLU, make_causal_mask


@dataclass
class TransformerConfig:
    """Configuration for the baseline transformer."""

    d_model: int = 512
    n_layers: int = 24
    n_heads: int = 8
    vocab_size: int = 50257
    seq_len: int = 256
    ffn_multiplier: float = 4.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TransformerBlock(nn.Module):
    """Standard transformer block with per-layer W_Q, W_K, W_V, W_O + FFN."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.ln1 = RMSNorm(d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, S, D = x.shape
        h = self.ln1(x)

        q = self.Wq(h).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(h).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(h).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if causal_mask is not None:
            scores = scores + causal_mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)

        x = x + self.Wo(out)
        x = x + self.ffn(self.ln2(x))
        return x


class Transformer(nn.Module):
    """
    Standard transformer baseline for controlled comparison.

    Architecture: pre-norm, RMSNorm, SwiGLU, learned positional embeddings,
    tied output weights. Per-layer W_Q and W_K (the STORE model).

    For fair comparison with ORN: use the same total parameter count.
    ORN saves (2L-2)·d² coupling params; those are redistributed to width.
    """

    def __init__(self, config: TransformerConfig | dict):
        super().__init__()
        if isinstance(config, dict):
            config = TransformerConfig.from_dict(config)
        self.config = config
        d = config.d_model

        self.tok_emb = nn.Embedding(config.vocab_size, d)
        self.pos_emb = nn.Embedding(config.seq_len, d)

        self.blocks = nn.ModuleList([
            TransformerBlock(d, config.n_heads)
            for _ in range(config.n_layers)
        ])

        self.ln_f = RMSNorm(d)
        self.head = nn.Linear(d, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        B, S = x.shape
        h = self.tok_emb(x) + self.pos_emb(torch.arange(S, device=x.device))
        mask = make_causal_mask(S, x.device)

        for block in self.blocks:
            h = block(h, causal_mask=mask)

        logits = self.head(self.ln_f(h))

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss
        return logits

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        coupling = sum(
            sum(p.numel() for n, p in block.named_parameters() if "Wq" in n or "Wk" in n)
            for block in self.blocks
        )
        return {
            "total": total,
            "coupling": coupling,
            "response": total - coupling,
            "coupling_pct": 100.0 * coupling / total,
        }

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new: int = 100,
                 temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        self.eval()
        ids = prompt_ids.unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids
        for _ in range(max_new):
            x = ids[:, -self.config.seq_len:]
            logits = self(x)[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids[0]
