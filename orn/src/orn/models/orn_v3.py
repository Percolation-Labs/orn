from __future__ import annotations
"""
ORN V3 (FullORN) — finalised architecture.
===========================================

V3 adds a Lx-wider *shared* FFN on top of V2's SharedM coupling. At matched
parameter count, one wide shared FFN beats L per-layer FFNs (empirical finding).

Shared across all layers:
  M = AB^T    — coupling geometry
  SharedFFN   — wide SwiGLU, d_ff = (8/3)d * ffn_width_mult, rounded to 64

Per-layer:
  Wk_proj, Wv, Wo   — response heads (GQA)
  gate, corr        — perturbative correction

Also supports per-sample low-rank attention deltas (U, V) for the LORN
HypernetBridge (see orn.models.lorn).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from orn.models.layers import RMSNorm, precompute_rope_freqs, apply_rope


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ORNV3Config:
    d_model: int = 576
    n_layers: int = 24
    n_q_heads: int = 9
    n_kv_heads: int = 3
    d_head: int = 64
    vocab_size: int = 49152
    seq_len: int = 2048
    ffn_width_mult: int | None = None   # default = n_layers (matches param budget)
    d_corr: int = 64

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "ORNV3Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════════
# Wide shared SwiGLU FFN
# ═══════════════════════════════════════════════════════════════════════════════

class WideSwiGLU(nn.Module):
    """Wide SwiGLU FFN shared across all layers."""

    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ═══════════════════════════════════════════════════════════════════════════════
# Block
# ═══════════════════════════════════════════════════════════════════════════════

class ORNV3Block(nn.Module):
    """SharedM attention + Wide SharedFFN + perturbative correction.

    Accepts an optional per-sample low-rank delta on the attention output
    (used by the LORN HypernetBridge)."""

    def __init__(self, d: int, n_q_heads: int, n_kv_heads: int, d_head: int,
                 A: nn.Parameter, B: nn.Parameter,
                 shared_ffn: WideSwiGLU, rope_freqs: torch.Tensor,
                 d_corr: int = 64):
        super().__init__()
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.n_rep = n_q_heads // n_kv_heads
        self.A = A
        self.B = B
        self.shared_ffn = shared_ffn
        self.rope_freqs = rope_freqs

        self.ln_attn = RMSNorm(d)
        self.ln_ffn = RMSNorm(d)
        self.ln_corr = RMSNorm(d)

        self.Wk_proj = nn.Linear(d, n_kv_heads * d_head, bias=False)
        self.Wv = nn.Linear(d, n_kv_heads * d_head, bias=False)
        self.Wo = nn.Linear(n_q_heads * d_head, d, bias=False)

        self.gate = nn.Linear(d, 1, bias=True)
        self.corr = nn.Sequential(
            nn.Linear(d, d_corr, bias=False),
            nn.SiLU(),
            nn.Linear(d_corr, d, bias=False),
        )
        nn.init.constant_(self.gate.bias, -3.0)
        nn.init.zeros_(self.corr[-1].weight)

    def forward(self, x, is_causal: bool = True, attn_mask=None, delta=None):
        B, S, D = x.shape
        h = self.ln_attn(x)

        q = (h @ self.A).view(B, S, self.n_q_heads, self.d_head).transpose(1, 2)
        k_raw = h @ self.B
        k = self.Wk_proj(k_raw).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.Wv(h).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)

        q = apply_rope(q, self.rope_freqs.to(q.device))
        k = apply_rope(k, self.rope_freqs.to(k.device))

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        if attn_mask is not None:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        attn_out = self.Wo(attn_out)
        if delta is not None:
            U, V = delta  # (B, d, r), (B, r, d)
            attn_out = attn_out + torch.bmm(attn_out, torch.bmm(U, V))
        x = x + attn_out
        x = x + self.shared_ffn(self.ln_ffn(x))
        h_c = self.ln_corr(x)
        x = x + torch.sigmoid(self.gate(h_c)) * self.corr(h_c)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Full model
# ═══════════════════════════════════════════════════════════════════════════════

class ORNV3(nn.Module):
    """ORN V3 (FullORN). SharedM + wide SharedFFN + per-layer response."""

    def __init__(self, config: ORNV3Config | dict):
        super().__init__()
        if isinstance(config, dict):
            config = ORNV3Config.from_dict(config)
        self.config = config
        d = config.d_model

        assert d == config.n_q_heads * config.d_head, \
            f"d={d} != n_q_heads*d_head={config.n_q_heads * config.d_head}"
        assert config.n_q_heads % config.n_kv_heads == 0, \
            "n_q_heads must be divisible by n_kv_heads"

        self.A = nn.Parameter(torch.randn(d, d) * 0.02)
        self.B = nn.Parameter(torch.randn(d, d) * 0.02)

        std_d_ff = int(8 / 3 * d)
        std_d_ff = ((std_d_ff + 63) // 64) * 64
        width_mult = config.ffn_width_mult or config.n_layers
        wide_d_ff = ((std_d_ff * width_mult + 63) // 64) * 64
        self.shared_ffn = WideSwiGLU(d, wide_d_ff)

        self.tok_emb = nn.Embedding(config.vocab_size, d)

        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(config.d_head, config.seq_len + 16),
            persistent=False,
        )

        self.blocks = nn.ModuleList([
            ORNV3Block(
                d, config.n_q_heads, config.n_kv_heads, config.d_head,
                self.A, self.B, self.shared_ffn, self.rope_freqs,
                d_corr=config.d_corr,
            )
            for _ in range(config.n_layers)
        ])

        self.ln_f = RMSNorm(d)
        self.head = nn.Linear(d, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens, targets=None, is_causal: bool = True,
                attn_mask=None, deltas=None):
        """deltas: optional list of per-layer (U, V) tuples for LORN bridge."""
        B, S = tokens.shape
        h = self.tok_emb(tokens)

        if deltas is None:
            deltas = [None] * len(self.blocks)
        assert len(deltas) == len(self.blocks), \
            f"deltas length {len(deltas)} != n_layers {len(self.blocks)}"

        for block, d in zip(self.blocks, deltas):
            h = block(h, is_causal=is_causal, attn_mask=attn_mask, delta=d)

        logits = self.head(self.ln_f(h))

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        coupling = self.A.numel() + self.B.numel()
        shared_ffn = sum(p.numel() for p in self.shared_ffn.parameters())
        return {
            "total": total,
            "coupling": coupling,
            "shared_ffn": shared_ffn,
            "per_layer": total - coupling - shared_ffn - self.tok_emb.weight.numel(),
            "coupling_pct": 100.0 * coupling / total,
        }

    def spectral_diagnostics(self) -> dict:
        import numpy as np
        A = self.A.detach().cpu().numpy()
        B = self.B.detach().cpu().numpy()
        M = A @ B.T
        svs = np.linalg.svd(M, compute_uv=False)
        cum_energy = np.cumsum(svs ** 2) / np.sum(svs ** 2)
        return {
            "eff_rank_90": int(np.searchsorted(cum_energy, 0.9) + 1),
            "condition_number": float(svs[0] / (svs[-1] + 1e-10)),
            "asymmetry": float(np.linalg.norm(M - M.T) / (np.linalg.norm(M) + 1e-10)),
            "top_svs": svs[:10].tolist(),
        }

    def freeze_backbone(self):
        """Freeze SharedM + SharedFFN for adaptation / frozen-M experiments."""
        self.A.requires_grad_(False)
        self.B.requires_grad_(False)
        for p in self.shared_ffn.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        self.A.requires_grad_(True)
        self.B.requires_grad_(True)
        for p in self.shared_ffn.parameters():
            p.requires_grad_(True)

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


# ═══════════════════════════════════════════════════════════════════════════════
# Preset constructors
# ═══════════════════════════════════════════════════════════════════════════════

def orn_v3_61m(vocab: int = 49152, seq_len: int = 2048) -> ORNV3:
    return ORNV3(ORNV3Config(
        d_model=576, n_q_heads=9, n_kv_heads=3, d_head=64,
        n_layers=24, vocab_size=vocab, seq_len=seq_len,
        ffn_width_mult=6, d_corr=64,
    ))


def orn_v3_108m(vocab: int = 49152, seq_len: int = 2048) -> ORNV3:
    return ORNV3(ORNV3Config(
        d_model=576, n_q_heads=9, n_kv_heads=3, d_head=64,
        n_layers=24, vocab_size=vocab, seq_len=seq_len, d_corr=64,
    ))


def orn_v3_350m(vocab: int = 49152, seq_len: int = 2048) -> ORNV3:
    return ORNV3(ORNV3Config(
        d_model=1024, n_q_heads=16, n_kv_heads=4, d_head=64,
        n_layers=24, vocab_size=vocab, seq_len=seq_len, d_corr=64,
    ))


def orn_v3_1b(vocab: int = 49152, seq_len: int = 2048) -> ORNV3:
    return ORNV3(ORNV3Config(
        d_model=2048, n_q_heads=16, n_kv_heads=4, d_head=128,
        n_layers=24, vocab_size=vocab, seq_len=seq_len, d_corr=128,
    ))
