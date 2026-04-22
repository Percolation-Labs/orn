from __future__ import annotations
"""
LORN v4 — Lateralised Two-Hemisphere Architecture
==================================================

Three components:
  R (observer)   : VQREncoder — attention + VQ codebook + entropy anti-collapse
  Bridge (ctrl)  : HypernetBridge — low-rank weight delta generated per-sample
  L (predictor)  : ORNV3 — shared-M coupling + wide shared FFN + corrections

Validated in the concept-circuit battery (see lorn/paper/). Core findings:
  CC-13b: R discovers latent multiset structure, purity 0.38 -> 0.97
  CC-18:  hypernet bridge beats AdaLN by 12 points under joint training
  CC-15:  do NOT mix VQ with contrastive on a shared encoder

Training guidelines:
  joint end-to-end with task_CE + vq_loss_scale * (commit + codebook - entropy*w)
  VQ codebook K <= expected latent class count (oversizing hurts purity, CC-16)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from orn.models.orn_v3 import ORNV3, ORNV3Config


# ═══════════════════════════════════════════════════════════════════════════════
# MPS-safe primitives
# ═══════════════════════════════════════════════════════════════════════════════

class _SelfAttn(nn.Module):
    """Minimal MPS-safe self-attention used inside the R-hemisphere encoder."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.nh = heads
        self.dh = d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.nh, self.dh).transpose(1, 2)
        k = k.view(B, T, self.nh, self.dh).transpose(1, 2)
        v = v.view(B, T, self.nh, self.dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v)
        return self.o(a.transpose(1, 2).contiguous().view(B, T, D))


def _mps_safe_dists(h: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """||h - c||_2 per pair, avoids aten::_cdist_backward (unimplemented on MPS)."""
    hh = (h * h).sum(-1, keepdim=True)
    cc = (codebook * codebook).sum(-1)[None, :]
    hc = h @ codebook.T
    return (hh + cc - 2 * hc).clamp_min(1e-12).sqrt()


# ═══════════════════════════════════════════════════════════════════════════════
# R-hemisphere: VQ-based structural observer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VQRConfig:
    vocab_size: int
    d: int = 128
    T_max: int = 128
    heads: int = 4
    depth: int = 2
    codebook_size: int = 16
    entropy_weight: float = 0.3


class VQREncoder(nn.Module):
    """Attention encoder + VQ codebook + entropy anti-collapse."""

    def __init__(self, cfg: VQRConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d)
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                ln1=nn.LayerNorm(cfg.d),
                attn=_SelfAttn(cfg.d, cfg.heads),
                ln2=nn.LayerNorm(cfg.d),
                mlp=nn.Sequential(
                    nn.Linear(cfg.d, 4 * cfg.d), nn.GELU(),
                    nn.Linear(4 * cfg.d, cfg.d)),
            )) for _ in range(cfg.depth)
        ])
        self.ln_f = nn.LayerNorm(cfg.d)
        self.codebook = nn.Parameter(torch.randn(cfg.codebook_size, cfg.d) * 0.1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """S_n-invariant pooled encoding (B, d)."""
        h = self.emb(x)
        for b in self.blocks:
            h = h + b["attn"](b["ln1"](h))
            h = h + b["mlp"](b["ln2"](h))
        return self.ln_f(h).mean(dim=1)

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        dists = _mps_safe_dists(h, self.codebook)
        chosen = dists.argmin(dim=-1)
        codes = self.codebook[chosen]
        alpha = h + (codes - h).detach()   # straight-through estimator
        commit_loss = F.mse_loss(h, codes.detach())
        codebook_loss = F.mse_loss(codes, h.detach())
        soft = F.softmax(-dists, dim=-1)
        marg = soft.mean(dim=0)
        entropy = -(marg * marg.clamp_min(1e-12).log()).sum()
        aux = {
            "commit": commit_loss,
            "codebook": codebook_loss,
            "entropy": entropy,
            "chosen": chosen,
            "alpha": alpha,
        }
        return alpha, aux

    def vq_loss(self, aux: dict) -> torch.Tensor:
        return (aux["commit"] + aux["codebook"]
                - self.cfg.entropy_weight * aux["entropy"])


# ═══════════════════════════════════════════════════════════════════════════════
# Bridge: hypernet-generated low-rank weight deltas
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BridgeConfig:
    d_alpha: int
    d_model: int
    rank: int = 4
    n_inject_layers: int = 2


class HypernetBridge(nn.Module):
    """Per-sample low-rank delta generator. Output is n_inject_layers (U, V) pairs."""

    def __init__(self, cfg: BridgeConfig):
        super().__init__()
        self.cfg = cfg
        self.U = nn.Linear(cfg.d_alpha, cfg.d_model * cfg.rank * cfg.n_inject_layers)
        self.V = nn.Linear(cfg.d_alpha, cfg.rank * cfg.d_model * cfg.n_inject_layers)
        self.ln = nn.LayerNorm(cfg.d_alpha)

    def forward(self, alpha: torch.Tensor):
        B = alpha.shape[0]
        a = self.ln(alpha)
        U = self.U(a).view(B, self.cfg.n_inject_layers, self.cfg.d_model, self.cfg.rank)
        V = self.V(a).view(B, self.cfg.n_inject_layers, self.cfg.rank, self.cfg.d_model)
        return [(U[:, l], V[:, l]) for l in range(self.cfg.n_inject_layers)]


# ═══════════════════════════════════════════════════════════════════════════════
# Full LORN = R + Bridge + L (ORNV3)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LORNConfig:
    vocab_size: int
    T_max: int = 128
    # R sub-config
    d_r: int = 128
    r_depth: int = 2
    r_heads: int = 4
    codebook_size: int = 16
    entropy_weight: float = 0.3
    # L sub-config (ORNV3)
    d_l: int = 256
    l_layers: int = 4
    l_q_heads: int = 4
    l_kv_heads: int = 2
    l_d_head: int = 64
    l_ffn_width_mult: Optional[int] = None
    l_d_corr: int = 64
    # Bridge
    bridge_rank: int = 4
    n_inject_layers: int = 2
    # Training
    vq_loss_scale: float = 0.1

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "LORNConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class LORN(nn.Module):
    """Two-hemisphere model: VQREncoder + HypernetBridge + ORNV3 predictor."""

    def __init__(self, cfg: LORNConfig):
        super().__init__()
        self.cfg = cfg

        self.R = VQREncoder(VQRConfig(
            vocab_size=cfg.vocab_size, d=cfg.d_r, T_max=cfg.T_max,
            heads=cfg.r_heads, depth=cfg.r_depth,
            codebook_size=cfg.codebook_size, entropy_weight=cfg.entropy_weight,
        ))

        self.bridge = HypernetBridge(BridgeConfig(
            d_alpha=cfg.d_r, d_model=cfg.d_l,
            rank=cfg.bridge_rank, n_inject_layers=cfg.n_inject_layers,
        ))

        self.L = ORNV3(ORNV3Config(
            d_model=cfg.d_l, n_layers=cfg.l_layers,
            n_q_heads=cfg.l_q_heads, n_kv_heads=cfg.l_kv_heads,
            d_head=cfg.l_d_head, vocab_size=cfg.vocab_size,
            seq_len=cfg.T_max, ffn_width_mult=cfg.l_ffn_width_mult,
            d_corr=cfg.l_d_corr,
        ))

    def forward(self, x_full: torch.Tensor, x_l: Optional[torch.Tensor] = None,
                targets: Optional[torch.Tensor] = None):
        """x_full: sequence R observes (S_n-invariant). x_l: sequence L predicts."""
        if x_l is None:
            x_l = x_full
        alpha, aux = self.R(x_full)
        per_layer_deltas = self.bridge(alpha)
        # Inject into the TOP n layers (last n blocks of L).
        deltas = [None] * (len(self.L.blocks) - len(per_layer_deltas)) + per_layer_deltas
        result = self.L(x_l, targets=targets, deltas=deltas)
        if targets is not None:
            logits, task_loss = result
            return logits, task_loss, aux
        return result, aux

    def joint_loss(self, task_loss: torch.Tensor, aux: dict) -> torch.Tensor:
        return task_loss + self.cfg.vq_loss_scale * self.R.vq_loss(aux)


# ═══════════════════════════════════════════════════════════════════════════════
# Preset constructors
# ═══════════════════════════════════════════════════════════════════════════════

def lorn_v4_smoke(vocab_size: int = 256, T_max: int = 32) -> LORN:
    """Tiny LORN for pipeline tests. ~1M params."""
    return LORN(LORNConfig(
        vocab_size=vocab_size, T_max=T_max,
        d_r=64, r_depth=2, r_heads=4, codebook_size=4,
        d_l=128, l_layers=4, l_q_heads=4, l_kv_heads=2, l_d_head=32,
        l_ffn_width_mult=2, bridge_rank=2, n_inject_layers=2,
    ))


def lorn_v4_66m(vocab_size: int = 49152, T_max: int = 512) -> LORN:
    """Production LORN v4. ~66M params total (L ~47M + R ~14M + bridge ~4M)."""
    return LORN(LORNConfig(
        vocab_size=vocab_size, T_max=T_max,
        d_r=256, r_depth=3, r_heads=4, codebook_size=32,
        d_l=512, l_layers=12, l_q_heads=8, l_kv_heads=2, l_d_head=64,
        l_ffn_width_mult=6, bridge_rank=4, n_inject_layers=4,
    ))
