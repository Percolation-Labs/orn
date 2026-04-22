from __future__ import annotations
"""
ORN — The Orbital Response Network
====================================

The central model of this project. Two versions:

- ORN (V1): LayerNorm, GELU FFN, learned positional embeddings.
  Used for all experiments through V1 (91M params, val 3.35 on OWT).

- ORNV2: RMSNorm, SwiGLU, RoPE, GQA, no bias. Modern 2025/2026 defaults.
  Used for V2 training (125M target) and forward.

Both share the defining ORN structure:
  1. SharedM = AB^T — a single coupling geometry shared across ALL layers.
     Where a transformer has 2L separate (W_Q, W_K) matrices, the ORN has
     just one pair (A, B). The coupling *rule* is memoised; per-layer
     variation comes from the evolving residual stream, not new parameters.

  2. Per-layer response heads (W_V, W_O, FFN) — the *response* to coupling
     varies with depth, so these remain per-layer.

  3. Perturbative corrections — context-gated residual corrections that
     allow small, instance-specific deviations from SharedM's geometry.

Key experimental findings:
  - M crystallises to rank 19-20 at 91M scale (only 20 coupling dimensions
    encode English grammar, out of 512).
  - Frozen-M matches full training (val 3.40 vs 3.35), confirming M encodes
    invariant relational structure.
  - Asymmetric M = AB^T is essential — symmetric LL^T fails on autoregressive tasks.
  - M IS a Koopman operator: its eigenvalues are dynamical timescales.

Config dictionary keys:
  d_model, n_layers, n_heads (V1) or n_q_heads/n_kv_heads (V2),
  d_head, d_corr, vocab_size, seq_len
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from orn.models.layers import (
    RMSNorm,
    SwiGLU,
    PerturbativeCorrection,
    precompute_rope_freqs,
    apply_rope,
)
# Note: causal masking is handled by F.scaled_dot_product_attention(is_causal=True)
# No manual mask creation needed — this halves attention memory via Flash Attention


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ORNConfig:
    """Configuration for both ORN V1 and V2."""

    d_model: int = 512
    n_layers: int = 24
    vocab_size: int = 50257
    seq_len: int = 256
    d_corr: int = 64

    # V1 uses n_heads; V2 uses n_q_heads / n_kv_heads (GQA)
    n_heads: int = 8              # V1 only
    n_q_heads: int | None = None  # V2 only (defaults to n_heads)
    n_kv_heads: int | None = None # V2 only (defaults to n_q_heads)
    d_head: int | None = None     # V2 only (defaults to d_model // n_q_heads)

    # Architecture variant
    version: int = 2  # 1 = LayerNorm/GELU/learned-pos, 2 = RMSNorm/SwiGLU/RoPE/GQA

    def __post_init__(self):
        if self.n_q_heads is None:
            self.n_q_heads = self.n_heads
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_q_heads
        if self.d_head is None:
            self.d_head = self.d_model // self.n_q_heads

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "ORNConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════════
# ORN V1 Block — LayerNorm + GELU + learned positional embeddings
# ═══════════════════════════════════════════════════════════════════════════════

class ORNV1Block(nn.Module):
    """
    One ORN V1 layer: SharedM attention + response heads + FFN + correction.

    This is the architecture used for all experiments through V1 (91M, val 3.35).
    Simple and well-tested.
    """

    def __init__(self, d_model: int, n_heads: int, A: nn.Parameter, B: nn.Parameter,
                 d_corr: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.A = A  # shared reference — NOT a copy
        self.B = B  # shared reference — NOT a copy

        # Pre-norm attention
        self.ln1 = nn.LayerNorm(d_model)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        # FFN (standard GELU, 4x expansion)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model, bias=False),
        )

        # Perturbative correction
        self.correction = PerturbativeCorrection(d_model, d_corr, use_silu=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        # SharedM attention: Q = h @ A, K = h @ B (same A, B at every layer)
        h = self.ln1(x)
        q = (h @ self.A).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = (h @ self.B).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(h).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # Memory-efficient attention (Flash Attention / SDPA)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        # Residual + response heads + FFN
        x = x + self.Wo(out)
        x = x + self.ff(self.ln2(x))

        # Perturbative correction (small, gated)
        x = self.correction(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# ORN V2 Block — RMSNorm + SwiGLU + RoPE + GQA
# ═══════════════════════════════════════════════════════════════════════════════

class ORNV2Block(nn.Module):
    """
    One ORN V2 layer with modern defaults.

    Changes from V1:
      - RMSNorm (faster, no centering)
      - SwiGLU (gated FFN, 8/3x expansion)
      - RoPE (relative position encoding, better extrapolation)
      - GQA (fewer KV heads than Q heads, saves params)
      - No bias terms (following LLaMA/Qwen convention)

    SharedM coupling is the same: Q from shared A, K from shared B.
    With GQA, K is projected from B's output down to n_kv_heads × d_head.
    """

    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int, d_head: int,
                 A: nn.Parameter, B: nn.Parameter, rope_freqs: torch.Tensor,
                 d_corr: int = 64):
        super().__init__()
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.n_rep = n_q_heads // n_kv_heads  # GQA repeat factor
        self.A = A  # shared
        self.B = B  # shared
        self.rope_freqs = rope_freqs

        self.ln1 = RMSNorm(d_model)

        # K projection: shared B maps d→d, then project to n_kv_heads × d_head
        self.Wk_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.Wv = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.Wo = nn.Linear(n_q_heads * d_head, d_model, bias=False)

        # SwiGLU FFN
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

        # Perturbative correction
        self.correction = PerturbativeCorrection(d_model, d_corr, use_silu=True)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match Q head count for GQA."""
        if self.n_rep == 1:
            return x
        B, H, S, D = x.shape
        return (
            x[:, :, None, :, :]
            .expand(B, H, self.n_rep, S, D)
            .reshape(B, H * self.n_rep, S, D)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = self.ln1(x)

        # SharedM coupling: Q from shared A, K from shared B + GQA projection
        q_all = h @ self.A  # (B, S, d_model) — full coupling
        k_raw = h @ self.B  # (B, S, d_model) — full coupling

        q = q_all.view(B, S, self.n_q_heads, self.d_head).transpose(1, 2)
        k = self.Wk_proj(k_raw).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.Wv(h).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)

        # RoPE (ensure freqs on same device)
        q = apply_rope(q, self.rope_freqs.to(q.device))
        k = apply_rope(k, self.rope_freqs.to(k.device))

        # GQA expansion for SDPA
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # Memory-efficient attention (Flash Attention / SDPA)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)

        x = x + self.Wo(out)
        x = x + self.ffn(self.ln2(x))
        x = self.correction(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Full ORN Model
# ═══════════════════════════════════════════════════════════════════════════════

class ORN(nn.Module):
    """
    The Orbital Response Network (V1).

    SharedM coupling + per-layer response + perturbative corrections.
    LayerNorm, GELU, learned positional embeddings, tied output weights.

    This is the proven architecture: 91M params, val 3.35 on OWT, HellaSwag
    33.0% (vs GPT-2 Small's 31.6% at 124M).
    """

    def __init__(self, config: ORNConfig | dict):
        super().__init__()
        if isinstance(config, dict):
            config = ORNConfig.from_dict(config)
        self.config = config
        d = config.d_model

        # Embeddings (learned positional for V1)
        self.tok_emb = nn.Embedding(config.vocab_size, d)
        self.pos_emb = nn.Embedding(config.seq_len, d)

        # ── SharedM: the defining feature ──
        # A single pair (A, B) replaces 2L separate (W_Q, W_K) matrices.
        # M = AB^T is the coupling geometry. Asymmetric by design — symmetric
        # LL^T fails on autoregressive tasks (experimentally confirmed).
        self.A = nn.Parameter(torch.randn(d, d) * 0.02)
        self.B = nn.Parameter(torch.randn(d, d) * 0.02)

        # Layers — each block references the SAME A, B (not copies)
        self.blocks = nn.ModuleList([
            ORNV1Block(d, config.n_heads, self.A, self.B, config.d_corr)
            for _ in range(config.n_layers)
        ])

        # Output head (tied to embedding)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None and m.bias.shape[0] != 1:  # preserve gate bias
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        """
        Args:
            x: (B, S) token ids
            targets: (B, S) optional target ids for loss computation

        Returns:
            logits (B, S, V) if targets is None, else (logits, loss)
        """
        B, S = x.shape
        h = self.tok_emb(x) + self.pos_emb(torch.arange(S, device=x.device))

        for block in self.blocks:
            h = block(h)  # SDPA handles causal masking internally

        logits = self.head(self.ln_f(h))

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss
        return logits

    def count_params(self) -> dict:
        """Break down parameter budget into coupling / correction / response."""
        total = sum(p.numel() for p in self.parameters())
        coupling = self.A.numel() + self.B.numel()
        correction = sum(
            sum(p.numel() for n, p in block.named_parameters()
                if "gate" in n or "correction" in n)
            for block in self.blocks
        )
        return {
            "total": total,
            "coupling": coupling,
            "correction": correction,
            "response": total - coupling - correction,
            "coupling_pct": 100.0 * coupling / total,
        }

    def spectral_diagnostics(self) -> dict:
        """
        Extract M's spectral structure for monitoring during training.

        Key observables:
          - eff_rank_90: How many singular values capture 90% of M's energy.
            V1 converges to rank 19-20 (out of 512). This is the "crystallisation".
          - condition_number: σ_max / σ_min. Kill criterion: must be > 2.
          - asymmetry: ||M - M^T|| / ||M||. Should be significantly > 0.
          - top_svs: Top 10 singular values for tracking spectral evolution.
        """
        import numpy as np
        A = self.A.detach().cpu().double().numpy()
        B = self.B.detach().cpu().double().numpy()
        with np.errstate(all="ignore"):
            M = A @ B.T
            svs = np.linalg.svd(M, compute_uv=False)
            cum_energy = np.cumsum(svs ** 2) / np.sum(svs ** 2)
        return {
            "eff_rank_90": int(np.searchsorted(cum_energy, 0.9) + 1),
            "condition_number": float(svs[0] / (svs[-1] + 1e-10)),
            "asymmetry": float(np.linalg.norm(M - M.T) / (np.linalg.norm(M) + 1e-10)),
            "top_svs": svs[:10].tolist(),
        }

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new: int = 100,
                 temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        """Autoregressive generation from a prompt."""
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


class ORNV2(nn.Module):
    """
    Orbital Response Network V2 — modern defaults.

    Same SharedM principle as V1, but with:
      - RMSNorm (no centering, faster)
      - SwiGLU (gated FFN)
      - RoPE (relative position, no learned positional embedding)
      - GQA (grouped query attention for values — fewer KV heads)
      - No bias terms

    Target config for V2-125M: d=576, L=24, 9Q/3KV, d_head=64, seq=2048.
    """

    def __init__(self, config: ORNConfig | dict):
        super().__init__()
        if isinstance(config, dict):
            config = ORNConfig.from_dict(config)
        self.config = config
        d = config.d_model

        # Embedding (no positional — RoPE handles position)
        self.tok_emb = nn.Embedding(config.vocab_size, d)

        # SharedM (scaled init for numerical stability at large d)
        self.A = nn.Parameter(torch.randn(d, d) * (0.02 / math.sqrt(d)))
        self.B = nn.Parameter(torch.randn(d, d) * (0.02 / math.sqrt(d)))

        # Precompute RoPE frequencies (2x seq_len for safety)
        self.register_buffer("rope_freqs", precompute_rope_freqs(config.d_head, config.seq_len * 2))

        # Layers
        self.blocks = nn.ModuleList([
            ORNV2Block(d, config.n_q_heads, config.n_kv_heads, config.d_head,
                       self.A, self.B, self.rope_freqs, config.d_corr)
            for _ in range(config.n_layers)
        ])

        # Output
        self.ln_f = RMSNorm(d)
        self.head = nn.Linear(d, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        B, S = x.shape
        h = self.tok_emb(x)

        for block in self.blocks:
            h = block(h)  # SDPA handles causal masking internally

        logits = self.head(self.ln_f(h))

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss
        return logits

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        coupling = self.A.numel() + self.B.numel()
        correction = sum(
            sum(p.numel() for n, p in block.named_parameters()
                if "gate" in n or "correction" in n)
            for block in self.blocks
        )
        return {
            "total": total,
            "coupling": coupling,
            "correction": correction,
            "response": total - coupling - correction,
            "coupling_pct": 100.0 * coupling / total,
        }

    def spectral_diagnostics(self) -> dict:
        import numpy as np
        A = self.A.detach().cpu().double().numpy()
        B = self.B.detach().cpu().double().numpy()
        with np.errstate(all="ignore"):
            M = A @ B.T
            svs = np.linalg.svd(M, compute_uv=False)
            cum_energy = np.cumsum(svs ** 2) / np.sum(svs ** 2)
        return {
            "eff_rank_90": int(np.searchsorted(cum_energy, 0.9) + 1),
            "condition_number": float(svs[0] / (svs[-1] + 1e-10)),
            "asymmetry": float(np.linalg.norm(M - M.T) / (np.linalg.norm(M) + 1e-10)),
            "top_svs": svs[:10].tolist(),
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


# ═══════════════════════════════════════════════════════════════════════════════
# Preset Configs
# ═══════════════════════════════════════════════════════════════════════════════

# V1: The proven 91M architecture (val 3.35 on OWT)
ORN_91M_V1 = ORNConfig(
    d_model=512, n_layers=24, n_heads=8, d_corr=64,
    vocab_size=50257, seq_len=256, version=1,
)

# V2: Target 125M with modern defaults
ORN_125M_V2 = ORNConfig(
    d_model=576, n_layers=24, n_q_heads=9, n_kv_heads=3,
    d_head=64, d_corr=64, vocab_size=50304, seq_len=2048, version=2,  # vocab must cover tiktoken GPT-2 (50257)
)

# Dry run: tiny model for testing pipelines
ORN_DRY_RUN = ORNConfig(
    d_model=64, n_layers=4, n_heads=4, d_corr=16,
    vocab_size=50257, seq_len=64, version=1,
)
