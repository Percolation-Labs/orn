from __future__ import annotations
"""
Shared building blocks for ORN models.
=======================================

These are the standard modern transformer components (2025/2026 best practices)
used across ORN, CORN, and the transformer baseline:

- RMSNorm: Root Mean Square Layer Normalisation (faster than LayerNorm, no centering)
- SwiGLU: Gated linear unit with SiLU activation (Shazeer 2020, adopted by LLaMA/Gemma)
- RoPE: Rotary Position Embeddings (Su et al. 2021, de facto standard)

Also includes the perturbative correction block — the ORN-specific context-gated
residual that allows per-instance deviations from M's coupling geometry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ═══════════════════════════════════════════════════════════════════════════════
# RMSNorm — replaces LayerNorm in modern architectures
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).

    Unlike LayerNorm, RMSNorm does not subtract the mean — it only rescales by
    the root mean square. This is ~10% faster and empirically equivalent. Used
    by LLaMA, Gemma, Qwen, SmolLM, and most post-2023 models.

    Args:
        d: Feature dimension
        eps: Numerical stability constant
    """

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ═══════════════════════════════════════════════════════════════════════════════
# RoPE — Rotary Position Embeddings
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_rope_freqs(d_head: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the complex-valued rotation frequencies for RoPE.

    RoPE encodes position by rotating Q and K vectors in 2D subspaces. Each
    subspace rotates at a different frequency, creating a multi-scale position
    encoding that decays with distance (tokens further apart have less aligned
    rotations).

    Args:
        d_head: Head dimension (must be even)
        max_seq_len: Maximum sequence length to precompute for
        theta: Base frequency (10000 is standard; larger = longer range)

    Returns:
        Complex tensor of shape (max_seq_len, d_head // 2) containing rotation
        factors as e^{i * position * frequency}.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)  # (seq, d_head/2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to Q or K tensor.

    Interprets pairs of dimensions as complex numbers, multiplies by the
    precomputed rotation factors, then converts back to real.

    Args:
        x: (batch, n_heads, seq_len, d_head) — Q or K tensor
        freqs: (max_seq_len, d_head // 2) — precomputed rotation factors

    Returns:
        Rotated tensor with same shape as x.
    """
    B, H, S, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, S, D // 2, 2))
    freqs = freqs[:S].unsqueeze(0).unsqueeze(0)  # (1, 1, S, d_head/2)
    x_rotated = torch.view_as_real(x_complex * freqs).reshape(B, H, S, D)
    return x_rotated.type_as(x)


# ═══════════════════════════════════════════════════════════════════════════════
# SwiGLU — the standard FFN for modern transformers
# ═══════════════════════════════════════════════════════════════════════════════

class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network (Shazeer 2020).

    FFN(x) = W2 · (SiLU(W_gate · x) ⊙ W1 · x)

    The gated variant outperforms standard GELU FFN at equal parameter count.
    The intermediate dimension is 8/3 × d_model (rounded to multiple of 64 for
    GPU efficiency), which gives roughly the same param count as 4 × d_model
    in a standard 2-layer FFN.

    Args:
        d_model: Model dimension
        d_intermediate: Override intermediate size (default: auto-computed)
    """

    def __init__(self, d_model: int, d_intermediate: int | None = None):
        super().__init__()
        if d_intermediate is None:
            d_intermediate = int(8 / 3 * d_model)
            d_intermediate = ((d_intermediate + 63) // 64) * 64  # round to 64
        self.w1 = nn.Linear(d_model, d_intermediate, bias=False)
        self.w_gate = nn.Linear(d_model, d_intermediate, bias=False)
        self.w2 = nn.Linear(d_intermediate, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))


# ═══════════════════════════════════════════════════════════════════════════════
# Perturbative Correction — the ORN-specific context-gated residual
# ═══════════════════════════════════════════════════════════════════════════════

class PerturbativeCorrection(nn.Module):
    """
    Context-gated perturbative correction for ORN blocks.

    The core ORN insight: SharedM provides the coupling *rule*, but individual
    instances may need small corrections. Rather than per-layer W_Q/W_K (which
    stores a full copy of the rule at each layer), we use a small gated MLP that
    reads the residual stream and applies a correction to the value stream.

    The gate is initialised to sigmoid(-3) ≈ 0.05, so corrections start near
    zero and grow only where needed. The correction output layer is zero-init'd,
    making the initial model equivalent to a pure SharedM model.

    Empirical finding: gate magnitudes are larger for underserved modalities
    (env sounds: 0.209, music: 0.052), confirming the corrections act as a
    modality-specific adaptation mechanism.

    Args:
        d_model: Model dimension
        d_corr: Correction bottleneck dimension (default 64)
        use_silu: Use SiLU instead of GELU (V2 default)
    """

    def __init__(self, d_model: int, d_corr: int = 64, use_silu: bool = True):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.gate = nn.Linear(d_model, 1, bias=True)

        act = nn.SiLU() if use_silu else nn.GELU()
        self.correction = nn.Sequential(
            nn.Linear(d_model, d_corr, bias=False),
            act,
            nn.Linear(d_corr, d_model, bias=False),
        )

        # Critical init: gate starts near-closed, correction output is zero
        nn.init.constant_(self.gate.bias, -3.0)
        nn.init.zeros_(self.correction[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        g = torch.sigmoid(self.gate(h))
        delta = self.correction(h)
        return x + g * delta

    def gate_magnitude(self, x: torch.Tensor) -> float:
        """Return mean gate activation — useful for diagnostics."""
        with torch.no_grad():
            h = self.norm(x)
            g = torch.sigmoid(self.gate(h))
            return g.mean().item()


# ═══════════════════════════════════════════════════════════════════════════════
# Causal mask utility
# ═══════════════════════════════════════════════════════════════════════════════

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal attention mask: 0 where allowed, -inf where blocked.

    Returns shape (1, 1, S, S) for broadcasting over (batch, heads, S, S).
    """
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask.masked_fill_(
        torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return mask.unsqueeze(0).unsqueeze(0)
