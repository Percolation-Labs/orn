from __future__ import annotations
"""
CORN — Composable Orbital Response Network
============================================

CORN extends the ORN with semigroup dynamics: the coupling strength is
parameterised by a continuous time variable t, and the model is trained to
satisfy the semigroup composition law:

    M(t1 + t2) = M(t1) ∘ M(t2)

In practice, this means applying the SharedM coupling at different t values
produces composable representations. The key breakthrough (Realisation 74):

    t = σ (coupling strength = noise level)

M's eigenspectrum IS the noise schedule:
  - High t → fast Koopman modes dominate → coarse structure (topic, syntax)
  - Low t → slow modes contribute → fine detail (specific words, agreement)
  - Iterating from high to low t IS coarse-to-fine refinement

This enables diffusion-style generation without an external noise schedule.

Training uses a dual loss:
  L = L_prediction + λ · L_semigroup

Where L_semigroup = MSE(blocks(h, t1+t2), blocks(blocks(h, t1), t2))

Key findings:
  - CORN time-scaled semigroup works: SG error drops 33% with σ₂₀=0.366
  - Prediction actively destroys composability (+57% SG error during CE training)
  - Consistency doesn't create meaning — curriculum (CE first, then CK) is needed
  - Implicit semigroup beats explicit (standard ORN: 3.802 vs explicit: 3.865)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from orn.models.layers import RMSNorm, PerturbativeCorrection


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CORNConfig:
    """Configuration for CORN."""

    d_model: int = 192
    n_layers: int = 8
    n_heads: int = 4
    d_corr: int = 48
    vocab_size: int = 50257
    seq_len: int = 128

    # Semigroup parameters
    sg_lambda: float = 0.05     # weight of semigroup composition loss
    t_min: float = 0.2          # minimum coupling strength
    t_max: float = 2.0          # maximum coupling strength
    causal: bool = True         # causal masking (False for diffusion-style)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "CORNConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════════
# CORN Block — SharedM with time-parameterised coupling
# ═══════════════════════════════════════════════════════════════════════════════

class CORNBlock(nn.Module):
    """
    One CORN layer: SharedM attention with coupling strength scaled by t.

    The coupling Q = h @ (t · A), K = h @ (t · B) means that at t=0 the
    attention is uniform (no coupling), and at large t the coupling is strong.
    This is analogous to inverse temperature in statistical mechanics.

    The t-scaling acts on M's eigenvalues: if M has eigenvalue λ, then
    t·M has eigenvalue t·λ. High t amplifies the dominant modes (coarse
    structure), low t lets all modes contribute (fine detail).
    """

    def __init__(self, d_model: int, n_heads: int, A: nn.Parameter, B: nn.Parameter,
                 d_corr: int = 48):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.A = A  # shared
        self.B = B  # shared

        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.correction = PerturbativeCorrection(d_model, d_corr)

    def forward(self, x: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        """
        Args:
            x: (B, S, D) input
            t: Coupling strength (semigroup time parameter). Default 1.0 = standard ORN.
        """
        Bsz, S, D = x.shape
        h = self.ln1(x)

        # t-scaled SharedM coupling
        q = (h @ (t * self.A)).view(Bsz, S, self.n_heads, self.d_head).transpose(1, 2)
        k = (h @ (t * self.B)).view(Bsz, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(h).view(Bsz, S, self.n_heads, self.d_head).transpose(1, 2)

        # SDPA — avoids materialising S×S attention matrix
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(Bsz, S, D)

        x = x + self.Wo(out)
        x = x + self.ff(self.ln2(x))
        x = self.correction(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Full CORN Model
# ═══════════════════════════════════════════════════════════════════════════════

class CORN(nn.Module):
    """
    Composable Orbital Response Network.

    Like ORN, but with t-parameterised coupling for semigroup dynamics.
    Supports two training modes:

    1. Standard (prediction only): Same as ORN with t=1.0
    2. Semigroup (prediction + composition): Adds L_semigroup loss that
       enforces M(t1+t2) ≈ M(t1) ∘ M(t2)

    For generation, iterates from high t (coarse) to low t (fine detail),
    using M's eigenspectrum as a natural noise schedule.
    """

    def __init__(self, config: CORNConfig | dict):
        super().__init__()
        if isinstance(config, dict):
            config = CORNConfig.from_dict(config)
        self.config = config
        d = config.d_model

        self.tok_emb = nn.Embedding(config.vocab_size, d)
        self.pos_emb = nn.Embedding(config.seq_len, d)

        # SharedM
        self.A = nn.Parameter(torch.randn(d, d) * 0.02)
        self.B = nn.Parameter(torch.randn(d, d) * 0.02)

        self.blocks = nn.ModuleList([
            CORNBlock(d, config.n_heads, self.A, self.B, config.d_corr)
            for _ in range(config.n_layers)
        ])

        self.ln_f = RMSNorm(d)
        self.head = nn.Linear(d, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None and m.bias.shape[0] != 1:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None,
                t: float = 1.0) -> tuple:
        """
        Forward pass with optional coupling strength t.

        Returns: logits or (logits, loss)
        """
        B, S = x.shape
        h = self.tok_emb(x) + self.pos_emb(torch.arange(S, device=x.device))

        for block in self.blocks:
            h = block(h, t=t)  # SDPA handles causal masking

        logits = self.head(self.ln_f(h))

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss
        return logits

    def apply_blocks(self, h: torch.Tensor, t: float) -> torch.Tensor:
        """Apply all blocks at coupling strength t (used for semigroup loss)."""
        for block in self.blocks:
            h = block(h, t=t)
        return h

    def semigroup_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute semigroup composition loss: L_sg = MSE(f(h, t1+t2), f(f(h, t1), t2))

        Samples random t1, t2 from [t_min, t_max] and checks whether applying
        blocks at t1+t2 gives the same result as applying at t1 then t2.
        """
        B, S = x.shape
        h = self.tok_emb(x) + self.pos_emb(torch.arange(S, device=x.device))

        t1 = torch.empty(1).uniform_(self.config.t_min, self.config.t_max).item()
        t2 = torch.empty(1).uniform_(self.config.t_min, self.config.t_max).item()

        # Direct: apply at t1+t2
        h_direct = self.apply_blocks(h.detach(), t1 + t2)

        # Composed: apply at t1, then at t2
        h_t1 = self.apply_blocks(h.detach(), t1)
        h_composed = self.apply_blocks(h_t1, t2)

        return F.mse_loss(h_composed, h_direct)

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
        A = self.A.detach().cpu().numpy()
        B = self.B.detach().cpu().numpy()
        M = A @ B.T
        svs = np.linalg.svd(M, compute_uv=False)
        eigvals = np.linalg.eigvals(M)
        cum_energy = np.cumsum(svs ** 2) / np.sum(svs ** 2)
        return {
            "eff_rank_90": int(np.searchsorted(cum_energy, 0.9) + 1),
            "condition_number": float(svs[0] / (svs[-1] + 1e-10)),
            "asymmetry": float(np.linalg.norm(M - M.T) / (np.linalg.norm(M) + 1e-10)),
            "top_svs": svs[:10].tolist(),
            "fraction_complex": float(np.mean(np.abs(eigvals.imag) > 1e-10)),
        }

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new: int = 100,
                 temperature: float = 0.8, top_k: int = 40,
                 t: float = 1.0) -> torch.Tensor:
        """Autoregressive generation at a fixed coupling strength."""
        self.eval()
        ids = prompt_ids.unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids
        for _ in range(max_new):
            x = ids[:, -self.config.seq_len:]
            logits = self(x, t=t)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids[0]

    @torch.no_grad()
    def generate_diffusion(self, seq_len: int, t_schedule: list[float] | None = None,
                           temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        """
        Diffusion-style generation: iterate from high t (coarse) to low t (fine).

        Starts with random token ids and refines by predicting at decreasing
        coupling strength. This uses M's eigenspectrum as a natural noise
        schedule — no external β schedule needed.

        Args:
            seq_len: Length of sequence to generate
            t_schedule: List of t values, high to low (default: [2.0, 1.5, 1.0, 0.5])
            temperature: Sampling temperature
            top_k: Top-k filtering
        """
        if t_schedule is None:
            t_schedule = [2.0, 1.5, 1.0, 0.7, 0.5]

        self.eval()
        device = self.A.device
        ids = torch.randint(0, self.config.vocab_size, (1, seq_len), device=device)

        for t in t_schedule:
            logits = self(ids, t=t)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, :, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            ids = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, seq_len)

        return ids[0]
