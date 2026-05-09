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

    @torch.no_grad()
    def fuse_for_inference(self):
        """Pre-fuse the K-projection chain `B @ Wk_proj^T` into one (d, n_kv*d_head)
        matrix. Saves one matmul per token per layer at inference. Stored as a
        non-persistent buffer so checkpoints stay portable; cleared automatically
        when blocks are quantized (the fused tensor would be dead code under
        post-training quantization, since Wk_proj's quantized representation is
        what we want to use)."""
        if hasattr(self, "_BWk"):
            return
        bwk = (self.B @ self.Wk_proj.weight.t()).contiguous()
        self.register_buffer("_BWk", bwk, persistent=False)

    def forward(self, x, is_causal: bool = True, attn_mask=None, delta=None,
                kv_cache=None, return_cache: bool = False):
        """Forward through one ORN block with optional KV cache.

        Cache contract:
        - `kv_cache=None, return_cache=False` (default): plain forward, returns `x`.
        - `kv_cache=None, return_cache=True`: prefill — runs the full forward and
          returns `(x, (K, V))` for subsequent decode steps to reuse.
        - `kv_cache=(K_past, V_past)`: decode — `x` should be the new tokens (S=1
          for one-at-a-time decode); we compute K, V for them, RoPE-rotate at
          position `T_past`, append to the cache, attend the new query against
          the full past+new K/V. Returns `(x, (K_new, V_new))` with the updated
          cache.

        The K and V cached are *pre-GQA-repeat* — the per-head broadcast for
        grouped-query attention is done at SDPA time via `enable_gqa=True`,
        keeping the cache O(n_kv_heads) rather than O(n_q_heads)."""
        B, S, D = x.shape
        h = self.ln_attn(x)

        # SharedM coupling. If fused (B @ Wk_proj^T) is precomputed, use one
        # matmul instead of two for the K projection.
        q = (h @ self.A).view(B, S, self.n_q_heads, self.d_head).transpose(1, 2)
        if hasattr(self, "_BWk"):
            k = (h @ self._BWk).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        else:
            k_raw = h @ self.B
            k = self.Wk_proj(k_raw).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.Wv(h).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)

        # RoPE — start_pos shifts when we're appending to a cache so the new
        # token gets its absolute-position rotation, not position 0.
        cache_pos = kv_cache[0].shape[2] if kv_cache is not None else 0
        rope = self.rope_freqs.to(q.device)
        q = apply_rope(q, rope, start_pos=cache_pos)
        k = apply_rope(k, rope, start_pos=cache_pos)

        # Append to KV cache pre-GQA-repeat (memory-efficient: cache stays at
        # n_kv_heads, broadcast happens inside SDPA via enable_gqa).
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        new_cache = (k, v) if (return_cache or kv_cache is not None) else None

        # SDPA — causal mask only on prefill (S queries, S keys). In decode
        # mode the new query attends to all of past+new (S queries against
        # T_past+S keys), so no causal mask is needed.
        use_causal = is_causal and (kv_cache is None)
        try:
            if attn_mask is not None:
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, enable_gqa=True)
            else:
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=use_causal, enable_gqa=True)
        except TypeError:
            # PyTorch < 2.5: no enable_gqa flag, materialize the broadcast.
            k_rep = k.repeat_interleave(self.n_rep, dim=1)
            v_rep = v.repeat_interleave(self.n_rep, dim=1)
            if attn_mask is not None:
                attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attn_mask)
            else:
                attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=use_causal)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        attn_out = self.Wo(attn_out)
        if delta is not None:
            U, V = delta  # (B, d, r), (B, r, d)
            attn_out = attn_out + torch.bmm(attn_out, torch.bmm(U, V))
        x = x + attn_out
        x = x + self.shared_ffn(self.ln_ffn(x))
        h_c = self.ln_corr(x)
        x = x + torch.sigmoid(self.gate(h_c)) * self.corr(h_c)

        if new_cache is not None:
            return x, new_cache
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
                attn_mask=None, deltas=None, kv_caches=None,
                return_cache: bool = False):
        """Forward through the model.

        Args:
            tokens: (B, S) input token ids.
            targets: optional (B, S) — if given, returns (logits, loss).
            is_causal: causal masking on attention (prefill only; decode mode
                automatically disables it because the new query already sees
                the full past via the cache).
            attn_mask: explicit attention mask, takes precedence over is_causal.
            deltas: optional list of per-layer (U, V) tuples for LORN bridge.
            kv_caches: optional list of `(K, V)` tuples per layer for
                incremental decoding. First call: pass `kv_caches=None,
                return_cache=True` and the model returns `(logits, new_caches)`.
                Subsequent calls: pass the returned list and only the new
                token(s); the model returns `(logits, updated_caches)`.
            return_cache: explicitly request a cache list be returned even on
                the first call. Implied if `kv_caches` is non-None.
        """
        B, S = tokens.shape
        h = self.tok_emb(tokens)

        if deltas is None:
            deltas = [None] * len(self.blocks)
        assert len(deltas) == len(self.blocks), \
            f"deltas length {len(deltas)} != n_layers {len(self.blocks)}"

        use_cache = return_cache or (kv_caches is not None)
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
        new_caches: list = []

        for block, d, cache in zip(self.blocks, deltas, kv_caches):
            if use_cache:
                h, new_c = block(h, is_causal=is_causal, attn_mask=attn_mask,
                                 delta=d, kv_cache=cache, return_cache=True)
                new_caches.append(new_c)
            else:
                h = block(h, is_causal=is_causal, attn_mask=attn_mask, delta=d)

        logits = self.head(self.ln_f(h))

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        if use_cache:
            return logits, new_caches
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
