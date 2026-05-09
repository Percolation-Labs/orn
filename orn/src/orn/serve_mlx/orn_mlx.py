"""MLX port of FullORN v3.

Mirrors orn/full_orn.py one-to-one. Module attribute names match the
PyTorch state_dict keys so load_fc_checkpoint can splice weights with
no remapping table beyond a flat-to-nested traversal.

Where MLX behaves differently from PyTorch:
  - mx.fast.scaled_dot_product_attention takes K/V un-tiled and infers GQA
    from N_q != N_kv. No enable_gqa flag, no repeat_interleave.
  - mx.fast.rope (traditional=True) replaces the apply_rope() helper. It
    handles the offset for cached decode natively.
  - Modules are eager-API but evaluation is lazy: callers must `mx.eval`
    after the forward to materialise (or the next op will pull through).
"""
from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ── RMSNorm ───────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((d,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Match PyTorch: x / sqrt(mean(x^2) + eps) * weight
        # Compute the variance in fp32 to avoid overflow when the residual
        # stream is large in fp16 — even d-norms ~600 will overflow x*x.
        in_dtype = x.dtype
        x32 = x.astype(mx.float32)
        var = mx.mean(x32 * x32, axis=-1, keepdims=True)
        normed = x32 * mx.rsqrt(var + self.eps)
        return (normed.astype(in_dtype)) * self.weight


# ── Wide Shared FFN ──────────────────────────────────────────────────

class WideSwiGLU(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_down(nn.silu(self.w_gate(x)) * self.w_up(x))


# ── ORN Block ─────────────────────────────────────────────────────────

class ORNBlock(nn.Module):
    def __init__(self, d: int, n_q_heads: int, n_kv_heads: int, d_head: int,
                 rope_base: float, d_corr: int = 64):
        super().__init__()
        self.d = d
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.rope_base = rope_base
        # NB: shared_ffn is NOT stored on the block. MLX's parameters() walk
        # would double-count anything held as a Module attribute even when
        # the underlying reference is shared. Instead, the caller passes
        # shared_ffn into __call__.

        self.ln_attn = RMSNorm(d)
        self.ln_ffn = RMSNorm(d)
        self.ln_corr = RMSNorm(d)

        self.Wk_proj = nn.Linear(d, n_kv_heads * d_head, bias=False)
        self.Wv = nn.Linear(d, n_kv_heads * d_head, bias=False)
        self.Wo = nn.Linear(n_q_heads * d_head, d, bias=False)
        # Fused B @ Wk_proj.T (computed once at load time, then reused). Set
        # by load.py after weights are loaded. Shape: (d, n_kv_heads*d_head).
        self.BWk_fused = None

        self.gate = nn.Linear(d, 1, bias=True)
        # Sequential equivalent: list of layers under "corr"; load_fc_checkpoint
        # places weights under blocks.X.corr.0.weight and corr.2.weight to match
        # PyTorch's nn.Sequential indexing.
        self.corr = [
            nn.Linear(d, d_corr, bias=False),
            nn.SiLU(),
            nn.Linear(d_corr, d, bias=False),
        ]

    def _corr_forward(self, x: mx.array) -> mx.array:
        h = x
        for layer in self.corr:
            h = layer(h)
        return h

    def __call__(self, x: mx.array, A: mx.array, B: mx.array,
                 shared_ffn: "WideSwiGLU",
                 kv_cache: Optional[tuple] = None,
                 return_cache: bool = False):
        Bsz, S, D = x.shape

        h = self.ln_attn(x)

        # SharedM coupling. q = (h @ A); k = (h @ B) @ Wk_proj.T.
        # We pre-fuse the B@Wk product at load time so the per-token K path
        # is one matmul instead of two.
        q = (h @ A).reshape(Bsz, S, self.n_q_heads, self.d_head).transpose(0, 2, 1, 3)
        if self.BWk_fused is not None:
            k = (h @ self.BWk_fused).reshape(Bsz, S, self.n_kv_heads, self.d_head).transpose(0, 2, 1, 3)
        else:
            k_raw = h @ B
            k = self.Wk_proj(k_raw).reshape(Bsz, S, self.n_kv_heads, self.d_head).transpose(0, 2, 1, 3)
        v = self.Wv(h).reshape(Bsz, S, self.n_kv_heads, self.d_head).transpose(0, 2, 1, 3)

        cache_pos = kv_cache[0].shape[2] if kv_cache is not None else 0
        q = mx.fast.rope(q, dims=self.d_head, traditional=True,
                         base=self.rope_base, scale=1.0, offset=cache_pos)
        k = mx.fast.rope(k, dims=self.d_head, traditional=True,
                         base=self.rope_base, scale=1.0, offset=cache_pos)

        if kv_cache is not None:
            k = mx.concatenate([kv_cache[0], k], axis=2)
            v = mx.concatenate([kv_cache[1], v], axis=2)

        new_cache = (k, v) if (return_cache or kv_cache is not None) else None

        # Causal mask only on prefill; decode (S queries, T_past+S keys, S=1)
        # attends to all past+current and needs no mask.
        use_causal = (kv_cache is None) and (S > 1)
        scale = 1.0 / math.sqrt(self.d_head)
        attn_out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale,
            mask=("causal" if use_causal else None),
        )

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(Bsz, S, -1)
        attn_out = self.Wo(attn_out)
        x = x + attn_out

        x = x + shared_ffn(self.ln_ffn(x))

        h_c = self.ln_corr(x)
        x = x + mx.sigmoid(self.gate(h_c)) * self._corr_forward(h_c)

        if new_cache is not None:
            return x, new_cache
        return x


# ── Full Model ────────────────────────────────────────────────────────

class FullORN_MLX(nn.Module):
    def __init__(self, d: int, n_q_heads: int, n_kv_heads: int, d_head: int,
                 n_layers: int, vocab: int, seq_len: int = 2048,
                 ffn_width_mult: Optional[int] = None, d_corr: int = 64,
                 rope_base: float = 10000.0):
        super().__init__()
        self.d = d
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.vocab = vocab
        self.rope_base = rope_base

        assert d == n_q_heads * d_head
        assert n_q_heads % n_kv_heads == 0

        # Shared coupling (top-level params; per-block A/B fields in state
        # dict are redundant copies and ignored by the loader).
        self.A = mx.zeros((d, d))
        self.B = mx.zeros((d, d))

        std_d_ff = int(8 / 3 * d)
        std_d_ff = ((std_d_ff + 63) // 64) * 64
        width_mult = ffn_width_mult or n_layers
        wide_d_ff = std_d_ff * width_mult
        wide_d_ff = ((wide_d_ff + 63) // 64) * 64
        self.shared_ffn = WideSwiGLU(d, wide_d_ff)

        self.tok_emb = nn.Embedding(vocab, d)

        self.blocks = [
            ORNBlock(d, n_q_heads, n_kv_heads, d_head,
                     rope_base=rope_base, d_corr=d_corr)
            for _ in range(n_layers)
        ]

        self.ln_f = RMSNorm(d)
        # Weight-tied head: we don't allocate; we matmul against tok_emb.weight
        # in __call__ to match PyTorch's `head.weight = tok_emb.weight`. The
        # state dict will have head.weight; the loader overwrites tok_emb if
        # we want to honour any post-load drift, but they should be identical.

    def __call__(self, tokens: mx.array,
                 kv_caches: Optional[list] = None,
                 return_cache: bool = False):
        Bsz, S = tokens.shape
        h = self.tok_emb(tokens)

        use_cache = return_cache or (kv_caches is not None)
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
        new_caches = []

        for blk, cache in zip(self.blocks, kv_caches):
            if use_cache:
                h, new_c = blk(h, self.A, self.B, self.shared_ffn,
                               kv_cache=cache, return_cache=True)
                new_caches.append(new_c)
            else:
                h = blk(h, self.A, self.B, self.shared_ffn)

        h = self.ln_f(h)
        # Tied head. After nn.quantize the embedding becomes a
        # QuantizedEmbedding whose .weight is a packed int representation,
        # so the fp16 path (h @ weight.T) breaks. QuantizedEmbedding
        # exposes as_linear() exactly for this tied-weight case.
        if hasattr(self.tok_emb, "as_linear"):
            logits = self.tok_emb.as_linear(h)
        else:
            logits = h @ self.tok_emb.weight.T

        if use_cache:
            return logits, new_caches
        return logits

    def summary(self) -> None:
        # Param count via leaves of the parameter tree.
        def _count(tree):
            if isinstance(tree, mx.array):
                return tree.size
            if isinstance(tree, dict):
                return sum(_count(v) for v in tree.values())
            if isinstance(tree, (list, tuple)):
                return sum(_count(v) for v in tree)
            return 0
        total = _count(self.parameters())
        # Subtract per-block redundant A/B/shared_ffn copies if present (we
        # don't allocate them in MLX so this should already be correct).
        d_ff = self.shared_ffn.w_gate.weight.shape[0]
        std_d_ff = int(8 / 3 * self.d)
        std_d_ff = ((std_d_ff + 63) // 64) * 64
        # Best dtype guess
        dtype = self.A.dtype
        print("FullORN_MLX Summary")
        print(f"  d={self.d}, L={self.n_layers}")
        print(f"  Wide FFN: d_ff={d_ff} ({d_ff // std_d_ff}x wider)")
        print(f"  Total params: {total:,}")
        print(f"  dtype: {dtype}, device: {mx.default_device()}")


# ── Configs ───────────────────────────────────────────────────────────

def orn_v3_from_config(mc: dict, rope_base: float = 10000.0) -> FullORN_MLX:
    """Build an MLX FullORN from a checkpoint's `config['model']` dict."""
    return FullORN_MLX(
        d=mc["d"], n_q_heads=mc["n_q_heads"], n_kv_heads=mc["n_kv_heads"],
        d_head=mc["d_head"], n_layers=mc["n_layers"], vocab=mc["vocab"],
        seq_len=mc["seq_len"],
        ffn_width_mult=mc.get("ffn_width_mult"),
        d_corr=mc.get("d_corr", 64),
        rope_base=rope_base,
    )
