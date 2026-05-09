"""Load a PyTorch ORN .pt checkpoint into the MLX model.

Strategy:
  1. torch.load on CPU
  2. Walk the flat state_dict and stuff each tensor into the MLX module's
     parameter tree by attribute path. Keys like
       blocks.0.shared_ffn.w_gate.weight
     are redundant per-block copies of the top-level shared_ffn — skip them.
  3. Cast to fp16 at load time.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .orn_mlx import FullORN_MLX, orn_v3_from_config


DEFAULT_CKPT = (
    "/Users/sirsh/.cache/huggingface/hub/models--mr-saoirse--orn-v3-3b-fc-sft/"
    "snapshots/44755f13d773ab208e3e0af85f9a0e123af97899/v3_3B_fc.pt"
)


def _is_redundant_block_key(key: str) -> bool:
    """Per-block A/B/shared_ffn fields are redundant copies of the top-level
    ones (PyTorch's nn.Parameter sharing was serialised naively)."""
    if not key.startswith("blocks."):
        return False
    parts = key.split(".", 2)
    # parts: ["blocks", "<n>", "<rest>"]
    if len(parts) < 3:
        return False
    rest = parts[2]
    return (
        rest in ("A", "B")
        or rest.startswith("shared_ffn.")
    )


def _set_by_path(obj, path: list[str], value):
    """Set attribute path on a Module tree. Handles list indices for blocks
    and the corr Sequential."""
    cur = obj
    for i, p in enumerate(path[:-1]):
        if p.isdigit():
            cur = cur[int(p)]
        else:
            cur = getattr(cur, p)
    last = path[-1]
    if last.isdigit():
        cur[int(last)] = value
    else:
        setattr(cur, last, value)


def load_fc_checkpoint(
    ckpt_path: Optional[str] = None,
    dtype: str = "fp16",
) -> tuple[FullORN_MLX, dict]:
    """Load the ORN FC-SFT checkpoint into an MLX FullORN.

    Returns (model, model_config_dict).
    """
    import torch  # local import; only needed at load time

    if ckpt_path is None:
        ckpt_path = DEFAULT_CKPT
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    print(f"[mlx-load] {ckpt_path}", flush=True)
    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mc = cfg["model"]

    # Honour long-context theta if present in config (matches what
    # train_long_context.rebuild_rope_freqs would have used).
    rope_base = 10000.0
    lc = cfg.get("long_context")
    if isinstance(lc, dict) and "theta_new" in lc:
        rope_base = float(lc["theta_new"])
        print(f"[mlx-load] using long-context rope theta={rope_base:.0f}",
              flush=True)

    model = orn_v3_from_config(mc, rope_base=rope_base)

    target_dtype = {"fp16": mx.float16, "fp32": mx.float32,
                    "bf16": mx.bfloat16}[dtype]

    sd = ckpt["model"]
    n_loaded = 0
    n_skipped = 0
    skipped_keys: list[str] = []
    for key, t in sd.items():
        if _is_redundant_block_key(key):
            n_skipped += 1
            continue
        # Skip head.weight — we use weight-tying via tok_emb.weight.T at forward
        # time and don't allocate a separate head module.
        if key == "head.weight":
            n_skipped += 1
            continue
        # Convert torch.Tensor -> mx.array, cast to target dtype.
        arr = mx.array(t.detach().to(torch.float32).numpy()).astype(target_dtype)
        path = key.split(".")
        # PyTorch's nn.Sequential names children "0", "1", "2" — that maps to
        # our list-based corr (so blocks.X.corr.0.weight -> corr[0].weight).
        try:
            _set_by_path(model, path, arr)
            n_loaded += 1
        except Exception as e:
            print(f"[mlx-load] WARN failed to set {key!r}: {e}", flush=True)
            skipped_keys.append(key)

    # Tie head to tok_emb (no separate head weight in our MLX model).
    # head.weight in the state dict is identical to tok_emb.weight under
    # weight tying — the loader has already written tok_emb from
    # tok_emb.weight; head.weight overwrites it with the same values, then
    # falls off the end (no head module). Either way, tok_emb has the right
    # values.

    # Pre-fuse B @ Wk_proj.T for each block (saves one matmul per layer per
    # token during decode). B is (d, d), Wk_proj.weight is (n_kv*d_head, d) so
    # MLX's nn.Linear computes x @ W.T. The fused operation we want is
    # k = (h @ B) @ Wk_proj.T = h @ (B @ Wk_proj.T).
    A_global = model.A
    B_global = model.B
    for blk in model.blocks:
        wk = blk.Wk_proj.weight  # (n_kv*d_head, d)
        bwk = (B_global @ wk.T).astype(target_dtype)  # (d, n_kv*d_head)
        blk.BWk_fused = bwk

    # Force materialisation
    mx.eval(model.parameters())

    elapsed = time.time() - t0
    n_params = 0
    def _count(t):
        nonlocal n_params
        if isinstance(t, mx.array):
            n_params += t.size
        elif isinstance(t, dict):
            for v in t.values():
                _count(v)
        elif isinstance(t, (list, tuple)):
            for v in t:
                _count(v)
    _count(model.parameters())

    print(f"[mlx-load] loaded {n_loaded} tensors, skipped {n_skipped} "
          f"(redundant per-block copies)  in {elapsed:.1f}s", flush=True)
    print(f"[mlx-load] total params: {n_params/1e6:.1f}M  dtype={dtype}",
          flush=True)

    return model, mc
