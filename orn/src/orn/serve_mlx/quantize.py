"""Post-training int4/int8 quantization for the MLX FullORN.

Wraps `mlx.nn.quantize` with a class predicate that selects only the
bandwidth-bound modules:

  * shared_ffn.{w_gate, w_up, w_down}   — ~80% of inference cost
  * blocks.*.{Wk_proj, Wv, Wo}          — per-layer attention projections
  * tok_emb (also lm_head via tying)    — large (50304 x 2048)

Excluded (and why):

  * A, B (shared M)            — raw mx.array, not Linear; touching them
                                  forces every layer's K and Q through
                                  quantization noise. ~16 MB total.
  * BWk_fused                  — raw tensor; small (~1M params/layer); we
                                  don't quantize it because it's not a
                                  Linear and rebuilding the fuse against
                                  a quantized Wk_proj would compose two
                                  quant noises. Worth at most ~32 MB.
  * gate (1-D out)             — output_dims=1 trivially small; the bias
                                  is a scalar.
  * corr.0, corr.2             — d_corr=64 hidden, ~256K params each;
                                  quantization gain dwarfed by error.
  * RMSNorm.weight             — raw mx.array, not Linear; small.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .orn_mlx import FullORN_MLX


# Module-path substrings for things we want to quantize. Path strings
# are dotted: e.g. "blocks.0.Wk_proj", "shared_ffn.w_gate", "tok_emb".
_INCLUDE_HINTS = (
    "shared_ffn.w_gate",
    "shared_ffn.w_up",
    "shared_ffn.w_down",
    ".Wk_proj",
    ".Wv",
    ".Wo",
    "tok_emb",
)

# Substrings which, if found in a module path, exclude it from quant
# even when the include set would otherwise pick it up.
_EXCLUDE_HINTS = (
    ".gate",       # the (d,1) Linear that emits a sigmoid scalar gate
    ".corr.",      # correction MLP (small, high-leverage)
    "ln_",         # RMSNorm.weight is mx.array, not Linear, but be safe
)


def _quant_predicate(path: str, module: nn.Module):
    """class_predicate for nn.quantize. Returns True if the module is a
    Linear/Embedding we want quantized; False otherwise.

    `path` is the dotted module path. `module` is the module instance.
    """
    if not hasattr(module, "to_quantized"):
        return False
    for ex in _EXCLUDE_HINTS:
        if ex in path:
            return False
    for inc in _INCLUDE_HINTS:
        if inc in path:
            return True
    return False


def _walk_modules(model: nn.Module, prefix: str = ""):
    """Yield (path, module) for every submodule. nn.Module.named_modules
    exists in mlx but its traversal of list-children differs slightly from
    PyTorch — this helper mirrors what nn.quantize uses internally."""
    yield prefix.rstrip("."), model
    for name, child in model.children().items():
        sub_prefix = f"{prefix}{name}"
        if isinstance(child, list):
            for i, c in enumerate(child):
                if isinstance(c, nn.Module):
                    yield from _walk_modules(c, prefix=f"{sub_prefix}.{i}.")
        elif isinstance(child, nn.Module):
            yield from _walk_modules(child, prefix=f"{sub_prefix}.")


def _module_param_bytes(m: nn.Module) -> int:
    total = 0
    for v in m.parameters().values() if isinstance(m.parameters(), dict) else []:
        if isinstance(v, mx.array):
            total += v.nbytes
    return total


def _count_params_bytes(model: nn.Module) -> tuple[int, int]:
    """Count (params, bytes) over the full parameter tree."""
    n_params = 0
    n_bytes = 0

    def _walk(t):
        nonlocal n_params, n_bytes
        if isinstance(t, mx.array):
            n_params += t.size
            n_bytes += t.nbytes
        elif isinstance(t, dict):
            for v in t.values():
                _walk(v)
        elif isinstance(t, (list, tuple)):
            for v in t:
                _walk(v)

    _walk(model.parameters())
    return n_params, n_bytes


def quantize_for_inference(
    model: FullORN_MLX,
    bits: int = 4,
    group_size: int = 64,
    verbose: bool = True,
) -> dict:
    """Quantize the FFN, attention projections, and embedding in-place.

    Returns a dict of stats (selected/excluded module counts, sizes).
    """
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")

    # Pre-quant accounting.
    pre_params, pre_bytes = _count_params_bytes(model)

    # Enumerate modules and decide what to quantize, for reporting.
    all_modules = list(_walk_modules(model))
    selected: list[str] = []
    excluded: list[str] = []
    for path, mod in all_modules:
        if path == "":
            continue
        if hasattr(mod, "to_quantized"):
            if _quant_predicate(path, mod):
                selected.append(f"{path}  ({type(mod).__name__})")
            else:
                excluded.append(f"{path}  ({type(mod).__name__})")

    nn.quantize(model, group_size=group_size, bits=bits,
                class_predicate=_quant_predicate)

    # Drop the fp16 BWk fuse: with Wk_proj quantized, the unfused path
    # (h @ B then Wk_proj) lets the K projection actually benefit from
    # quantization. The cost of one extra (d×d) matmul per layer is
    # ~10 ms/layer and is dwarfed by the FFN savings.
    n_unfused = 0
    for blk in model.blocks:
        if blk.BWk_fused is not None:
            blk.BWk_fused = None
            n_unfused += 1
    if verbose and n_unfused:
        print(f"[quantize] dropped fused B@Wk for {n_unfused} blocks "
              f"(K now uses quantized Wk_proj)")

    # Post-quant accounting.
    post_params, post_bytes = _count_params_bytes(model)

    if verbose:
        print(f"[quantize] bits={bits} group_size={group_size}")
        print(f"[quantize] selected ({len(selected)}):")
        for s in selected:
            print(f"  + {s}")
        print(f"[quantize] excluded from quant ({len(excluded)}):")
        for s in excluded:
            print(f"  - {s}")
        print(f"[quantize] params  pre  : {pre_params/1e6:9.1f}M  "
              f"({pre_bytes/1e6:7.1f} MB)")
        print(f"[quantize] params  post : {post_params/1e6:9.1f}M  "
              f"({post_bytes/1e6:7.1f} MB)")
        print(f"[quantize] size  ratio  : {post_bytes/max(1,pre_bytes):.2f}x  "
              f"(saved {(pre_bytes-post_bytes)/1e6:.0f} MB)")

    return {
        "bits": bits,
        "group_size": group_size,
        "selected_count": len(selected),
        "excluded_count": len(excluded),
        "selected": selected,
        "excluded": excluded,
        "params_pre": pre_params,
        "params_post": post_params,
        "bytes_pre": pre_bytes,
        "bytes_post": post_bytes,
    }
