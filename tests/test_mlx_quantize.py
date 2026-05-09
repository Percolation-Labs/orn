"""Sanity tests for quantize.py on a tiny random model.

Verifies:
  * the predicate selects exactly the modules we expect
  * nn.quantize replaces selected Linear layers with QuantizedLinear
  * the model still forward-passes and produces finite logits
  * cached==uncached argmax decoding still holds after quantization
    (the quantization noise is the same on both paths since it's
    weight-only)
"""
from __future__ import annotations

import os
import sys

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orn.serve_mlx.orn_mlx import FullORN_MLX
from orn.serve_mlx.quantize import _quant_predicate, _walk_modules, quantize_for_inference


def _build_model(seed: int = 0) -> FullORN_MLX:
    mx.random.seed(seed)
    m = FullORN_MLX(
        d=64, n_q_heads=4, n_kv_heads=2, d_head=16,
        n_layers=3, vocab=64, seq_len=128, ffn_width_mult=2, d_corr=16,
    )

    def _rand(t):
        if isinstance(t, mx.array):
            return mx.random.normal(t.shape) * 0.02
        if isinstance(t, dict):
            return {k: _rand(v) for k, v in t.items()}
        if isinstance(t, list):
            return [_rand(v) for v in t]
        return t

    m.update(_rand(m.parameters()))
    mx.eval(m.parameters())
    return m


def test_predicate_selection():
    m = _build_model()
    selected: list[str] = []
    excluded: list[str] = []
    for path, mod in _walk_modules(m):
        if path == "" or not hasattr(mod, "to_quantized"):
            continue
        (selected if _quant_predicate(path, mod) else excluded).append(path)

    expected_selected = {
        "shared_ffn.w_gate", "shared_ffn.w_up", "shared_ffn.w_down",
        "tok_emb",
        "blocks.0.Wk_proj", "blocks.0.Wv", "blocks.0.Wo",
        "blocks.1.Wk_proj", "blocks.1.Wv", "blocks.1.Wo",
        "blocks.2.Wk_proj", "blocks.2.Wv", "blocks.2.Wo",
    }
    expected_excluded = {
        "blocks.0.gate", "blocks.0.corr.0", "blocks.0.corr.2",
        "blocks.1.gate", "blocks.1.corr.0", "blocks.1.corr.2",
        "blocks.2.gate", "blocks.2.corr.0", "blocks.2.corr.2",
    }
    assert set(selected) == expected_selected, (
        f"selected mismatch: missing={expected_selected-set(selected)} "
        f"extra={set(selected)-expected_selected}"
    )
    assert set(excluded) == expected_excluded, (
        f"excluded mismatch: missing={expected_excluded-set(excluded)} "
        f"extra={set(excluded)-expected_excluded}"
    )
    print(f"  predicate selects {len(selected)} modules, excludes "
          f"{len(excluded)} modules — exactly as designed")


def test_quantize_in_place():
    m = _build_model()
    x = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
    logits_pre = m(x)
    mx.eval(logits_pre)

    stats = quantize_for_inference(m, bits=4, group_size=32, verbose=False)
    mx.eval(m.parameters())
    assert stats["selected_count"] == 13
    assert stats["excluded_count"] == 9

    # Type swap check: the FFN gates should now be QuantizedLinear.
    assert type(m.shared_ffn.w_gate).__name__ == "QuantizedLinear"
    assert type(m.shared_ffn.w_down).__name__ == "QuantizedLinear"
    # gate (the per-block sigmoid scalar gate) must still be Linear.
    assert type(m.blocks[0].gate).__name__ == "Linear"
    # corr stays Linear.
    assert type(m.blocks[0].corr[0]).__name__ == "Linear"
    # tok_emb becomes QuantizedEmbedding.
    assert "Quantized" in type(m.tok_emb).__name__

    logits_post = m(x)
    mx.eval(logits_post)
    finite = mx.all(mx.isfinite(logits_post)).item()
    assert finite, "post-quant logits contain non-finite values"
    print(f"  quantized {stats['selected_count']} modules; "
          f"post-quant forward finite. "
          f"size {stats['bytes_pre']/1e6:.2f}→{stats['bytes_post']/1e6:.2f} MB")


def test_cached_equals_uncached_after_quant():
    """Weight-only quant: introduces error, but the same error on both
    cached and uncached paths. They must agree on greedy decode."""
    m = _build_model()
    quantize_for_inference(m, bits=4, group_size=32, verbose=False)
    mx.eval(m.parameters())

    mx.random.seed(123)
    prompt = mx.random.randint(0, 64, (1, 16))
    n_new = 8

    def _decode_uncached():
        x = prompt
        out = []
        for _ in range(n_new):
            logits = m(x)
            mx.eval(logits)
            tok = int(mx.argmax(logits[0, -1, :]).item())
            out.append(tok)
            x = mx.concatenate([x, mx.array([[tok]], dtype=x.dtype)], axis=1)
        return out

    def _decode_cached():
        logits, caches = m(prompt, return_cache=True)
        mx.eval(logits, caches)
        tok = int(mx.argmax(logits[0, -1, :]).item())
        out = [tok]
        for _ in range(n_new - 1):
            x_new = mx.array([[tok]], dtype=prompt.dtype)
            logits, caches = m(x_new, kv_caches=caches)
            mx.eval(logits, caches)
            tok = int(mx.argmax(logits[0, -1, :]).item())
            out.append(tok)
        return out

    a = _decode_uncached()
    b = _decode_cached()
    assert a == b, f"\n  uncached: {a}\n  cached:   {b}"
    print(f"  cached==uncached after int4 quant: {a}")


if __name__ == "__main__":
    print("[test] predicate_selection…", flush=True)
    test_predicate_selection()
    print("  PASS")
    print("[test] quantize_in_place…", flush=True)
    test_quantize_in_place()
    print("  PASS")
    print("[test] cached_equals_uncached_after_quant…", flush=True)
    test_cached_equals_uncached_after_quant()
    print("  PASS")
    print("\n  ALL CHECKS PASSED")
