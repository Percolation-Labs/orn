"""Verify cached generation == uncached generation for the MLX FullORN.

Builds a tiny MLX model, runs argmax decoding with and without KV cache
from the same prompt; first N tokens must match exactly.
"""
from __future__ import annotations

import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orn.serve_mlx.orn_mlx import FullORN_MLX


def _build_model(seed: int = 0) -> FullORN_MLX:
    mx.random.seed(seed)
    m = FullORN_MLX(
        d=64, n_q_heads=4, n_kv_heads=2, d_head=16,
        n_layers=3, vocab=64, seq_len=128, ffn_width_mult=2, d_corr=16,
    )
    # Random init the parameters (otherwise everything is zeros and the
    # forward is degenerate).
    def _rand_like(t):
        if isinstance(t, mx.array):
            return mx.random.normal(t.shape) * 0.02
        if isinstance(t, dict):
            return {k: _rand_like(v) for k, v in t.items()}
        if isinstance(t, list):
            return [_rand_like(v) for v in t]
        return t
    params = _rand_like(m.parameters())
    m.update(params)
    mx.eval(m.parameters())
    return m


def _decode_uncached(model: FullORN_MLX, prompt_ids: mx.array, n_new: int) -> list[int]:
    x = prompt_ids
    out: list[int] = []
    for _ in range(n_new):
        logits = model(x)
        mx.eval(logits)
        last = logits[0, -1, :]
        tok = int(mx.argmax(last).item())
        out.append(tok)
        x = mx.concatenate([x, mx.array([[tok]], dtype=x.dtype)], axis=1)
    return out


def _decode_cached(model: FullORN_MLX, prompt_ids: mx.array, n_new: int) -> list[int]:
    logits, caches = model(prompt_ids, return_cache=True)
    mx.eval(logits, caches)
    tok = int(mx.argmax(logits[0, -1, :]).item())
    out: list[int] = [tok]
    for _ in range(n_new - 1):
        x_new = mx.array([[tok]], dtype=prompt_ids.dtype)
        logits, caches = model(x_new, kv_caches=caches)
        mx.eval(logits, caches)
        tok = int(mx.argmax(logits[0, -1, :]).item())
        out.append(tok)
    return out


def _build_prompt(length: int = 32) -> mx.array:
    mx.random.seed(123)
    return mx.random.randint(0, 64, (1, length))


def test_prefill_logits_match_uncached():
    model = _build_model()
    prompt = _build_prompt()
    a = model(prompt)
    b, _ = model(prompt, return_cache=True)
    mx.eval(a, b)
    diff = float(mx.max(mx.abs(a[0, -1, :] - b[0, -1, :])).item())
    assert diff < 1e-3, f"prefill logits differ from uncached by {diff}"
    print(f"  prefill==uncached  max|delta|={diff:.2e}")


def test_cached_decode_equals_uncached():
    model = _build_model()
    prompt = _build_prompt(length=24)
    n_new = 12
    uncached = _decode_uncached(model, prompt, n_new)
    cached = _decode_cached(model, prompt, n_new)
    assert cached == uncached, (
        f"\n  uncached: {uncached}"
        f"\n  cached:   {cached}"
    )
    print(f"  cached==uncached  first {n_new} tokens: {cached}")


def test_short_prompt_long_decode():
    model = _build_model()
    prompt = _build_prompt(length=4)
    n_new = 16
    uncached = _decode_uncached(model, prompt, n_new)
    cached = _decode_cached(model, prompt, n_new)
    assert cached == uncached
    print(f"  short_prompt_long_decode  {n_new} tokens match")


if __name__ == "__main__":
    print("[test] prefill_logits_match_uncached…", flush=True)
    test_prefill_logits_match_uncached()
    print("  PASS")
    print("[test] cached_decode_equals_uncached…", flush=True)
    test_cached_decode_equals_uncached()
    print("  PASS")
    print("[test] short_prompt_long_decode…", flush=True)
    test_short_prompt_long_decode()
    print("  PASS")
    print("\n  ALL CHECKS PASSED")
