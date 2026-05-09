"""KV-cache regression test for ORNV3.

Verifies that incremental cached decoding produces bitwise-identical tokens
to a naive un-cached decode loop. The cache contract relies on three
properties of the architecture (causal attention, per-token-linear K/V
projections, position-local RoPE) — any change that breaks one of these
will be caught here.

Mirrors the pattern used in the private serve/test_kv_cache.py for FullORN.
"""
from __future__ import annotations

import torch

from orn.models.orn_v3 import ORNV3, ORNV3Config


def _build_model(seed: int = 0) -> ORNV3:
    torch.manual_seed(seed)
    cfg = ORNV3Config(
        d_model=64, n_layers=3, n_q_heads=4, n_kv_heads=2, d_head=16,
        vocab_size=64, seq_len=128, ffn_width_mult=2, d_corr=16,
    )
    model = ORNV3(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def _decode_uncached(model: ORNV3, prompt_ids: torch.Tensor, n_new: int) -> list[int]:
    x = prompt_ids.clone()
    out: list[int] = []
    for _ in range(n_new):
        logits = model(x, is_causal=True)
        tok = int(logits[0, -1, :].argmax().item())
        out.append(tok)
        x = torch.cat([x, torch.tensor([[tok]], device=x.device, dtype=x.dtype)], dim=1)
    return out


@torch.no_grad()
def _decode_cached(model: ORNV3, prompt_ids: torch.Tensor, n_new: int) -> list[int]:
    logits, caches = model(prompt_ids, is_causal=True, return_cache=True)
    tok = int(logits[0, -1, :].argmax().item())
    out: list[int] = [tok]
    for _ in range(n_new - 1):
        x_new = torch.tensor([[tok]], device=prompt_ids.device, dtype=prompt_ids.dtype)
        logits, caches = model(x_new, is_causal=True, kv_caches=caches)
        tok = int(logits[0, -1, :].argmax().item())
        out.append(tok)
    return out


def _build_prompt(length: int = 24) -> torch.Tensor:
    torch.manual_seed(123)
    return torch.randint(0, 64, (1, length))


def test_prefill_logits_match_uncached():
    """Prefill (return_cache=True) must produce identical last-position logits
    to a plain forward — same computation, just stashing K/V on the way out."""
    model = _build_model()
    prompt = _build_prompt()
    a = model(prompt, is_causal=True)[0, -1, :].float()
    b, _ = model(prompt, is_causal=True, return_cache=True)
    b = b[0, -1, :].float()
    diff = (a - b).abs().max().item()
    assert diff < 1e-5, f"prefill logits differ from uncached by {diff}"


def test_cached_decode_equals_uncached():
    """First N argmax tokens must match exactly between cached and un-cached
    decode. Logit drift at later positions is bounded by fp tolerance."""
    model = _build_model()
    prompt = _build_prompt(length=24)
    n_new = 12
    uncached = _decode_uncached(model, prompt, n_new)
    cached = _decode_cached(model, prompt, n_new)
    assert cached == uncached, (
        f"\n  uncached: {uncached}"
        f"\n  cached:   {cached}"
    )


def test_short_prompt_long_decode():
    """Edge case: tiny prompt, longer decode — exercises cache growth."""
    model = _build_model()
    prompt = _build_prompt(length=4)
    n_new = 16
    uncached = _decode_uncached(model, prompt, n_new)
    cached = _decode_cached(model, prompt, n_new)
    assert cached == uncached


def test_fuse_for_inference_preserves_output():
    """fuse_for_inference() pre-computes B @ Wk_proj^T into one matmul.
    Output must be numerically identical (same math, fewer ops)."""
    model = _build_model()
    prompt = _build_prompt()
    a = model(prompt, is_causal=True)[0, -1, :].float()
    for block in model.blocks:
        block.fuse_for_inference()
    b = model(prompt, is_causal=True)[0, -1, :].float()
    diff = (a - b).abs().max().item()
    assert diff < 1e-5, f"fused K projection drifted by {diff}"


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
    print("[test] fuse_for_inference_preserves_output…", flush=True)
    test_fuse_for_inference_preserves_output()
    print("  PASS")
    print("\n  ALL CHECKS PASSED")
