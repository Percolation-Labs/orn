"""Prefill + cached decode loop for the MLX FullORN.

Mirrors serve/hf_backend.HFBackend.generate. Argmax at temperature=0,
top-p+temperature otherwise. Stops on EOT (50256) or the
`<|im_end|>` token sequence.
"""
from __future__ import annotations

import time
from typing import Optional

import mlx.core as mx

from .orn_mlx import FullORN_MLX


def _sample(logits: mx.array, temperature: float, top_p: float) -> int:
    # mx.argmax on fp16 is fine — no need to cast to fp32 just to tie-break.
    if temperature <= 0.01:
        return int(mx.argmax(logits).item())
    logits = logits / temperature
    if top_p < 1.0:
        # MLX has no scatter-with-sort easily; do top-p on CPU via numpy.
        import numpy as np
        l = np.array(logits.astype(mx.float32))
        idx = np.argsort(-l)
        sorted_l = l[idx]
        probs = np.exp(sorted_l - sorted_l.max())
        probs = probs / probs.sum()
        cum = np.cumsum(probs)
        cutoff = np.searchsorted(cum, top_p) + 1
        keep = idx[:cutoff]
        kp = probs[:cutoff]
        kp = kp / kp.sum()
        choice = np.random.choice(keep, p=kp)
        return int(choice)
    probs = mx.softmax(logits, axis=-1)
    return int(mx.random.categorical(mx.log(probs + 1e-30)).item())


def generate(
    model: FullORN_MLX,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eot_id: Optional[int] = None,
    im_end_seq: Optional[list[int]] = None,
    print_progress: bool = True,
) -> tuple[str, dict]:
    """Returns (decoded_text, stats) where stats includes prefill/decode timing."""
    if eot_id is None:
        eot_id = tokenizer.eot_token
    if im_end_seq is None:
        im_end_seq = tokenizer.encode_ordinary("<|im_end|>")

    ids = tokenizer.encode_ordinary(prompt)
    x = mx.array([ids], dtype=mx.int32)
    n_prefix = len(ids)

    t0 = time.time()
    logits, kv_caches = model(x, return_cache=True)
    last_logits = logits[0, -1, :]
    mx.eval(last_logits, kv_caches)  # one materialisation point
    t_prefill = time.time() - t0
    if print_progress:
        print(f"[gen] prefix={n_prefix}  prefill={t_prefill:.1f}s  cache=on", flush=True)

    generated: list[int] = []
    t_decode_start = time.time()
    for step in range(max_tokens):
        tok_id = _sample(last_logits, temperature, top_p)
        if tok_id == eot_id:
            break
        generated.append(tok_id)

        x_new = mx.array([[tok_id]], dtype=mx.int32)
        logits, kv_caches = model(x_new, kv_caches=kv_caches)
        last_logits = logits[0, -1, :]
        mx.eval(last_logits, kv_caches)

        if print_progress and (step + 1) % 20 == 0:
            rate = (step + 1) / (time.time() - t_decode_start)
            print(f"[gen] step={step+1}  {rate:.2f} tok/s (decode)", flush=True)

        if im_end_seq and len(generated) >= len(im_end_seq):
            if generated[-len(im_end_seq):] == list(im_end_seq):
                generated = generated[:-len(im_end_seq)]
                break

    n_dec = len(generated)
    t_decode = time.time() - t_decode_start
    if print_progress and n_dec > 0:
        rate = n_dec / max(1e-6, t_decode)
        print(f"[gen] done  {n_dec} tokens  {rate:.2f} tok/s decode", flush=True)
    text = tokenizer.decode(generated)

    stats = {
        "n_prefix": n_prefix,
        "n_decoded": n_dec,
        "prefill_s": t_prefill,
        "decode_s": t_decode,
        "decode_tok_s": (n_dec / t_decode) if t_decode > 0 else 0.0,
        "wall_s": t_prefill + t_decode,
    }
    return text, stats
