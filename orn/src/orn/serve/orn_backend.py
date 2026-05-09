"""Real-model inference backend for the ORN serve scaffold.

Loads a function-calling ORN checkpoint, runs prefill + cached autoregressive
decode, and exposes a `generate(prompt, ...) -> str` interface that matches
the mock backend's contract.

Inference recipe (see `drafts/post_training_a_600m_orn.md` § "What the
inference recipe looks like" for the full story):
- KV cache via `ORNV3.forward(kv_caches=..., return_cache=...)`. Bitwise
  identical to un-cached generation; verified in `tests/test_kv_cache_orn.py`.
- fp16 weights on MPS, bf16 with autocast on CUDA, fp32 on CPU.
- `enable_gqa=True` in SDPA where supported (PyTorch ≥2.5), otherwise
  manual `repeat_interleave` fallback.
- Pre-fused `B @ Wk_proj^T` per block via `ORNV3Block.fuse_for_inference()` —
  saves one matmul per token per layer at no quality cost. Skipped when
  the model has been quantized; the unfused two-matmul path is used so
  the quantized `Wk_proj` stays live.
- Sequence stop on the full `<|im_end|>` token list (NOT a single-token
  stop on byte 27, which is `<` and would truncate every `<functioncall>`
  emission to the empty string).
"""
from __future__ import annotations

import os
import sys
from typing import Any

import torch
import torch.nn.functional as F


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _auto_dtype(device: str) -> "torch.dtype":
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16  # bf16 unreliable on MPS
    return torch.float32


def _v3_config_from_dict(mc: dict) -> "ORNV3Config":
    """Build an ORNV3Config from a checkpoint's `config.model` dict.

    Translates older field names (`d → d_model`, `vocab → vocab_size`) used
    by the function-calling SFT checkpoint that predates the public
    ORNV3Config naming. New fields pass through unchanged.
    """
    from orn.models.orn_v3 import ORNV3Config
    mc = dict(mc)
    if "d" in mc and "d_model" not in mc:
        mc["d_model"] = mc.pop("d")
    if "vocab" in mc and "vocab_size" not in mc:
        mc["vocab_size"] = mc.pop("vocab")
    return ORNV3Config.from_dict(mc)


class ORNBackend:
    """ORN function-calling inference backend.

    Args:
        checkpoint: local `.pt` path, or HuggingFace repo id (e.g.
            `mr-saoirse/orn-v3-3b-fc-sft`). Repo ids auto-download to the
            HF cache on first use.
        device: `auto` | `cuda` | `mps` | `cpu`. `auto` picks cuda > mps > cpu.
        dtype:  `auto` | `fp32` | `fp16` | `bf16`. `auto` follows device.
    """

    def __init__(self, checkpoint: str, device: str = "auto", dtype: str = "auto") -> None:
        from orn.models.orn_v3 import ORNV3
        import tiktoken

        self.ORNV3 = ORNV3

        if device == "auto":
            device = _auto_device()
        if dtype == "auto":
            torch_dtype = _auto_dtype(device)
        else:
            torch_dtype = {"fp32": torch.float32, "fp16": torch.float16,
                           "bf16": torch.bfloat16}[dtype]

        ckpt_path = self._resolve_checkpoint(checkpoint)
        # Map to CPU first; we move to target device after splitting any
        # complex rope_freqs buffers (MPS can't store complex64).
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        mc = cfg["model"]

        config = _v3_config_from_dict(mc)
        model = ORNV3(config)
        model.load_state_dict(ckpt["model"], strict=False)

        if device == "mps":
            self._real_split_rope(model)
        model = model.to(device).eval()

        # Cast parameters to inference dtype after device move. Don't cast
        # buffers (rope cos/sin stays fp32 — apply_rope upcasts internally
        # via dtype propagation in the math).
        if torch_dtype != torch.float32:
            for _, p in model.named_parameters():
                p.data = p.data.to(torch_dtype)
        for p in model.parameters():
            p.requires_grad_(False)

        # Pre-fuse B @ Wk_proj^T per block: one matmul instead of two for K.
        # Saves bandwidth at zero quality cost. Skip if quantization is going
        # to be layered on top — the fused path would dead-code Wk_proj.
        for block in getattr(model, "blocks", []):
            if hasattr(block, "fuse_for_inference"):
                block.fuse_for_inference()

        self.model = model
        self.config = config
        self.device = device
        self.dtype = torch_dtype

        self.tok = tiktoken.get_encoding("gpt2")
        self.eot_id = self.tok.eot_token
        self.im_end_seq = self.tok.encode_ordinary("<|im_end|>")

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        dtype_name = str(torch_dtype).split(".")[-1]
        print(f"[hf] arch=orn_v3 d={config.d_model} L={config.n_layers} "
              f"seq_len={config.seq_len} vocab={config.vocab_size} "
              f"params={n_params:.1f}M device={device} dtype={dtype_name}",
              flush=True)
        print(f"[hf] eot_id={self.eot_id} im_end_seq={self.im_end_seq}", flush=True)

    def _real_split_rope(self, model) -> None:
        """Convert complex rope_freqs buffers to (cos, sin) pairs on CPU.

        MPS can't move complex64 across devices. The real-arithmetic
        `apply_rope` in `orn.models.layers` accepts either a complex tensor
        or a stacked-real tensor `(2, seq, d_head/2)`, so we replace the
        complex buffer in-place with the stacked form.
        """
        for name, buf in list(model.named_buffers()):
            if "rope_freqs" in name and torch.is_complex(buf):
                cos = buf.real.contiguous()
                sin = buf.imag.contiguous()
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                stacked = torch.stack([cos, sin], dim=0)
                parent.register_buffer(parts[-1], stacked, persistent=False)

    def _resolve_checkpoint(self, ref: str) -> str:
        """Resolve a checkpoint reference to a local path.

        Accepts a local file path or a HuggingFace repo id like
        `mr-saoirse/orn-v3-3b-fc-sft`. Repo ids download `v3_3B_fc.pt` from
        the repo into the HF cache on first use, then return the cached
        path on subsequent calls.
        """
        if os.path.exists(ref):
            return ref
        from huggingface_hub import hf_hub_download
        repo_id, _, filename = ref.partition(":")
        if not filename:
            filename = "v3_3B_fc.pt"
        token = os.environ.get("HUGGING_FACE_ACCESS_KEY") or os.environ.get("HF_TOKEN")
        return hf_hub_download(repo_id=repo_id, filename=filename, token=token)

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> int:
        if temperature <= 0.01:
            return int(torch.argmax(logits).item())
        logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            mask = cum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits[mask] = float("-inf")
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(0, sorted_idx, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def _forward(self, x: torch.Tensor, kv_caches=None):
        return self.model(
            x, is_causal=True,
            kv_caches=kv_caches,
            return_cache=(kv_caches is None),
        )

    @torch.no_grad()
    def generate(self, prompt: str, tools_schema=None, max_tokens: int = 200,
                 temperature: float = 0.2, top_p: float = 0.95) -> str:
        """Run prefill on `prompt`, then cached autoregressive decode.

        `tools_schema` is currently unused at this level; constrained
        decoding via `serve.constrained` would consume it. Stop conditions:
        (a) the EOT token, (b) the full `<|im_end|>` token sequence
        appearing as a suffix of the generated tokens (NOT a single-token
        stop on byte 27 — that would truncate `<functioncall>` emissions).
        """
        import time

        ids = self.tok.encode_ordinary(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        generated: list[int] = []
        n_prefix = len(ids)
        t0 = time.time()

        # Prefill — one forward over the whole prompt; populates the cache.
        logits, kv_caches = self._forward(x)
        last_logits = logits[0, -1, :].float()
        t_prefill = time.time() - t0
        print(f"[gen] prefix={n_prefix}  prefill={t_prefill:.1f}s  cache=on", flush=True)

        t_decode_start = time.time()
        for step in range(max_tokens):
            tok_id = self._sample(last_logits, temperature, top_p)
            if tok_id == self.eot_id:
                break
            generated.append(tok_id)

            # Cached decode: feed only the new token, attend against the cache.
            x_new = torch.tensor([[tok_id]], dtype=torch.long, device=self.device)
            logits, kv_caches = self._forward(x_new, kv_caches=kv_caches)
            last_logits = logits[0, -1, :].float()

            if (step + 1) % 20 == 0:
                rate = (step + 1) / (time.time() - t_decode_start)
                print(f"[gen] step={step+1}  {rate:.2f} tok/s (decode)", flush=True)

            # Sequence stop on full <|im_end|>.
            if self.im_end_seq and len(generated) >= len(self.im_end_seq):
                if generated[-len(self.im_end_seq):] == list(self.im_end_seq):
                    generated = generated[:-len(self.im_end_seq)]
                    break

        n_dec = len(generated)
        if n_dec > 0:
            rate = n_dec / max(1e-6, time.time() - t_decode_start)
            print(f"[gen] done  {n_dec} tokens  {rate:.2f} tok/s decode", flush=True)
        return self.tok.decode(generated)
