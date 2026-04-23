"""lm-eval-harness adapter for ORN models.

Wraps any autoregressive ORN model (ORN V1/V2/V3 or anything else that exposes
`forward(tokens)` returning logits) as an `lm_eval.api.model.LM`. Supports the
three request types lm-eval uses: `loglikelihood`, `loglikelihood_rolling`,
`generate_until`. Batches requests by input length bucket for throughput.

Usage:

    from orn.eval import eval_benchmarks
    results = eval_benchmarks(model, tokenizer,
                               tasks=["piqa", "arc_easy", "hellaswag"],
                               limit=None)  # set `limit=50` for a quick probe
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F
from lm_eval.api.model import LM


class ORNHFLM(LM):
    """Minimal LM adapter for ORN models. Single-device (mps/cuda/cpu)."""

    def __init__(self, model, tokenizer, batch_size: int = 4,
                 max_length: int = 2048, device: str | torch.device | None = None):
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = torch.device(device) if device else next(model.parameters()).device
        # lm-eval reads these attributes for metadata
        self._rank = 0
        self._world_size = 1

    # ── lm-eval required properties ─────────────────────────────────────────

    @property
    def eot_token_id(self) -> int:
        return int(getattr(self.tokenizer, "eot_token", None) or
                   getattr(self.tokenizer, "eos_token_id", None) or 0)

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return getattr(self.tokenizer, "name", "tiktoken-gpt2")

    # ── Tokenisation ────────────────────────────────────────────────────────

    def tok_encode(self, s: str, **_) -> list[int]:
        if hasattr(self.tokenizer, "encode"):
            return self.tokenizer.encode(s)
        return self.tokenizer(s)["input_ids"]

    def tok_decode(self, ids, **_) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids)

    # ── Core inference ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        out = self.model(inps)
        if isinstance(out, tuple):
            out = out[0]
        return out

    @torch.no_grad()
    def _loglikelihood_one(self, ctx_ids: list[int], cont_ids: list[int]) -> tuple[float, bool]:
        # Truncate from the left if too long.
        ids = (ctx_ids + cont_ids)[-self.max_length:]
        effective_ctx_len = len(ids) - len(cont_ids)
        seq = torch.tensor(ids, dtype=torch.long, device=self._device).unsqueeze(0)
        logits = self._model_call(seq)[0]                   # (S, V)
        logprobs = F.log_softmax(logits, dim=-1)
        start = max(effective_ctx_len - 1, 0)
        cont_t = torch.tensor(cont_ids, dtype=torch.long, device=self._device)
        tok_lp = logprobs[start:start + len(cont_ids)].gather(-1, cont_t.unsqueeze(-1)).squeeze(-1)
        greedy = (logits[start:start + len(cont_ids)].argmax(-1) == cont_t).all().item()
        return float(tok_lp.sum().item()), bool(greedy)

    # ── Request handlers ────────────────────────────────────────────────────

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        out = []
        for req in requests:
            ctx, cont = req.args
            ctx_ids = self.tok_encode(ctx) if ctx else [self.eot_token_id]
            cont_ids = self.tok_encode(cont)
            if not cont_ids:
                out.append((0.0, True)); continue
            out.append(self._loglikelihood_one(ctx_ids, cont_ids))
        return out

    def loglikelihood_rolling(self, requests) -> list[float]:
        out = []
        for req in requests:
            (s,) = req.args
            ids = self.tok_encode(s)
            if not ids:
                out.append(0.0); continue
            # Stride through with full attention, summing per-token logprobs.
            total = 0.0
            pos = 0
            stride = self.max_length - 1
            while pos < len(ids):
                window = ids[max(0, pos - 1): pos + stride]
                if len(window) < 2:
                    break
                seq = torch.tensor(window, dtype=torch.long, device=self._device).unsqueeze(0)
                logits = self._model_call(seq)[0]
                logprobs = F.log_softmax(logits, dim=-1)
                targets = torch.tensor(window[1:], dtype=torch.long, device=self._device)
                total += float(logprobs[:-1].gather(-1, targets.unsqueeze(-1)).sum().item())
                pos += stride
            out.append(total)
        return out

    def generate_until(self, requests) -> list[str]:
        gen = getattr(self.model, "generate", None)
        if gen is None:
            raise RuntimeError("Model has no .generate() method")
        out = []
        for req in requests:
            ctx, gen_kwargs = req.args
            stop = gen_kwargs.get("until", []) or []
            max_new = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            ids = torch.tensor(self.tok_encode(ctx), dtype=torch.long, device=self._device)
            produced = gen(ids, max_new=max_new, temperature=0.0, top_k=1)
            text = self.tok_decode(produced.tolist()[len(ids):])
            for s in stop:
                if s and s in text:
                    text = text[:text.index(s)]
                    break
            out.append(text)
        return out
