"""Thin adapter: make an ORN model look like an lm-eval `HFLM`.

Only implements what the benchmarks we care about (HellaSwag / PIQA / ARC)
actually call: loglikelihood of continuation given prompt, and generation.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class ORNHFLM:
    def __init__(self, model, tokenizer, batch_size: int = 8, max_length: int = 2048):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = next(model.parameters()).device

    @property
    def eot_token_id(self):
        return getattr(self.tokenizer, "eos_token_id", 0)

    def tok_encode(self, s: str) -> list[int]:
        if hasattr(self.tokenizer, "encode"):
            return self.tokenizer.encode(s)
        return self.tokenizer(s)["input_ids"]

    def tok_decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    @torch.no_grad()
    def loglikelihood(self, requests: list[tuple[str, str]]) -> list[tuple[float, bool]]:
        """For each (context, continuation) pair, return (logprob_sum, is_greedy)."""
        out = []
        for ctx, cont in requests:
            ctx_ids = self.tok_encode(ctx)
            cont_ids = self.tok_encode(cont)
            seq = torch.tensor(ctx_ids + cont_ids, device=self._device).unsqueeze(0)
            seq = seq[:, -self.max_length:]
            logits = self.model(seq) if not isinstance(self.model(seq), tuple) else self.model(seq)[0]
            logprobs = F.log_softmax(logits[0], dim=-1)
            # continuation logprobs align with logits at positions [len(ctx)-1 .. len(ctx)+len(cont)-2]
            start = len(ctx_ids) - 1
            cont_logprobs = logprobs[start:start + len(cont_ids)].gather(
                -1, torch.tensor(cont_ids, device=self._device).unsqueeze(-1)
            ).squeeze(-1)
            greedy = (logits[0, start:start + len(cont_ids)].argmax(-1) ==
                      torch.tensor(cont_ids, device=self._device)).all().item()
            out.append((cont_logprobs.sum().item(), bool(greedy)))
        return out
