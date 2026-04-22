"""Model evaluation: held-out perplexity + HF lm-eval tasks."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


@torch.no_grad()
def eval_perplexity(model: torch.nn.Module, batches: Iterable, device: str | torch.device = "cpu") -> dict:
    """Compute token-averaged cross-entropy + perplexity over `batches`.

    Each batch must yield (x, targets) with shape (B, S). Works for any model
    whose forward(x, targets=...) returns (logits, loss).
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, targets in batches:
        x = x.to(device); targets = targets.to(device)
        _, loss = model(x, targets=targets)
        n = targets.numel()
        total_loss += loss.item() * n
        total_tokens += n
    ce = total_loss / max(total_tokens, 1)
    return {"loss": ce, "perplexity": float(torch.tensor(ce).exp()), "tokens": total_tokens}


def eval_benchmarks(model, tokenizer, tasks: list[str] | None = None) -> dict:
    """HF lm-eval harness wrapper. Requires `pip install lm-eval`.

    Default tasks: hellaswag, piqa, arc_easy — the three we quote vs Pythia.
    """
    tasks = tasks or ["hellaswag", "piqa", "arc_easy"]
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM  # noqa: F401
    except ImportError as e:
        raise ImportError("pip install lm-eval  (or `pip install -e '.[eval]'`)") from e

    # ORN models aren't HF AutoModels; we evaluate via a thin HFLM-shaped adapter.
    from orn.eval._hflm_adapter import ORNHFLM
    lm = ORNHFLM(model=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(model=lm, tasks=tasks)
    return results.get("results", results)
