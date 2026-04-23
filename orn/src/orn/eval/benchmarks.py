"""Model evaluation: held-out perplexity + HF lm-eval tasks."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


# Default benchmarks quoted in the paper.
DEFAULT_TASKS = [
    "hellaswag", "piqa", "arc_easy", "arc_challenge",
    "winogrande", "boolq", "openbookqa",
]


@torch.no_grad()
def eval_perplexity(model: torch.nn.Module, batches: Iterable,
                     device: str | torch.device = "cpu") -> dict:
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


def eval_benchmarks(model, tokenizer, tasks: list[str] | None = None,
                     limit: int | None = None, batch_size: int = 4,
                     max_length: int | None = None,
                     device: str | torch.device | None = None) -> dict:
    """HF lm-eval-harness wrapper. Requires `pip install -e '.[eval]'`.

    Args:
      tasks: defaults to DEFAULT_TASKS (the seven from the paper table).
      limit: per-task example cap for fast runs. None runs the full task.
      batch_size: forwarded to the adapter (ORN adapter does one request at
                   a time today; this is a placeholder for future batching).
      max_length: context window; defaults to the model's seq_len.
    """
    tasks = tasks or DEFAULT_TASKS
    try:
        from lm_eval import evaluator  # noqa: F401
    except ImportError as e:
        raise ImportError("pip install lm-eval  (or `pip install -e '.[eval]'`)") from e

    from lm_eval import evaluator
    from orn.eval._hflm_adapter import ORNHFLM

    seq_len = max_length or getattr(getattr(model, "config", None), "seq_len", None) \
        or getattr(model, "seq_len", 2048)

    lm = ORNHFLM(model=model, tokenizer=tokenizer, batch_size=batch_size,
                  max_length=seq_len, device=device)

    res = evaluator.simple_evaluate(model=lm, tasks=tasks, limit=limit,
                                     bootstrap_iters=100)
    return res.get("results", res)
