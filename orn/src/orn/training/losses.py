"""Loss adapters.

A trainer needs a uniform `loss_fn(model, x, y) -> (loss, logs)` interface so
the training loop is agnostic to model-specific auxiliary losses (CORN's
consistency loss, LORN's VQ loss). The dispatcher here picks the right adapter
based on config `loss.type`.

Adapters:
  lm_ce          — standard LM cross-entropy (ORN V1/V2/V3, Transformer, CORN baseline)
  lm_ce+lorn_vq  — LM + vq_loss_scale * (commit + codebook − entropy_weight·entropy)
  lm_ce+corn_ck  — LM + ck_weight * Chapman-Kolmogorov composability loss (CORN)
"""
from __future__ import annotations

from typing import Callable

import torch


def _lm_ce(model, x: torch.Tensor, y: torch.Tensor):
    _, loss = model(x, targets=y)
    return loss, {"ce": loss.item()}


def _lorn_vq(model, x: torch.Tensor, y: torch.Tensor):
    # LORN forward: (logits, task_loss, aux) when targets is provided
    _, task_loss, aux = model(x, targets=y)
    joint = model.joint_loss(task_loss, aux)
    return joint, {
        "ce": task_loss.item(),
        "vq": (aux["commit"] + aux["codebook"] - model.cfg.entropy_weight * aux["entropy"]).item(),
        "codes_used": int(aux["chosen"].unique().numel()),
    }


def _corn_ck(model, x: torch.Tensor, y: torch.Tensor, ck_weight: float = 0.1):
    _, loss = model(x, targets=y)
    ck = getattr(model, "ck_loss", None)
    if callable(ck):
        ck_val = ck(x)
        total = loss + ck_weight * ck_val
        return total, {"ce": loss.item(), "ck": ck_val.item()}
    return loss, {"ce": loss.item()}


_REGISTRY: dict[str, Callable] = {
    "lm_ce": _lm_ce,
    "lm_ce+lorn_vq": _lorn_vq,
    "lm_ce+corn_ck": _corn_ck,
}


def get_loss_fn(kind: str = "lm_ce", **kwargs) -> Callable:
    """Return a loss adapter. Extra kwargs are baked in as defaults."""
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown loss '{kind}'. Options: {list(_REGISTRY)}")
    fn = _REGISTRY[kind]
    if not kwargs:
        return fn
    def wrapped(model, x, y):
        return fn(model, x, y, **kwargs)
    return wrapped
