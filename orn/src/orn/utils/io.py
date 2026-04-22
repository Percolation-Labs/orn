"""Checkpoint save / load helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    config: dict | None = None, extra: dict | None = None) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if config is not None:
        state["config"] = config
    if extra:
        state.update(extra)
    torch.save(state, p)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
