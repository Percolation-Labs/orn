"""Device selection — CUDA > MPS > CPU."""
from __future__ import annotations

import torch


def pick_device(preferred: str | None = None) -> torch.device:
    """Pick a device. Explicit preferred value wins; otherwise auto-detect."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
