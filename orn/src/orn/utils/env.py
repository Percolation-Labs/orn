"""Lightweight .env loader (no python-dotenv dep)."""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> dict[str, str]:
    """Load KEY=VALUE lines from `path` into os.environ. Returns the dict loaded.

    Missing file is not an error — just returns an empty dict. Comments (#) and
    blank lines are ignored. Existing os.environ values are NOT overridden.
    """
    p = Path(path)
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
        loaded[k] = v
    return loaded


def get_env(key: str, default: str | None = None, required: bool = False) -> str | None:
    v = os.environ.get(key, default)
    if required and not v:
        raise RuntimeError(f"Environment variable {key} is required (set in .env or shell)")
    return v
