"""Reproducible key-result suite.

Each result is a self-contained module under `orn/reproduce/results/` that
exposes a `run(device=None, **kw) -> dict` function returning the result's
headline metrics. The registry maps result codes/slugs to modules and tier
metadata so they can be discovered from the CLI or from pytest.
"""
from orn.reproduce.registry import (
    REGISTRY, Result, get, list_results, resolve,
)

__all__ = ["REGISTRY", "Result", "get", "list_results", "resolve"]
