"""Compatibility shim for the post-training trainers.

The trainers were authored against the private research repo where the
top-level ORN class was named ``FullORN`` and constructed via a positional
helper ``build_orn(mc)``. The public package uses ``ORNV3`` and a config
object ``ORNV3Config``.

This shim re-exports the public class under the legacy name and provides
``build_orn`` / ``build_transformer`` helpers that translate the older
config-dict shape (``d`` / ``vocab``) into ``ORNV3Config`` (``d_model`` /
``vocab_size``). All five post-training scripts import from here, so the
trainer bodies can stay nearly verbatim with the private repo.
"""
from __future__ import annotations

from typing import Any

from orn.models.orn_v3 import ORNV3, ORNV3Config
from orn.models.transformer import Transformer, TransformerConfig


# Re-export under legacy names so the verbatim trainer code still finds
# `FullORN` and `VanillaTransformer`.
FullORN = ORNV3
VanillaTransformer = Transformer


def _v3_config_from_dict(mc: dict) -> ORNV3Config:
    """Translate older field names (d → d_model, vocab → vocab_size)."""
    mc = dict(mc)
    if "d" in mc and "d_model" not in mc:
        mc["d_model"] = mc.pop("d")
    if "vocab" in mc and "vocab_size" not in mc:
        mc["vocab_size"] = mc.pop("vocab")
    return ORNV3Config.from_dict(mc)


def build_orn(mc: dict) -> ORNV3:
    """Construct an ORNV3 from a legacy-shape model-config dict."""
    return ORNV3(_v3_config_from_dict(mc))


def build_transformer(mc: dict) -> Transformer:
    """Construct a vanilla Transformer baseline from a legacy-shape dict.

    Used by trainers that compare ORN against a matched-FFN-budget vanilla
    transformer baseline. The mc dict carries the same fields as for ORN;
    we drop ORN-specific entries (`ffn_width_mult`, `d_corr`) the
    Transformer config doesn't know about.
    """
    mc = dict(mc)
    if "d" in mc and "d_model" not in mc:
        mc["d_model"] = mc.pop("d")
    if "vocab" in mc and "vocab_size" not in mc:
        mc["vocab_size"] = mc.pop("vocab")
    for k in ("ffn_width_mult", "d_corr"):
        mc.pop(k, None)
    return Transformer(TransformerConfig.from_dict(mc))
