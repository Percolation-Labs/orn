"""
Model architectures.

- layers       shared building blocks (RMSNorm, SwiGLU, RoPE, PerturbativeCorrection)
- orn          ORN V1 (LayerNorm/GELU/learned-pos) and V2 (RMSNorm/SwiGLU/RoPE/GQA)
- orn_v3       ORN V3 / FullORN — V2 plus a single wide shared FFN
- corn         Composable ORN with semigroup dynamics
- lorn         Lateralised two-hemisphere model (VQREncoder + Bridge + ORNV3)
- transformer  Vanilla transformer baseline
"""

from __future__ import annotations
from orn.models.orn import ORN, ORNV2, ORNConfig
from orn.models.orn_v3 import ORNV3, ORNV3Config, orn_v3_61m, orn_v3_108m, orn_v3_350m, orn_v3_1b
from orn.models.corn import CORN, CORNConfig
from orn.models.lorn import (
    LORN, LORNConfig, VQREncoder, VQRConfig, HypernetBridge, BridgeConfig,
    lorn_v4_smoke, lorn_v4_66m,
)
from orn.models.transformer import Transformer, TransformerConfig


def build_model(arch: str, config: dict):
    """Dispatch by architecture name (used by the training config loader)."""
    arch = arch.lower()
    if arch in ("orn", "orn_v1"):
        return ORN(ORNConfig.from_dict({**config, "version": 1}))
    if arch == "orn_v2":
        return ORNV2(ORNConfig.from_dict({**config, "version": 2}))
    if arch == "orn_v3":
        return ORNV3(ORNV3Config.from_dict(config))
    if arch == "corn":
        return CORN(CORNConfig.from_dict(config))
    if arch in ("lorn", "lorn_v4"):
        return LORN(LORNConfig.from_dict(config))
    if arch == "transformer":
        return Transformer(TransformerConfig.from_dict(config))
    raise ValueError(f"Unknown architecture '{arch}'. "
                     "Valid: orn_v1, orn_v2, orn_v3, corn, lorn_v4, transformer")


__all__ = [
    "ORN", "ORNV2", "ORNConfig",
    "ORNV3", "ORNV3Config", "orn_v3_61m", "orn_v3_108m", "orn_v3_350m", "orn_v3_1b",
    "CORN", "CORNConfig",
    "LORN", "LORNConfig", "VQREncoder", "VQRConfig", "HypernetBridge", "BridgeConfig",
    "lorn_v4_smoke", "lorn_v4_66m",
    "Transformer", "TransformerConfig",
    "build_model",
]
