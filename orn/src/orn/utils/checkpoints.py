"""Registry of published ORN checkpoints on HuggingFace.

These are the public checkpoints from the paper. They were produced by the
legacy trainer which didn't write an `arch` key, so the registry carries
arch + config overrides that are needed to rebuild the model from the
raw state_dict.

Usage:

    from orn.utils.checkpoints import known_checkpoints, fetch_checkpoint, load_known
    known_checkpoints()                        # list what's available
    path = fetch_checkpoint("orn-v3-605m")     # download, return local path
    model, meta = load_known("orn-v3-605m")    # download + build + load → ready-to-run

Or from the CLI:

    orn pull-checkpoint orn-v3-605m
    orn predict --checkpoint orn-v3-605m --prompt "The"
    orn diagnose --checkpoint orn-v3-605m
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orn.utils.env import get_env


@dataclass
class CheckpointSpec:
    name: str                 # short handle: orn-v3-605m
    repo_id: str              # HuggingFace repo: mr-saoirse/orn-v3-605m
    filename: str             # path inside the repo
    arch: str                 # orn_v1 / orn_v2 / orn_v3 / corn / lorn_v4 / transformer
    config: dict[str, Any]    # args forwarded to build_model(arch, config)
    description: str = ""
    training_tokens: int | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

_SPECS: dict[str, CheckpointSpec] = {}


def _register(s: CheckpointSpec):
    _SPECS[s.name] = s


_register(CheckpointSpec(
    name="orn-v3-605m",
    repo_id="mr-saoirse/orn-v3-605m",
    filename="checkpoints/v3_latest.pt",
    arch="orn_v3",
    config=dict(
        d_model=2048, n_layers=32, n_q_heads=32, n_kv_heads=8, d_head=64,
        d_corr=128, vocab_size=50304, seq_len=2048, ffn_width_mult=8,
    ),
    description="605M ORN V3 on FineWeb-Edu, 8x wide shared FFN. Largest published ORN.",
    training_tokens=2_000_000_000,
))

_register(CheckpointSpec(
    name="orn-v3-605m-1B",
    repo_id="mr-saoirse/orn-v3-605m",
    filename="checkpoints/v3_1B_tok.pt",
    arch="orn_v3",
    config=dict(
        d_model=2048, n_layers=32, n_q_heads=32, n_kv_heads=8, d_head=64,
        d_corr=128, vocab_size=50304, seq_len=2048, ffn_width_mult=8,
    ),
    description="Mid-training branch of orn-v3-605m at 1B tokens.",
    training_tokens=1_000_000_000,
))

_register(CheckpointSpec(
    name="orn-v2-108m",
    repo_id="mr-saoirse/orn-v2-108m",
    filename="checkpoints/orn_v2_7000M_tok.pt",
    arch="orn_v2",
    config=dict(
        d_model=576, n_layers=24, n_q_heads=9, n_kv_heads=3, d_head=64,
        d_corr=64, vocab_size=50304, seq_len=2048, version=2,
    ),
    description="108M ORN V2 on FineWeb-Edu (7B tokens). Reference baseline.",
    training_tokens=7_000_000_000,
))

_register(CheckpointSpec(
    name="orn-v2-108m-crystallised",
    repo_id="mr-saoirse/orn-v2-108m",
    filename="checkpoints/orn_v2_crystallised_step18500.pt",
    arch="orn_v2",
    config=dict(
        d_model=576, n_layers=24, n_q_heads=9, n_kv_heads=3, d_head=64,
        d_corr=64, vocab_size=50304, seq_len=2048, version=2,
    ),
    description="ORN V2 checkpoint at the crystallisation point. Use for rank / spectral diagnostics.",
))


# ═══════════════════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════════════════

def known_checkpoints() -> list[CheckpointSpec]:
    return list(_SPECS.values())


def get_spec(name: str) -> CheckpointSpec:
    if name not in _SPECS:
        raise KeyError(f"Unknown checkpoint '{name}'. Options: {list(_SPECS)}")
    return _SPECS[name]


def resolve_local(path_or_name: str | Path) -> tuple[Path, CheckpointSpec | None]:
    """Accept either a local .pt path or a known checkpoint name.

    If the argument matches a registered name, download the file (cached) and
    return its local path together with the spec. Otherwise treat as a path.
    """
    s = str(path_or_name)
    if s in _SPECS:
        return fetch_checkpoint(s), _SPECS[s]
    p = Path(s)
    if not p.exists():
        raise FileNotFoundError(
            f"'{s}' is neither a known checkpoint nor an existing file. "
            f"Known: {list(_SPECS)}"
        )
    return p, None


def fetch_checkpoint(name: str, cache_dir: str | Path | None = None) -> Path:
    """Download the named checkpoint from HuggingFace (cached) and return its path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required. Install with `pip install -e '.[data]'`."
        ) from e
    spec = get_spec(name)
    path = hf_hub_download(
        repo_id=spec.repo_id,
        filename=spec.filename,
        cache_dir=str(cache_dir) if cache_dir else None,
        token=get_env("HF_TOKEN"),  # optional; the public repos don't need it
    )
    return Path(path)


def _remap_legacy_v2_keys(state: dict) -> dict:
    """Translate the pre-refactor ORNV2 state-dict layout to the current one.

    Legacy (flat, per-block):
        blocks.N.A, blocks.N.B                                 (duplicate shared refs)
        blocks.N.ln_corr.weight
        blocks.N.gate.weight, blocks.N.gate.bias
        blocks.N.correction.0.weight, blocks.N.correction.2.weight

    Current (wrapped PerturbativeCorrection):
        (top-level A / B only)
        blocks.N.correction.norm.weight
        blocks.N.correction.gate.{weight,bias}
        blocks.N.correction.correction.{0,2}.weight
    """
    out = {}
    for k, v in state.items():
        if k.startswith("blocks."):
            parts = k.split(".", 2)        # ['blocks', '<i>', '<rest>']
            rest = parts[2] if len(parts) > 2 else ""
            if rest.startswith("ln_corr."):
                k = f"{parts[0]}.{parts[1]}.correction.norm.{rest.split('.', 1)[1]}"
            elif rest.startswith("gate."):
                k = f"{parts[0]}.{parts[1]}.correction.gate.{rest.split('.', 1)[1]}"
            elif rest.startswith("correction."):
                k = f"{parts[0]}.{parts[1]}.correction.correction.{rest.split('.', 1)[1]}"
        out[k] = v
    return out


def load_known(name: str, device: str | Any = "cpu",
               cache_dir: str | Path | None = None) -> tuple[Any, CheckpointSpec]:
    """Download the checkpoint, build the matching model, load weights, return both.

    Legacy ORN V2 checkpoints had a flatter block layout (PerturbativeCorrection
    was not a submodule). We translate their keys to the current scheme on load.
    """
    import torch
    from orn.models import build_model

    spec = get_spec(name)
    path = fetch_checkpoint(name, cache_dir=cache_dir)

    model = build_model(spec.arch, dict(spec.config))
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)

    if spec.arch == "orn_v2":
        state = _remap_legacy_v2_keys(state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    model.eval().to(device)
    return model, spec
