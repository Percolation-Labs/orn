"""JSON training-config loader.

All training runs are specified by a JSON file under `orn/training/configs/`.
The schema is uniform across architectures; the `model.arch` field picks the
model class and the rest of `model.*` is forwarded to its config dataclass.

Schema:
  model      {arch, d_model|d_l|..., n_layers, n_q_heads, n_kv_heads, d_head,
              vocab_size, seq_len, d_corr, ffn_width_mult, ...}
  data       {dataset, path}
  tokenizer  {name}
  training   TrainingConfig keys (lr, wd, betas, max_tokens, batch_size, ...)
  loss       {type, aux_weights}
  eval       {tasks}
  hf         {push_repo, push_every_tokens}

Presets: `dry_run`, `orn_v2_108m_8b`, `orn_v3_61m_2b`, `lorn_v4_66m_100m`, ...
Resolve by name (preset) or by path (custom).
"""
from __future__ import annotations

import json
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "configs"


def list_configs() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


def load_config(name_or_path: str) -> dict:
    """Resolve `name_or_path` to a config dict.

    If a preset by that name exists under CONFIG_DIR, use it; otherwise treat
    as a filesystem path.
    """
    preset = CONFIG_DIR / f"{name_or_path}.json"
    path = preset if preset.exists() else Path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No config named '{name_or_path}' (looked at {preset}) and no such file"
        )
    with open(path) as f:
        return json.load(f)
