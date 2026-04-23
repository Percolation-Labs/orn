"""orn configs — list available training presets."""
from __future__ import annotations


def cmd(args) -> None:
    from orn.training import list_configs, load_config, CONFIG_DIR
    print(f"Training configs ({CONFIG_DIR}):")
    for name in list_configs():
        cfg = load_config(name)
        desc = cfg.get("_description", "")
        arch = cfg.get("model", {}).get("arch", "?")
        print(f"  {name:<30s}  [{arch}]  {desc}")


def register(subparsers) -> None:
    sp = subparsers.add_parser("configs", help="List training presets")
    sp.set_defaults(fn=cmd)
