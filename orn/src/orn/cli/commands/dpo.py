"""orn dpo — Direct Preference Optimisation (helpful-assistant register).

Wraps `orn.training.posttrain.dpo.main()`. See
`drafts/post_training_a_600m_orn.md` § "DPO Stage 3". DPO transfers the
register reliably; it does NOT transfer factuality, refusal of bad-faith
prompts, or reasoning quality at small-model scale.
"""
from __future__ import annotations


def cmd(args) -> None:
    from orn.training.posttrain.dpo import main as dpo_main
    import sys
    sys.argv = ["orn-dpo"] + args.passthrough
    raise SystemExit(dpo_main())


def register(subparsers) -> None:
    sp = subparsers.add_parser(
        "dpo",
        help="DPO fine-tuning on a preference dataset. Pass-through args.",
        epilog="Run `orn dpo -- --help` to see the trainer's own arguments.",
    )
    sp.add_argument("passthrough", nargs="*",
                    help="Forwarded verbatim to the DPO trainer's argparse.")
    sp.set_defaults(fn=cmd)
