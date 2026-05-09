"""orn sft — supervised fine-tuning (instruction or multi-turn chat).

Wraps `orn.training.posttrain.sft.main()` and plumbs CLI args through to
its argparse. Used for Stage 2 (instruction SFT on OpenHermes-2.5) and
Stage 3 (multi-turn chat SFT on UltraChat) — same trainer, different
config / dataset.

See `drafts/post_training_a_600m_orn.md` § "SFT Stage 1" and § "SFT Stage 2".
"""
from __future__ import annotations


def cmd(args) -> None:
    from orn.training.posttrain.sft import main as sft_main
    import sys
    sys.argv = ["orn-sft"] + args.passthrough
    raise SystemExit(sft_main())


def register(subparsers) -> None:
    sp = subparsers.add_parser(
        "sft",
        help="Supervised fine-tuning (instruction / chat). Pass-through args.",
        epilog="Run `orn sft -- --help` to see the trainer's own arguments.",
    )
    sp.add_argument("passthrough", nargs="*",
                    help="Forwarded verbatim to the SFT trainer's argparse.")
    sp.set_defaults(fn=cmd)
