"""orn fc-sft — function-calling supervised fine-tuning.

Stage 6 of the post-training pipeline. Teaches the model to emit GlaiveAI's
structured tool-call format (`<functioncall> {...} <|endoftext|>`) when a
user query matches a tool definition in the system prompt, and to refuse
cleanly when no tool matches.

See `drafts/post_training_a_600m_orn.md` § "Stage five: function-calling
SFT — does it transfer real capability?". The default dataset is
`glaiveai/glaive-function-calling-v2`; the data layer at
`orn.training.posttrain.data_fc` also supports xLAM-FC and
Gorilla-OpenFunctions if those datasets are prepared.
"""
from __future__ import annotations


def cmd(args) -> None:
    from orn.training.posttrain.fc_sft import main as fc_main
    import sys
    sys.argv = ["orn-fc-sft"] + args.passthrough
    raise SystemExit(fc_main())


def register(subparsers) -> None:
    sp = subparsers.add_parser(
        "fc-sft",
        help="Function-calling SFT (GlaiveAI / xLAM / Gorilla).",
        epilog="Run `orn fc-sft -- --help` for the trainer arguments.",
    )
    sp.add_argument("passthrough", nargs="*",
                    help="Forwarded verbatim to the trainer's argparse.")
    sp.set_defaults(fn=cmd)
