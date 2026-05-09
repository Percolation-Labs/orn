"""orn long-ctx-ft — long-context fine-tuning via PoSE + NTK-rescaled RoPE.

Stage 5 of the post-training pipeline. Trains the DPO'd checkpoint on
short-content / long-virtual-position chunks (PoSE, Zhu et al. 2024) with
NTK-aware RoPE rescaling (Peng et al. 2023) so the model can use 16K-token
inputs without retraining at 16K content length.

See `drafts/post_training_a_600m_orn.md` § "Stage four: long-context
fine-tuning — landed".

Note: this stage requires `orn.rope_pose` and a sharded long-document
loader; the public package keeps placeholders so the trainer imports
cleanly, but to actually run the stage you'll need to port those two
modules from the private repo.
"""
from __future__ import annotations


def cmd(args) -> None:
    from orn.training.posttrain.long_context import main as lc_main
    import sys
    sys.argv = ["orn-long-ctx-ft"] + args.passthrough
    raise SystemExit(lc_main())


def register(subparsers) -> None:
    sp = subparsers.add_parser(
        "long-ctx-ft",
        help="Long-context fine-tuning (PoSE + NTK-rescaled RoPE).",
        epilog="Run `orn long-ctx-ft -- --help` for the trainer arguments.",
    )
    sp.add_argument("passthrough", nargs="*",
                    help="Forwarded verbatim to the trainer's argparse.")
    sp.set_defaults(fn=cmd)
