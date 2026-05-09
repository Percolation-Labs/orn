"""orn eval-fc — qualitative function-calling evaluation.

Runs ten hand-curated FC prompts against a checkpoint with greedy and
low-temperature sampling, classifies each response (tool-call vs refusal
vs hallucinated tool name), and writes a JSON report.

See `drafts/post_training_a_600m_orn.md` § "Did function-calling SFT
transfer real capability?" — this command produces the data behind that
section's success/failure examples.
"""
from __future__ import annotations


def cmd(args) -> None:
    from orn.eval.fc_qualitative import main as eval_main
    import sys
    sys.argv = ["orn-eval-fc"] + args.passthrough
    raise SystemExit(eval_main())


def register(subparsers) -> None:
    sp = subparsers.add_parser(
        "eval-fc",
        help="Qualitative function-calling evaluation (10 hand-curated prompts).",
        epilog="Run `orn eval-fc -- --help` for the eval arguments.",
    )
    sp.add_argument("passthrough", nargs="*",
                    help="Forwarded verbatim to the eval script's argparse.")
    sp.set_defaults(fn=cmd)
