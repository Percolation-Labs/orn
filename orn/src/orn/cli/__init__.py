"""
ORN Command-Line Interface.

Each subcommand lives in its own module under `orn.cli.commands.*`; this
file is only the dispatcher. Commands self-register by exposing a
`register(subparsers)` function that creates their subparser(s) and wires
the handler(s).

Programmatic users should prefer the library API directly (e.g.
`from orn.reproduce import get`, `from orn.eval import eval_benchmarks`).
"""
from __future__ import annotations

import argparse
import sys

from orn.cli.commands import (
    configs,
    prepare,
    train,
    predict,
    diagnose,
    evaluation,
    hf,
    reproduce,
    serve,
    agent_demo,
    mlx_bench,
    sft,
    dpo,
    long_ctx_ft,
    fc_sft,
    eval_fc,
)


_COMMAND_MODULES = [
    configs,
    prepare,
    train,
    predict,
    diagnose,
    evaluation,
    hf,
    reproduce,
    serve,
    agent_demo,
    mlx_bench,
    sft,
    dpo,
    long_ctx_ft,
    fc_sft,
    eval_fc,
]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="orn",
        description="Orbital Response Network CLI. Run `orn <command> --help` for details.",
    )
    subparsers = parser.add_subparsers(dest="command")

    for mod in _COMMAND_MODULES:
        mod.register(subparsers)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.fn(args)


if __name__ == "__main__":
    main()
