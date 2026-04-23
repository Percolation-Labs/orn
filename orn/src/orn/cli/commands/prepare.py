"""orn prepare <dataset> — download + tokenise training data."""
from __future__ import annotations

import sys


def cmd(args) -> None:
    from orn.data.prepare import prepare_tinystories, prepare_openwebtext
    if args.dataset == "tinystories":
        prepare_tinystories(output_dir=args.output, max_tokens=args.max_tokens)
    elif args.dataset in ("openwebtext", "fineweb-edu"):
        prepare_openwebtext(output_dir=args.output, max_tokens=args.max_tokens)
    else:
        print(f"Unknown dataset '{args.dataset}'. Options: tinystories, openwebtext, fineweb-edu")
        sys.exit(1)


def register(subparsers) -> None:
    sp = subparsers.add_parser("prepare", help="Download + tokenise data")
    sp.add_argument("dataset", choices=["tinystories", "openwebtext", "fineweb-edu"])
    sp.add_argument("--output", default="data")
    sp.add_argument("--max-tokens", type=int, default=50_000_000)
    sp.set_defaults(fn=cmd)
