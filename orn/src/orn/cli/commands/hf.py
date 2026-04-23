"""orn push / pull / pull-checkpoint — HuggingFace interactions."""
from __future__ import annotations


def cmd_push(args) -> None:
    from orn.utils.hf import push_checkpoint
    url = push_checkpoint(args.checkpoint, args.repo, private=not args.public)
    print(f"Uploaded: {url}")


def cmd_pull(args) -> None:
    from orn.utils.hf import pull_model
    mdl, tok = pull_model(args.model, cache_dir=args.cache)
    print(f"Pulled {args.model}: {type(mdl).__name__}, vocab={tok.vocab_size}")


def cmd_pull_checkpoint(args) -> None:
    from orn.utils import known_checkpoints, fetch_checkpoint
    if args.list or not args.name:
        print(f"{'NAME':<28s} {'ARCH':<10s} DESCRIPTION")
        for s in known_checkpoints():
            print(f"  {s.name:<26s} {s.arch:<10s} {s.description}")
        print("\nDefault: the unsuffixed name (e.g. `orn-v3-605m`) is the latest/most-trained\n"
              "checkpoint for each model. Suffixed names (-1B, -2B, -branch-2B, ...) pin\n"
              "to a specific training marker.")
        print("\nUsage: orn pull-checkpoint <name>")
        return
    path = fetch_checkpoint(args.name, cache_dir=args.cache)
    print(f"Cached at: {path}")


def register(subparsers) -> None:
    sp = subparsers.add_parser("push", help="Upload a checkpoint to HuggingFace")
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--repo", required=True)
    sp.add_argument("--public", action="store_true")
    sp.set_defaults(fn=cmd_push)

    sp = subparsers.add_parser("pull", help="Pull a baseline model from HuggingFace")
    sp.add_argument("--model", required=True,
                    help="gpt2, smollm2-135m, pythia-70m, pythia-410m")
    sp.add_argument("--cache", default=None)
    sp.set_defaults(fn=cmd_pull)

    sp = subparsers.add_parser("pull-checkpoint",
                                help="Download a published ORN checkpoint from HuggingFace")
    sp.add_argument("name", nargs="?", default=None,
                    help="e.g. orn-v3-605m. Omit or pass --list to see options.")
    sp.add_argument("--list", action="store_true")
    sp.add_argument("--cache", default=None)
    sp.set_defaults(fn=cmd_pull_checkpoint)
