"""orn agent-demo — single-command laptop demo of the ORN function-calling model.

Boots the OpenAI-compatible server in-process on a free port, runs three
demo prompts (weather, currency, chitchat) through Pydantic AI, prints
the results, and tears the server down.

First run downloads the 2.4 GB checkpoint to the HuggingFace cache;
subsequent runs are about 5 seconds of startup. Auto-detects device.
"""
from __future__ import annotations


def cmd(args) -> None:
    from orn.serve.agent_demo import main
    raise SystemExit(main())


def register(subparsers) -> None:
    sp = subparsers.add_parser(
        "agent-demo",
        help="Run a Pydantic AI demo against a local in-process FC server",
    )
    sp.set_defaults(fn=cmd)
