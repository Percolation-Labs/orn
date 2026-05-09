"""orn serve — boot the OpenAI-compatible FastAPI server.

Exposes a function-calling ORN checkpoint as a `POST /v1/chat/completions`
endpoint that Pydantic AI (and other OpenAI-compatible clients) can drive
without modification. See `drafts/post_training_a_600m_orn.md` § "Running
it locally" for the full recipe and `orn.serve` for the scaffold.
"""
from __future__ import annotations


def cmd(args) -> None:
    import uvicorn
    from orn.serve.server import build_app

    app = build_app(
        backend=args.backend,
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def register(subparsers) -> None:
    sp = subparsers.add_parser("serve", help="Boot the OpenAI-compatible FC server")
    sp.add_argument("--backend", choices=["mock", "orn"], default="mock",
                    help="`mock` (no GPU, canned replies) or `orn` (real ORN inference)")
    sp.add_argument("--checkpoint", default="mr-saoirse/orn-v3-3b-fc-sft",
                    help="Local .pt path or HuggingFace repo id")
    sp.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                    help="Inference device. 'auto' picks cuda > mps > cpu.")
    sp.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto",
                    help="Inference dtype. 'auto' = bf16 on cuda, fp16 on mps, fp32 on cpu.")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8080)
    sp.set_defaults(fn=cmd)
