"""orn mlx-bench — single-prompt MLX benchmark on a function-calling checkpoint.

Loads the checkpoint into MLX (fp16 by default; `--quant 4` or `--quant 8`
to quantize), runs one weather-tool prompt end-to-end, prints prefill +
decode timings and the model's response. See `drafts/post_training_a_600m_orn.md`
§ "What the inference recipe looks like" for the headline numbers (10s/turn
on M4 at int4).
"""
from __future__ import annotations


def cmd(args) -> None:
    from orn.serve_mlx.benchmark import main as bench_main
    import sys
    sys.argv = ["orn-mlx-bench"]
    if args.quant != "none":
        sys.argv += ["--quant", args.quant]
    if args.checkpoint:
        sys.argv += ["--checkpoint", args.checkpoint]
    raise SystemExit(bench_main())


def register(subparsers) -> None:
    sp = subparsers.add_parser(
        "mlx-bench",
        help="Single-prompt MLX inference benchmark (Apple Silicon)",
    )
    sp.add_argument("--quant", choices=["none", "4", "8"], default="none",
                    help="Quantize weights to int4 or int8 before inference. "
                         "'none' keeps fp16. int4 is the recommended default "
                         "for laptop demos.")
    sp.add_argument("--checkpoint", default=None,
                    help="Local .pt path or HF repo id (default: "
                         "mr-saoirse/orn-v3-3b-fc-sft).")
    sp.set_defaults(fn=cmd)
