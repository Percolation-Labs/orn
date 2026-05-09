#!/usr/bin/env python3
"""Single-prompt MLX benchmark for the ORN FC-SFT 600M checkpoint.

No server, no Pydantic AI — direct call into the MLX model. Mirrors the
prompt used by serve/bench_local.py so timings are comparable to the
PyTorch baseline (~35s on M4 MPS).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orn.serve_mlx.load import load_fc_checkpoint
from orn.serve_mlx.generate import generate
from orn.serve_mlx.quantize import quantize_for_inference


WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name, e.g. 'Paris, France'"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"],
    },
}


def glaive_system(tool_defs: list[dict]) -> str:
    if not tool_defs:
        return "You are a helpful assistant."
    body = "You are a helpful assistant with access to the following functions. Use them if required -\n"
    for td in tool_defs:
        body += json.dumps(td, indent=4) + "\n\n"
    return body.rstrip()


def chatml_prompt(system: str, user: str) -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=["none", "4", "8"], default="none",
                    help="int4 / int8 weight quantization on FFN+attn+embedding")
    ap.add_argument("--group_size", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=80)
    args = ap.parse_args()

    print(f"[bench-mlx] loading FC checkpoint…  quant={args.quant}", flush=True)
    t_load_0 = time.time()
    model, mc = load_fc_checkpoint(dtype="fp16")
    if args.quant != "none":
        bits = int(args.quant)
        quantize_for_inference(model, bits=bits, group_size=args.group_size)
        # Re-materialise after weight surgery.
        import mlx.core as mx
        mx.eval(model.parameters())
    t_load = time.time() - t_load_0
    print(f"[bench-mlx] load wall-clock: {t_load:.1f}s", flush=True)
    model.summary()

    import tiktoken
    tok = tiktoken.get_encoding("gpt2")

    sys_text = glaive_system([WEATHER_TOOL])
    user_text = "What's the weather in Paris right now? Use celsius."
    prompt = chatml_prompt(sys_text, user_text)

    print(f"\n[bench-mlx] prompt len: {len(tok.encode_ordinary(prompt))} tokens", flush=True)

    t0 = time.time()
    text, stats = generate(
        model, tok, prompt,
        max_tokens=args.max_tokens,
        temperature=0.0, top_p=1.0,
        print_progress=True,
    )
    wall = time.time() - t0

    label = "fp16" if args.quant == "none" else f"int{args.quant}"
    print(f"\n[bench-mlx][{label}] wall-clock: {wall:.1f}s for one tool call")
    print(f"[bench-mlx][{label}] prefill: {stats['prefill_s']:.2f}s  "
          f"decode: {stats['decode_s']:.2f}s  "
          f"decode_tok_s: {stats['decode_tok_s']:.2f}")
    print(f"[bench-mlx][{label}] OUTPUT:\n{text}\n")

    is_fc = ("<functioncall>" in text or "functioncall" in text.lower()
             or '"name"' in text)
    print(f"[bench-mlx][{label}] looks like a function-call emission: {is_fc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
