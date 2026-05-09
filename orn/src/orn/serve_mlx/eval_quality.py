#!/usr/bin/env python3
"""Compare fp16 vs int4 outputs on the 10-prompt FC qualitative suite.

Replicates the prompts from training/eval_fc_qualitative.py exactly, runs
greedy decode at fp16 and int4, then compares:
  - Did each version emit a tool call?
  - Same tool name?
  - Same parsed arguments JSON?

Prints a 10-row table and a single summary line ("X/10 prompts had
identical tool emission") that can go in the article.

Greedy only — temperature=0 — to make the comparison deterministic.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlx.core as mx

from orn.serve_mlx.load import load_fc_checkpoint
from orn.serve_mlx.generate import generate
from orn.serve_mlx.quantize import quantize_for_inference


# ── Prompt suite (cloned from training/eval_fc_qualitative.py) ────────

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
NEWS_TOOL = {
    "name": "get_news_headlines",
    "description": "Get the latest news headlines",
    "parameters": {
        "type": "object",
        "properties": {
            "country": {"type": "string", "description": "The country for which to fetch news"}
        },
        "required": ["country"],
    },
}
CURRENCY_TOOL = {
    "name": "convert_currency",
    "description": "Convert an amount from one currency to another",
    "parameters": {
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "The amount to convert"},
            "from_currency": {"type": "string"},
            "to_currency": {"type": "string"},
        },
        "required": ["amount", "from_currency", "to_currency"],
    },
}
CALC_TOOL = {
    "name": "calculate",
    "description": "Evaluate an arithmetic expression",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Arithmetic expression e.g. '2+3*4'"}
        },
        "required": ["expression"],
    },
}
EMAIL_TOOL = {
    "name": "send_email",
    "description": "Send an email to a recipient",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["to", "subject", "body"],
    },
}

PROMPTS = [
    {"label": "should_call_weather", "tools": [WEATHER_TOOL],
     "user": "What's the weather in Paris right now? Use celsius."},
    {"label": "should_call_news", "tools": [NEWS_TOOL],
     "user": "Can you tell me the latest news headlines for Japan?"},
    {"label": "should_call_currency", "tools": [CURRENCY_TOOL],
     "user": "Convert 250 USD to EUR for me please."},
    {"label": "chitchat_no_call", "tools": [WEATHER_TOOL],
     "user": "Hi! What's your favourite colour?"},
    {"label": "outofscope_no_call", "tools": [WEATHER_TOOL],
     "user": "Can you book me a flight from London to New York for tomorrow?"},
    {"label": "multi_tool_weather", "tools": [WEATHER_TOOL, NEWS_TOOL, CURRENCY_TOOL],
     "user": "What's the weather like in Tokyo?"},
    {"label": "multi_tool_currency", "tools": [WEATHER_TOOL, NEWS_TOOL, CURRENCY_TOOL],
     "user": "How many euros are 1000 Japanese yen worth?"},
    {"label": "calc_should_call", "tools": [CALC_TOOL],
     "user": "What is 47 times 113 plus 1024?"},
    {"label": "email_args", "tools": [EMAIL_TOOL],
     "user": "Send an email to bob@example.com with subject 'Lunch?' and body 'Free at 1pm tomorrow.'"},
    {"label": "general_chat_no_tools", "tools": [],
     "user": "Tell me a fun fact about octopuses."},
]


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


def classify(text: str) -> dict:
    """Pull out (name, arguments) from a Glaive-style function-call.

    Two surface forms appear in our SFT outputs:
      A) {"name": "X", "arguments": '{"k": "v"}'}   — single-quoted JSON string
      B) {"name": "X", "arguments": { "k": "v" }}   — bare JSON object
    We accept both. The argument-string boundary is ambiguous (no escaping)
    so we extract the name first and the arguments lazily.
    """
    has_fc = "<functioncall>" in text or "functioncall" in text.lower()
    name = None
    args = None
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    if name_match:
        name = name_match.group(1)
        # Form A: single-quoted JSON string
        a_match = re.search(r'"arguments"\s*:\s*\'(\{[^\']*\})\'', text, flags=re.DOTALL)
        if a_match:
            try:
                args = json.loads(a_match.group(1))
            except Exception:
                args = None
        else:
            # Form B: try to balance braces after "arguments":
            b_idx = text.find('"arguments"')
            if b_idx >= 0:
                brace_idx = text.find('{', b_idx)
                if brace_idx >= 0:
                    depth = 0
                    end = -1
                    for i in range(brace_idx, len(text)):
                        if text[i] == '{':
                            depth += 1
                        elif text[i] == '}':
                            depth -= 1
                            if depth == 0:
                                end = i
                                break
                    if end > brace_idx:
                        try:
                            args = json.loads(text[brace_idx:end + 1])
                        except Exception:
                            args = None
    return {
        "emitted": has_fc or (name is not None),
        "name": name,
        "arguments": args,
        "raw_first_block": text[:200],
    }


def _run_all(model, tok, eot, im_end_seq, max_tokens=80) -> list[dict]:
    rows = []
    for spec in PROMPTS:
        sys_text = glaive_system(spec["tools"])
        prompt = chatml_prompt(sys_text, spec["user"])
        t0 = time.time()
        text, _stats = generate(
            model, tok, prompt,
            max_tokens=max_tokens, temperature=0.0, top_p=1.0,
            eot_id=eot, im_end_seq=im_end_seq, print_progress=False,
        )
        cls = classify(text)
        cls["wall_s"] = time.time() - t0
        cls["text"] = text
        cls["label"] = spec["label"]
        rows.append(cls)
        print(f"  [{spec['label']:<24s}] emit={cls['emitted']} "
              f"name={cls['name']} ({cls['wall_s']:.1f}s)", flush=True)
    return rows


def _compare(fp16_row: dict, int4_row: dict) -> tuple[bool, str]:
    """Return (match, why)."""
    if fp16_row["emitted"] != int4_row["emitted"]:
        return False, f"emit mismatch (fp16={fp16_row['emitted']}, int4={int4_row['emitted']})"
    if not fp16_row["emitted"]:
        # Both no-call: count as match (both behaved the same way).
        return True, "both natural reply"
    if fp16_row["name"] != int4_row["name"]:
        return False, f"tool name mismatch ({fp16_row['name']} vs {int4_row['name']})"
    if fp16_row["arguments"] != int4_row["arguments"]:
        return False, f"args differ"
    return True, "exact match"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_tokens", type=int, default=80)
    ap.add_argument("--out", default="serve_mlx/eval_quant_quality.json")
    ap.add_argument("--bits", type=int, default=4)
    args = ap.parse_args()

    import tiktoken
    tok = tiktoken.get_encoding("gpt2")
    im_end_seq = tok.encode_ordinary("<|im_end|>")
    eot = tok.eot_token

    print(f"\n[eval-quant] === fp16 baseline ===", flush=True)
    model_fp, _ = load_fc_checkpoint(dtype="fp16")
    fp16_rows = _run_all(model_fp, tok, eot, im_end_seq, args.max_tokens)
    del model_fp
    import gc; gc.collect()

    print(f"\n[eval-quant] === int{args.bits} (re-loaded + quantized) ===",
          flush=True)
    model_q, _ = load_fc_checkpoint(dtype="fp16")
    quantize_for_inference(model_q, bits=args.bits, group_size=64, verbose=False)
    mx.eval(model_q.parameters())
    intq_rows = _run_all(model_q, tok, eot, im_end_seq, args.max_tokens)

    # Comparison table.
    print(f"\n{'='*88}")
    print(f"{'label':<24s} {'fp16 emit':<10s} {'int{} emit'.format(args.bits):<11s} "
          f"{'fp16 name':<22s} {'int{} name'.format(args.bits):<22s} match")
    print("-"*88)
    n_match = 0
    n_emit_match = 0  # both emitted same tool, regardless of args
    table = []
    for fp_r, int_r in zip(fp16_rows, intq_rows):
        ok, why = _compare(fp_r, int_r)
        if ok:
            n_match += 1
        # Looser: both emitted? same tool name? (args are e.g. location strings
        # that may differ between Paris,France vs Paris in benign ways.)
        same_tool = (fp_r["emitted"] == int_r["emitted"]
                     and fp_r["name"] == int_r["name"])
        if same_tool:
            n_emit_match += 1
        mark = "OK" if ok else "MISS"
        print(f"{fp_r['label']:<24s} {str(fp_r['emitted']):<10s} "
              f"{str(int_r['emitted']):<11s} "
              f"{str(fp_r['name'])[:21]:<22s} "
              f"{str(int_r['name'])[:21]:<22s} {mark}  {why}")
        table.append({"label": fp_r["label"], "match": ok, "why": why,
                      "fp16": fp_r, f"int{args.bits}": int_r})
    print("-"*88)
    print(f"SUMMARY: {n_match}/{len(PROMPTS)} prompts had identical tool emission "
          f"(name + arguments).")
    print(f"         {n_emit_match}/{len(PROMPTS)} prompts had matching emit decision + tool name.")
    print(f"{'='*88}\n")

    with open(args.out, "w") as f:
        json.dump({"summary": {"identical": n_match,
                                "tool_name_match": n_emit_match,
                                "n": len(PROMPTS)},
                   "rows": table},
                  f, indent=2, default=str)
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
