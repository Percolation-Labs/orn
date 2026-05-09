#!/usr/bin/env python3
"""Qualitative function-calling evaluation for ORN v3 3B FC-SFT.

Tests whether the FC SFT actually taught the model to emit function-call
JSON when given prompts with tool definitions, vs just modifying its
general SFT character.

The training data was glaiveai/glaive-function-calling-v2, which has the
shape:

  SYSTEM: You are a helpful assistant with access to the following functions.
  Use them if required - {json tool def}

  USER: <user query>
  A: <functioncall> {"name": ..., "arguments": '...'}  <|endoftext|>

When wrapped in ChatML for SFT, this is most naturally:

  <|im_start|>system\n<glaive system text incl JSON tool def><|im_end|>\n
  <|im_start|>user\n<user query><|im_end|>\n
  <|im_start|>assistant\n  ... model continues ...

The assistant either answers directly OR emits '<functioncall> {...}'.

We test ~10 prompts covering: should-call, should-not-call, ambiguous,
multi-tool, unknown-tool. We use greedy + low-temp sampling.
"""
import argparse, json, os, sys, time, re
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/orn")

from orn.training.posttrain._compat import (
    FullORN, VanillaTransformer, build_orn, build_transformer,
)


# Glaive system prompt template
def glaive_system(tool_defs):
    """tool_defs: list of dicts (the JSON tool schemas)."""
    if not tool_defs:
        return "You are a helpful assistant."
    body = "You are a helpful assistant with access to the following functions. Use them if required -\n"
    for td in tool_defs:
        body += json.dumps(td, indent=4) + "\n\n"
    return body.rstrip()


def chatml_prompt(system, user):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")


# Tool definitions for testing
WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name, e.g. 'Paris, France'"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
}
NEWS_TOOL = {
    "name": "get_news_headlines",
    "description": "Get the latest news headlines",
    "parameters": {
        "type": "object",
        "properties": {
            "country": {"type": "string", "description": "The country for which to fetch news"}
        },
        "required": ["country"]
    }
}
CURRENCY_TOOL = {
    "name": "convert_currency",
    "description": "Convert an amount from one currency to another",
    "parameters": {
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "The amount to convert"},
            "from_currency": {"type": "string"},
            "to_currency": {"type": "string"}
        },
        "required": ["amount", "from_currency", "to_currency"]
    }
}
CALC_TOOL = {
    "name": "calculate",
    "description": "Evaluate an arithmetic expression",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Arithmetic expression e.g. '2+3*4'"}
        },
        "required": ["expression"]
    }
}
EMAIL_TOOL = {
    "name": "send_email",
    "description": "Send an email to a recipient",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"}
        },
        "required": ["to", "subject", "body"]
    }
}

# Test prompts
PROMPTS = [
    {
        "label": "should_call_weather",
        "category": "should_call",
        "tools": [WEATHER_TOOL],
        "user": "What's the weather in Paris right now? Use celsius.",
        "expect": "tool_call",
    },
    {
        "label": "should_call_news",
        "category": "should_call",
        "tools": [NEWS_TOOL],
        "user": "Can you tell me the latest news headlines for Japan?",
        "expect": "tool_call",
    },
    {
        "label": "should_call_currency",
        "category": "should_call",
        "tools": [CURRENCY_TOOL],
        "user": "Convert 250 USD to EUR for me please.",
        "expect": "tool_call",
    },
    {
        "label": "should_NOT_call_chitchat",
        "category": "should_not_call",
        "tools": [WEATHER_TOOL],
        "user": "Hi! What's your favourite colour?",
        "expect": "natural_reply",
    },
    {
        "label": "should_NOT_call_outofscope",
        "category": "should_not_call",
        "tools": [WEATHER_TOOL],
        "user": "Can you book me a flight from London to New York for tomorrow?",
        "expect": "natural_reply_or_refusal",
    },
    {
        "label": "multi_tool_pick_right",
        "category": "multi_tool",
        "tools": [WEATHER_TOOL, NEWS_TOOL, CURRENCY_TOOL],
        "user": "What's the weather like in Tokyo?",
        "expect": "tool_call:get_weather",
    },
    {
        "label": "multi_tool_currency_pick",
        "category": "multi_tool",
        "tools": [WEATHER_TOOL, NEWS_TOOL, CURRENCY_TOOL],
        "user": "How many euros are 1000 Japanese yen worth?",
        "expect": "tool_call:convert_currency",
    },
    {
        "label": "calc_should_call",
        "category": "should_call",
        "tools": [CALC_TOOL],
        "user": "What is 47 times 113 plus 1024?",
        "expect": "tool_call",
    },
    {
        "label": "email_args",
        "category": "should_call",
        "tools": [EMAIL_TOOL],
        "user": "Send an email to bob@example.com with subject 'Lunch?' and body 'Free at 1pm tomorrow.'",
        "expect": "tool_call",
    },
    {
        "label": "general_chat_no_tools",
        "category": "should_not_call",
        "tools": [],
        "user": "Tell me a fun fact about octopuses.",
        "expect": "natural_reply",
    },
]


def load_l(ckpt_path, device):
    print(f"[load] {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    mc = cfg["model"]
    arch = cfg.get("arch") or "orn"
    if arch == "orn":
        model = build_orn(mc)
    else:
        model = build_transformer(mc)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, mc, arch, ckpt


@torch.no_grad()
def generate(model, tok, prompt_text, max_new=200, temperature=0.2, top_p=0.95,
             eot_id=None, im_end_seq=None, device="cuda"):
    """Stop on (a) eot_id token, or (b) the full <|im_end|> token SEQUENCE
    appearing as a suffix of the generation. Single-token stops on the
    components of <|im_end|> would falsely stop on '<' which is the start
    of '<functioncall>'."""
    model.eval()
    ids = tok.encode_ordinary(prompt_text)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []
    for _ in range(max_new):
        seq_max = getattr(model, "seq_len", 16384)
        if x.shape[1] > seq_max:
            x = x[:, -seq_max:]
        with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                  dtype=torch.bfloat16, enabled=(device == "cuda")):
            if isinstance(model, FullORN):
                logits = model(x, is_causal=True)
            else:
                logits = model(x)
        logits = logits[0, -1, :].float()
        if temperature <= 0.01:
            tok_id = int(torch.argmax(logits).item())
        else:
            logits = logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = probs.cumsum(dim=-1)
                mask = cum > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits[mask] = float("-inf")
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(0, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            tok_id = int(torch.multinomial(probs, 1).item())
        # Stop checks
        if eot_id is not None and tok_id == eot_id:
            break
        generated.append(tok_id)
        x = torch.cat([x, torch.tensor([[tok_id]], device=device)], dim=1)
        # Check <|im_end|> token-sequence suffix
        if im_end_seq and len(generated) >= len(im_end_seq):
            if generated[-len(im_end_seq):] == list(im_end_seq):
                # Strip the trailing im_end seq from the generated tokens
                generated = generated[:-len(im_end_seq)]
                break
    return tok.decode(generated)


def classify_response(text):
    """Heuristic classification of the model's response.

    GlaiveAI emits `<functioncall> {"name": "...", "arguments": '{...}'} <|endoftext|>`
    where the arguments value is a SINGLE-QUOTED string of JSON (not nested JSON).
    """
    has_functioncall_tag = "<functioncall>" in text or "functioncall" in text.lower()
    has_tool_call_tag = "tool_call" in text.lower() or "<tool_call>" in text
    # GlaiveAI-style block: { ... "name" ... "arguments" ... '{...}' ... }
    glaive_blocks = re.findall(
        r"\{\s*\"name\"\s*:\s*\"([^\"]+)\"\s*,\s*\"arguments\"\s*:\s*'(\{[^']*\})'\s*\}",
        text, flags=re.DOTALL)
    # Also look for any name+arguments JSON-ish object
    generic_blocks = re.findall(
        r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}',
        text, flags=re.DOTALL)
    parsed = None
    parsed_name = None
    parsed_args = None
    parse_err = None
    if glaive_blocks:
        name, args_str = glaive_blocks[0]
        parsed_name = name
        try:
            parsed_args = json.loads(args_str)
            parsed = {"name": name, "arguments": parsed_args}
        except Exception as e:
            parse_err = f"{type(e).__name__}: {e} on {args_str!r}"
    return {
        "has_functioncall_tag": has_functioncall_tag,
        "has_tool_call_tag": has_tool_call_tag,
        "glaive_blocks_found": len(glaive_blocks),
        "generic_blocks_found": len(generic_blocks),
        "parsed_name": parsed_name,
        "parsed_arguments": parsed_args,
        "parse_err": parse_err,
        "first_block_raw": (text[text.find("<functioncall>"):text.find("<functioncall>")+400]
                              if has_functioncall_tag else None),
        "emitted_tool_call": has_functioncall_tag or has_tool_call_tag or len(glaive_blocks) > 0 or len(generic_blocks) > 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/workspace/checkpoints/v3_3B_fc.pt")
    ap.add_argument("--out", default="/workspace/results/eval_fc_qualitative.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_new", type=int, default=200)
    args = ap.parse_args()

    import tiktoken
    tok = tiktoken.get_encoding("gpt2")
    im_end_seq = tok.encode_ordinary("<|im_end|>")
    eot = tok.eot_token
    print(f"[tok] <|im_end|> seq={im_end_seq}  eot={eot}", flush=True)

    model, mc, arch, ckpt = load_l(args.checkpoint, args.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[init] arch={arch} d={mc['d']} L={mc['n_layers']} "
          f"seq_len={mc['seq_len']} vocab={mc['vocab']} params={n_params:.1f}M",
          flush=True)

    results = {
        "checkpoint": args.checkpoint,
        "step": ckpt.get("step"),
        "arch": arch,
        "n_params_M": n_params,
        "timestamp": time.time(),
        "sampling": {"temperature_low": 0.2, "temperature_zero": 0.0, "top_p": 0.95,
                       "max_new": args.max_new},
        "prompts": [],
    }

    for spec in PROMPTS:
        sys_text = glaive_system(spec["tools"])
        prompt = chatml_prompt(sys_text, spec["user"])

        print(f"\n{'='*72}")
        print(f"[{spec['label']}] expect={spec['expect']}")
        print(f"USER: {spec['user']}")
        out_greedy = generate(model, tok, prompt, max_new=args.max_new,
                               temperature=0.0, top_p=1.0,
                               eot_id=eot, im_end_seq=im_end_seq, device=args.device)
        out_lt = generate(model, tok, prompt, max_new=args.max_new,
                          temperature=0.2, top_p=0.95,
                          eot_id=eot, im_end_seq=im_end_seq, device=args.device)
        cls_g = classify_response(out_greedy)
        cls_l = classify_response(out_lt)
        print(f"GREEDY: {out_greedy[:500]}")
        print(f"  classified: tool_call={cls_g['emitted_tool_call']} glaive={cls_g['glaive_blocks_found']} generic={cls_g['generic_blocks_found']}")
        print(f"LOW-T : {out_lt[:500]}")
        print(f"  classified: tool_call={cls_l['emitted_tool_call']} glaive={cls_l['glaive_blocks_found']} generic={cls_l['generic_blocks_found']}")

        results["prompts"].append({
            "label": spec["label"],
            "category": spec["category"],
            "expect": spec["expect"],
            "user": spec["user"],
            "system_prompt": sys_text,
            "chatml_prompt": prompt,
            "greedy": {"text": out_greedy, "classification": cls_g},
            "low_temp": {"text": out_lt, "classification": cls_l},
        })

    n = len(PROMPTS)
    tool_call_count_greedy = sum(1 for r in results["prompts"]
                                   if r["greedy"]["classification"]["emitted_tool_call"])
    tool_call_count_lowt = sum(1 for r in results["prompts"]
                                  if r["low_temp"]["classification"]["emitted_tool_call"])
    expected_call = sum(1 for s in PROMPTS if s["expect"].startswith("tool_call"))
    expected_nocall = sum(1 for s in PROMPTS if s["expect"].startswith("natural"))

    results["summary"] = {
        "n_prompts": n,
        "expected_tool_call": expected_call,
        "expected_no_call": expected_nocall,
        "emitted_tool_call_greedy": tool_call_count_greedy,
        "emitted_tool_call_low_temp": tool_call_count_lowt,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*72}\n SUMMARY\n{'='*72}")
    print(f"Prompts: {n}")
    print(f"Expected tool_call: {expected_call}, expected no_call: {expected_nocall}")
    print(f"Greedy emitted tool_call:    {tool_call_count_greedy}/{n}")
    print(f"Low-temp emitted tool_call:  {tool_call_count_lowt}/{n}")
    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
