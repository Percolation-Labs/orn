#!/usr/bin/env python3
"""Function-calling SFT data prep: download → ChatML-ify → tokenise → shard.

Stage 4 of the 605M post-training stack — runs AFTER long-context FT and
BEFORE FC-SFT (`train_fc_sft.py`).

Datasets combined (~150-200K examples after filtering):
  - glaiveai/glaive-function-calling-v2  (~110K, multi-turn chat)
  - Salesforce/xlam-function-calling-60k (~60K, single-turn query/tools/answers)
  - ShishirPatil/gorilla-openfunctions-v1 (alt for ToolBench multi-turn)

Output:
  /workspace/data_shards/fc_sft/{glaive,xlam,gorilla}.bin   (uint32 token ids)
  /workspace/data_shards/fc_sft/{glaive,xlam,gorilla}_mask.bin  (uint8 loss mask)
  /workspace/data_shards/fc_sft/{glaive,xlam,gorilla}_offsets.bin  (uint64 row starts)
  /workspace/data_shards/fc_sft/metadata.json

ChatML-extended template (de-facto convention; matches Hermes/Llama-3 tool-use):

    <|im_start|>system
    You have access to these tools:
    {tools_json_schema}
    <|im_end|>
    <|im_start|>user
    {user message}
    <|im_end|>
    <|im_start|>assistant
    {optional reasoning}
    <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    <|im_end|>
    <|im_start|>tool
    <tool_response>{result_json}</tool_response>
    <|im_end|>
    <|im_start|>assistant
    {final answer}
    <|im_end|>

Note on special tokens:
  Like the SFT trainer, we use `tok.encode_ordinary(...)`. The ChatML wrappers
  (<|im_start|>, <|im_end|>) and the new tool wrappers (<tool_call>,
  </tool_call>, <tool_response>, </tool_response>) get encoded as ordinary BPE
  subword sequences; no vocab expansion required. The model learns them as
  distinctive token patterns. This matches how OpenHermes/Llama-3 tool-use
  models train.

Run:
  # Local smoke (100 examples, fast, writes to data/fc_sft/)
  python3 training/data_fc.py --smoke

  # On RunPod, full dataset to /workspace path:
  python3 training/data_fc.py \\
      --out_dir /workspace/data_shards/fc_sft \\
      --max_len 4096

Auth note: Glaive + xLAM may require `huggingface-cli login` first.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reuse the ChatML constants + tokenisation primitives from the SFT trainer
from orn.training.posttrain.sft import (
    CHATML_START, CHATML_END,
    format_chatml, tokenise_with_mask, _norm_message,
)


# ══════════════════════════════════════════════════════════════════════
# Tool-call wrapper tokens (subword sequences, no vocab expansion)
# ══════════════════════════════════════════════════════════════════════

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESP_OPEN = "<tool_response>"
TOOL_RESP_CLOSE = "</tool_response>"


# ══════════════════════════════════════════════════════════════════════
# Per-dataset normalisation → list[{role, content}] in our extended ChatML
# ══════════════════════════════════════════════════════════════════════

def _wrap_assistant_with_tool_call(reasoning, name, arguments):
    """Build assistant content that emits a tool-call, with optional reasoning."""
    call_obj = {"name": name, "arguments": arguments}
    call_str = json.dumps(call_obj, ensure_ascii=False)
    parts = []
    if reasoning:
        parts.append(reasoning.strip())
    parts.append(f"{TOOL_CALL_OPEN}{call_str}{TOOL_CALL_CLOSE}")
    return "\n".join(parts)


def _wrap_tool_response(result):
    """Tool turn content."""
    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False)
    return f"{TOOL_RESP_OPEN}{result}{TOOL_RESP_CLOSE}"


def _system_with_tools(base_system, tools_obj):
    """Attach tool schemas to the system message."""
    base = (base_system or "You are a helpful assistant with tool access.").strip()
    if isinstance(tools_obj, str):
        tools_str = tools_obj.strip()
    else:
        tools_str = json.dumps(tools_obj, ensure_ascii=False, indent=2)
    return f"{base}\n\nYou have access to these tools:\n{tools_str}"


# ─── Glaive FC v2 ────────────────────────────────────────────────────
# Schema: {"system": "...", "chat": "USER: ... ASSISTANT: ... <functioncall> ..."}
# Multi-turn, parsed from the `chat` string. Function calls in glaive look like:
#     ASSISTANT: <functioncall> {"name": "...", "arguments": ...}
#     FUNCTION RESPONSE: {"result": ...}

def _parse_glaive_chat(chat_str):
    """Best-effort parse of the glaive `chat` field into role-tagged turns."""
    if not chat_str:
        return None
    turns = []
    # Glaive uses these literal markers (uppercase, with trailing colon)
    markers = ["USER:", "ASSISTANT:", "FUNCTION RESPONSE:"]
    # Find marker positions in order
    cursor = 0
    found = []
    while cursor < len(chat_str):
        next_pos = -1
        next_marker = None
        for m in markers:
            p = chat_str.find(m, cursor)
            if p >= 0 and (next_pos < 0 or p < next_pos):
                next_pos = p
                next_marker = m
        if next_pos < 0:
            break
        found.append((next_pos, next_marker))
        cursor = next_pos + len(next_marker)
    if not found:
        return None
    # Slice content between consecutive markers
    for i, (pos, marker) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(chat_str)
        content = chat_str[pos + len(marker):end].strip()
        if marker == "USER:":
            turns.append(("user", content))
        elif marker == "ASSISTANT:":
            turns.append(("assistant", content))
        elif marker == "FUNCTION RESPONSE:":
            turns.append(("tool", content))
    return turns


def _glaive_to_messages(row):
    """Glaive row → list[{role, content}] using extended ChatML.
    `<functioncall>` markers in assistant content are converted to <tool_call>.
    """
    chat = row.get("chat") or row.get("conversations")
    system = row.get("system", "")
    if isinstance(chat, list):
        # Already structured (rare variant)
        turns = [(t.get("from", t.get("role", "user")).lower(),
                  t.get("value", t.get("content", "")))
                 for t in chat]
    else:
        turns = _parse_glaive_chat(chat)
    if not turns:
        return None

    messages = []
    if system:
        messages.append({"role": "system", "content": system.strip()})

    for role, content in turns:
        if role == "user" or role == "human":
            messages.append({"role": "user", "content": content})
        elif role == "assistant" or role == "gpt":
            # Convert <functioncall> {...}  →  <tool_call>{...}</tool_call>
            if "<functioncall>" in content:
                pre, _, rest = content.partition("<functioncall>")
                # rest may have trailing content after the JSON; we keep
                # only up to a reasonable JSON terminator. Glaive rows are
                # consistent: rest is one JSON object, possibly multi-line.
                rest = rest.strip()
                # Try to extract a single JSON object greedily
                fc_json, tail = _extract_first_json(rest)
                if fc_json is not None:
                    name = fc_json.get("name", "")
                    raw_args = fc_json.get("arguments", {})
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except Exception:
                            pass
                    asst = _wrap_assistant_with_tool_call(pre, name, raw_args)
                    if tail:
                        asst = asst + "\n" + tail.strip()
                    messages.append({"role": "assistant", "content": asst})
                else:
                    messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "assistant", "content": content})
        elif role == "tool" or role == "function":
            # Wrap the response in <tool_response>...</tool_response>
            messages.append({"role": "tool",
                             "content": _wrap_tool_response(content)})
    return messages


def _extract_first_json(s):
    """Extract the first balanced top-level JSON object from string `s`.
    Returns (parsed_obj, remainder_str) or (None, s)."""
    s = s.lstrip()
    if not s.startswith("{"):
        return None, s
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = s[: i + 1]
                tail = s[i + 1 :]
                try:
                    return json.loads(blob), tail
                except Exception:
                    return None, s
    return None, s


# ─── xLAM FC 60K ────────────────────────────────────────────────────
# Schema: {"query": "...", "tools": "[json schema]", "answers": "[json calls]"}
# Single-turn: user asks query, assistant emits one or more tool calls.

def _xlam_to_messages(row):
    query = row.get("query", "")
    tools = row.get("tools", "[]")
    answers = row.get("answers", "[]")
    if not query:
        return None
    # tools / answers are JSON-encoded strings in xLAM
    tools_obj = tools
    answers_obj = answers
    try:
        if isinstance(tools, str):
            tools_obj = json.loads(tools)
    except Exception:
        tools_obj = tools  # fall back to raw string
    try:
        if isinstance(answers, str):
            answers_obj = json.loads(answers)
    except Exception:
        return None
    if not isinstance(answers_obj, list) or not answers_obj:
        return None
    # Build messages
    messages = [
        {"role": "system",
         "content": _system_with_tools("You are a helpful assistant with tool access.",
                                        tools_obj)},
        {"role": "user", "content": query.strip()},
    ]
    # Concatenate one or more tool_call blocks in a single assistant turn
    parts = []
    for ans in answers_obj:
        if not isinstance(ans, dict):
            continue
        name = ans.get("name", "")
        args = ans.get("arguments", ans.get("args", {}))
        call_str = json.dumps({"name": name, "arguments": args},
                              ensure_ascii=False)
        parts.append(f"{TOOL_CALL_OPEN}{call_str}{TOOL_CALL_CLOSE}")
    if not parts:
        return None
    messages.append({"role": "assistant", "content": "\n".join(parts)})
    return messages


# ─── Gorilla openfunctions-v1 (alt for ToolBench) ────────────────────
# Common schema: {"Instruction": "...", "Functions": [...], "Output": "..."}
# Output is a function-call string we wrap in <tool_call>.

def _gorilla_to_messages(row):
    inst = row.get("Instruction") or row.get("question") or row.get("prompt")
    fns = row.get("Functions") or row.get("functions") or row.get("tools") or []
    out = row.get("Output") or row.get("answer") or row.get("output") or ""
    if not inst or not out:
        return None
    messages = [
        {"role": "system",
         "content": _system_with_tools("You are a helpful assistant with tool access.",
                                        fns)},
        {"role": "user", "content": str(inst).strip()},
    ]
    # `Output` is often "func_name(arg1=..., arg2=...)" or a JSON dict.
    # Try JSON first; fall back to wrapping the raw string as the call body.
    out_str = str(out).strip()
    parsed, _ = _extract_first_json(out_str)
    if parsed is not None and "name" in parsed:
        name = parsed["name"]
        args = parsed.get("arguments", parsed.get("args", {}))
        call_str = json.dumps({"name": name, "arguments": args},
                              ensure_ascii=False)
    else:
        # Heuristic: keep it as a free-form call wrapper
        call_str = json.dumps({"name": "_raw", "arguments": {"call": out_str}},
                              ensure_ascii=False)
    messages.append({"role": "assistant",
                     "content": f"{TOOL_CALL_OPEN}{call_str}{TOOL_CALL_CLOSE}"})
    return messages


# ══════════════════════════════════════════════════════════════════════
# Synthetic smoke fixtures (no HF download required)
# ══════════════════════════════════════════════════════════════════════

def _synth_smoke_row(i, kind="glaive"):
    """Generate a synthetic row matching each dataset's schema, for testing."""
    if kind == "glaive":
        chat = (
            f"USER: Multiply {i} and {i+1} please.\n"
            f"ASSISTANT: I'll call a tool. <functioncall> "
            f'{{"name": "multiply", "arguments": {{"a": {i}, "b": {i+1}}}}}\n'
            f'FUNCTION RESPONSE: {{"result": {i*(i+1)}}}\n'
            f"ASSISTANT: The product of {i} and {i+1} is {i*(i+1)}.\n"
        )
        return {"system": "You can call tools when needed.", "chat": chat}
    if kind == "xlam":
        return {
            "query": f"What is {i} squared?",
            "tools": json.dumps([{
                "name": "power",
                "description": "Raise base to exponent",
                "parameters": {"base": "number", "exp": "number"},
            }]),
            "answers": json.dumps([{
                "name": "power",
                "arguments": {"base": i, "exp": 2},
            }]),
        }
    # gorilla
    return {
        "Instruction": f"Get the weather for city #{i}.",
        "Functions": [{"name": "get_weather",
                       "parameters": {"city": "string"}}],
        "Output": json.dumps({"name": "get_weather",
                              "arguments": {"city": f"city_{i}"}}),
    }


# ══════════════════════════════════════════════════════════════════════
# Streaming + writing
# ══════════════════════════════════════════════════════════════════════

DATASETS = {
    "glaive": {
        "hf_id": "glaiveai/glaive-function-calling-v2",
        "to_messages": _glaive_to_messages,
        "split": "train",
    },
    "xlam": {
        "hf_id": "Salesforce/xlam-function-calling-60k",
        "to_messages": _xlam_to_messages,
        "split": "train",
    },
    "gorilla": {
        "hf_id": "ShishirPatil/gorilla-openfunctions-v1",
        "to_messages": _gorilla_to_messages,
        "split": "train",
    },
}


def _stream_dataset(name, smoke_n=None):
    """Yield raw rows. If smoke_n is set, yield synthetic rows instead."""
    if smoke_n is not None:
        for i in range(smoke_n):
            yield _synth_smoke_row(i, kind=name)
        return
    spec = DATASETS[name]
    from datasets import load_dataset
    last_err = None
    for split in (spec["split"], "train", "default"):
        try:
            ds = load_dataset(spec["hf_id"], split=split, streaming=True)
            print(f"[data:{name}] loaded hf_id={spec['hf_id']} split={split}",
                  flush=True)
            for row in ds:
                yield row
            return
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not load {spec['hf_id']}: {last_err}")


def _write_shard(out_dir, name, ids_list, mask_list):
    """Write one dataset's shard as three flat .bin files + offsets.
    ids: uint32 (we may exceed 65535 with gpt2 tok_50257)
    mask: uint8
    offsets: uint64 cumulative starts (last entry = total length)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ids_list:
        return {"examples": 0, "tokens": 0}
    flat_ids = np.concatenate([np.asarray(x, dtype=np.uint32) for x in ids_list])
    flat_mask = np.concatenate([np.asarray(m, dtype=np.uint8) for m in mask_list])
    offsets = np.zeros(len(ids_list) + 1, dtype=np.uint64)
    for i, x in enumerate(ids_list):
        offsets[i + 1] = offsets[i] + len(x)
    flat_ids.tofile(out_dir / f"{name}.bin")
    flat_mask.tofile(out_dir / f"{name}_mask.bin")
    offsets.tofile(out_dir / f"{name}_offsets.bin")
    return {
        "examples": len(ids_list),
        "tokens": int(flat_ids.size),
        "asst_tokens": int(flat_mask.sum()),
        "files": {
            "ids": f"{name}.bin",
            "mask": f"{name}_mask.bin",
            "offsets": f"{name}_offsets.bin",
        },
    }


def process_dataset(name, out_dir, tok, max_len, max_examples,
                    smoke_n=None, log_every=2000):
    spec = DATASETS[name]
    to_msgs = spec["to_messages"]
    ids_list, mask_list = [], []
    seen = 0
    kept = 0
    tool_call_token_count = 0
    tool_resp_token_count = 0
    t0 = time.time()
    for row in _stream_dataset(name, smoke_n=smoke_n):
        seen += 1
        try:
            msgs = to_msgs(row)
        except Exception:
            msgs = None
        if not msgs:
            continue
        # Need at least one user + one assistant turn
        if not any(m["role"] == "user" for m in msgs):
            continue
        if not any(m["role"] == "assistant" for m in msgs):
            continue
        ids, mask = tokenise_with_mask(msgs, tok, max_len)
        if len(ids) < 16:
            continue
        if sum(mask) < 8:
            continue
        # Sanity: count occurrences of literal <tool_call> / <tool_response>
        # in the rendered text (cheap; uses string from format_chatml call
        # would re-tokenise — skip for speed; do on first few only).
        if kept < 5:
            rendered = format_chatml(msgs)
            if TOOL_CALL_OPEN in rendered:
                tool_call_token_count += 1
            if TOOL_RESP_OPEN in rendered:
                tool_resp_token_count += 1
        ids_list.append(ids)
        mask_list.append(mask)
        kept += 1
        if (kept % log_every) == 0:
            dt = time.time() - t0
            mean_len = float(np.mean([len(x) for x in ids_list[-log_every:]]))
            print(f"  [{name}] seen={seen} kept={kept} "
                  f"mean_len={mean_len:.0f}  t={dt:.0f}s", flush=True)
        if kept >= max_examples:
            break
    info = _write_shard(out_dir, name, ids_list, mask_list)
    info["seen"] = seen
    info["had_tool_call_in_first5"] = tool_call_token_count
    info["had_tool_response_in_first5"] = tool_resp_token_count
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output dir for shards. Defaults: "
                         "/workspace/data_shards/fc_sft (if /workspace exists), "
                         "else data/fc_sft/")
    ap.add_argument("--max_len", type=int, default=4096,
                    help="Per-example tokenisation cap")
    ap.add_argument("--max_per_dataset", type=int, default=200_000)
    ap.add_argument("--datasets", type=str, default="glaive,xlam,gorilla",
                    help="Comma list from {glaive, xlam, gorilla}")
    ap.add_argument("--smoke", action="store_true",
                    help="Use 100 synthetic rows per dataset (no HF download).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Hard cap on examples per dataset (overrides smoke "
                         "default of 100).")
    args = ap.parse_args()

    if args.out_dir is None:
        if Path("/workspace").exists() and not args.smoke:
            args.out_dir = "/workspace/data_shards/fc_sft"
        else:
            args.out_dir = "data/fc_sft"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import tiktoken
    tok = tiktoken.get_encoding("gpt2")
    vocab = tok.n_vocab
    print(f"[init] tokenizer=gpt2 vocab={vocab} out_dir={out_dir}", flush=True)

    smoke_n = None
    if args.smoke:
        smoke_n = args.limit if args.limit is not None else 100

    wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
    metadata = {
        "tokenizer": "gpt2",
        "vocab_size": vocab,
        "max_len": args.max_len,
        "smoke": bool(args.smoke),
        "datasets": {},
        "tool_call_open": TOOL_CALL_OPEN,
        "tool_call_close": TOOL_CALL_CLOSE,
        "tool_response_open": TOOL_RESP_OPEN,
        "tool_response_close": TOOL_RESP_CLOSE,
    }

    total_ex = 0
    total_tok = 0
    for name in wanted:
        if name not in DATASETS:
            print(f"[warn] unknown dataset '{name}'; skipping", flush=True)
            continue
        cap = args.limit if args.limit is not None else args.max_per_dataset
        print(f"\n[run] processing {name} (cap={cap}, smoke={args.smoke})",
              flush=True)
        info = process_dataset(
            name, out_dir, tok,
            max_len=args.max_len,
            max_examples=cap,
            smoke_n=smoke_n,
        )
        metadata["datasets"][name] = info
        total_ex += info.get("examples", 0)
        total_tok += info.get("tokens", 0)
        print(f"[done] {name}: {info}", flush=True)

    metadata["total_examples"] = total_ex
    metadata["total_tokens"] = total_tok

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[finish] wrote {meta_path}", flush=True)
    print(f"  total_examples = {total_ex:,}")
    print(f"  total_tokens   = {total_tok:,}")


if __name__ == "__main__":
    main()
