from __future__ import annotations

import json
import re
from typing import Any


def _tool_schema_to_glaive_dict(tool: dict[str, Any]) -> dict[str, Any]:
    fn = tool.get("function", tool)
    return {
        "name": fn.get("name", ""),
        "description": fn.get("description", ""),
        "parameters": fn.get("parameters", {}),
    }


def glaive_system(tool_defs: list[dict[str, Any]] | None) -> str:
    if not tool_defs:
        return "You are a helpful assistant."
    body = "You are a helpful assistant with access to the following functions. Use them if required -\n"
    for td in tool_defs:
        body += json.dumps(_tool_schema_to_glaive_dict(td), indent=4) + "\n\n"
    return body.rstrip()


def _render_assistant_tool_call(tool_calls: list[dict[str, Any]]) -> str:
    call = tool_calls[0]
    fn = call.get("function", call)
    name = fn.get("name", "")
    args = fn.get("arguments", "{}")
    if isinstance(args, dict):
        args_str = json.dumps(args, ensure_ascii=False)
    else:
        args_str = str(args)
    return f"<functioncall> {{\"name\": \"{name}\", \"arguments\": '{args_str}'}} "


def openai_to_chatml_glaive(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> str:
    system_text = None
    rendered_turns: list[str] = []

    has_user_system = any(m.get("role") == "system" for m in messages)
    if has_user_system:
        for m in messages:
            if m.get("role") == "system":
                system_text = m.get("content") or ""
                break
    if tools:
        if system_text:
            system_text = system_text.strip() + "\n\n" + glaive_system(tools)
        else:
            system_text = glaive_system(tools)
    elif system_text is None:
        system_text = "You are a helpful assistant."

    rendered_turns.append(f"<|im_start|>system\n{system_text}<|im_end|>")

    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            rendered_turns.append(f"<|im_start|>user\n{m.get('content','')}<|im_end|>")
        elif role == "assistant":
            tcs = m.get("tool_calls")
            if tcs:
                content = _render_assistant_tool_call(tcs)
            else:
                content = m.get("content") or ""
            rendered_turns.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        elif role == "tool":
            rendered_turns.append(f"<|im_start|>tool\n{m.get('content','')}<|im_end|>")

    rendered_turns.append("<|im_start|>assistant\n")
    return "\n".join(rendered_turns)


_GLAIVE_BLOCK_RE = re.compile(
    r"\{\s*\"name\"\s*:\s*\"([^\"]+)\"\s*,\s*\"arguments\"\s*:\s*'(\{[^']*\})'\s*\}",
    flags=re.DOTALL,
)
_TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    flags=re.DOTALL,
)
_GENERIC_NAME_ARGS_RE = re.compile(
    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^{}]*\})\s*\}',
    flags=re.DOTALL,
)


def _strip_terminators(text: str) -> str:
    for term in ("<|endoftext|>", "<|im_end|>"):
        idx = text.find(term)
        if idx >= 0:
            text = text[:idx]
    return text.strip()


def parse_glaive_output(text: str) -> dict[str, Any]:
    cleaned = _strip_terminators(text)

    # The training-eval bug noted: stop-tokens on byte 27 ('<') wrongly halted at
    # '<' inside '<functioncall>'. Don't repeat it: parse on text, not on tokens.
    glaive_match = _GLAIVE_BLOCK_RE.search(cleaned)
    if glaive_match:
        name, args_str = glaive_match.group(1), glaive_match.group(2)
        try:
            args_obj = json.loads(args_str)
        except json.JSONDecodeError:
            args_obj = None
        return {
            "tool_call": {
                "name": name,
                "arguments": json.dumps(args_obj) if args_obj is not None else args_str,
                "arguments_obj": args_obj,
                "raw_args_string": args_str,
            },
            "dialect": "glaive",
        }

    tag_match = _TOOL_CALL_TAG_RE.search(cleaned)
    if tag_match:
        blob = tag_match.group(1)
        try:
            obj = json.loads(blob)
            name = obj.get("name", "")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args_obj = json.loads(args)
                except json.JSONDecodeError:
                    args_obj = None
                args_str = args
            else:
                args_obj = args
                args_str = json.dumps(args, ensure_ascii=False)
            return {
                "tool_call": {
                    "name": name,
                    "arguments": args_str,
                    "arguments_obj": args_obj,
                    "raw_args_string": args_str,
                },
                "dialect": "tool_call_tag",
            }
        except json.JSONDecodeError:
            pass

    if "<functioncall>" in cleaned:
        gen_match = _GENERIC_NAME_ARGS_RE.search(cleaned)
        if gen_match:
            name, args_str = gen_match.group(1), gen_match.group(2)
            try:
                args_obj = json.loads(args_str)
            except json.JSONDecodeError:
                args_obj = None
            return {
                "tool_call": {
                    "name": name,
                    "arguments": args_str,
                    "arguments_obj": args_obj,
                    "raw_args_string": args_str,
                },
                "dialect": "glaive_loose",
            }

    return {"content": cleaned, "dialect": "text"}
