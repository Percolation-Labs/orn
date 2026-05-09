"""Single-prompt speed benchmark for the local FC server.

Bypasses Pydantic AI's multi-turn retry loop so the timing is clean —
one POST, one tool-call emission, one set of `[gen]` timings.

  python3 -m orn.serve.bench_local

Boots the server in-process on a free port, sends one `chat/completions`
request with a single `get_weather` tool definition, prints prefill +
decode timings, and exits.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.request
from contextlib import closing
from functools import partial


log = partial(print, flush=True)


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(url: str, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy in {timeout}s")


def main() -> int:
    import uvicorn
    from orn.serve.orn_backend import _auto_device
    from orn.serve.server import build_app
    from orn.serve import DEFAULT_FC_CHECKPOINT

    device = _auto_device()
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    log(f"[bench] device={device}  port={port}")
    log(f"[bench] booting server (auto-dtype: bf16 CUDA / fp16 MPS / fp32 CPU)…")

    app = build_app(backend="orn", checkpoint=DEFAULT_FC_CHECKPOINT, device=device)
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    try:
        _wait_for_health(f"{base_url}/health")
        log(f"[bench] server ready\n")

        body = {
            "model": "orn-fc",
            "messages": [
                {"role": "user", "content": "What's the weather in Paris right now? Use celsius."}
            ],
            "tools": [{
                "type": "function", "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                }
            }],
            "temperature": 0.0,
            "max_tokens": 80,
        }
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
        wall = time.time() - t0

        log(f"\n[bench] wall-clock: {wall:.1f}s for one tool call")
        msg = resp["choices"][0]["message"]
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]["function"]
            log(f"[bench] tool_call: name={tc['name']}  args={tc['arguments']}")
        else:
            log(f"[bench] content: {msg.get('content', '')[:200]}")
        return 0
    finally:
        server.should_exit = True
        thread.join(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
