"""Single-command laptop demo of the ORN function-calling model.

  python3 -m orn.serve.try_it
  # or, after `pip install -e .`:
  orn try-fc

Boots the OpenAI-compatible server in-process on a free port, waits for
`/health`, runs three demo prompts (weather, currency, chitchat) through
Pydantic AI, prints the results, and tears down the server on exit.

First run pulls the 2.4 GB checkpoint from HuggingFace into ~/.cache;
subsequent boots are about 5 seconds. Auto-detects device (cuda > mps > cpu)
and dtype (bf16 on CUDA, fp16 on MPS, fp32 on CPU). Caps at 80 tokens per
turn so debugging on a slow laptop backend doesn't take forever.

Set `HUGGING_FACE_ACCESS_KEY` for the first run if the checkpoint repo is
private; subsequent runs use the local HF cache.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
from contextlib import closing
from functools import partial


log = partial(print, flush=True)


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(url: str, timeout: float = 180.0) -> None:
    import urllib.request
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

    log(f"[try_it] device={device}  port={port}")
    log(f"[try_it] checkpoint={DEFAULT_FC_CHECKPOINT} (cached after first run)")
    log(f"[try_it] booting server… (~10–30s on MPS for first model load)")

    app = build_app(backend="orn", checkpoint=DEFAULT_FC_CHECKPOINT, device=device)
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    try:
        _wait_for_health(f"{base_url}/health")
        log(f"[try_it] server ready at {base_url}\n")
        _run_demo(base_url)
        return 0
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _run_demo(base_url: str) -> None:
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.settings import ModelSettings

    model = OpenAIChatModel(
        "orn-fc",
        provider=OpenAIProvider(base_url=f"{base_url}/v1", api_key="not-needed"),
        settings=ModelSettings(max_tokens=80, temperature=0.0),
    )
    agent = Agent(model=model)

    @agent.tool_plain
    def get_weather(location: str, unit: str = "celsius") -> str:
        return f"The weather in {location} is cloudy, 14 degrees {unit}."

    @agent.tool_plain
    def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
        rate = 0.92  # placeholder
        return f"{amount} {from_currency} = {amount * rate:.2f} {to_currency}."

    prompts = [
        ("weather",  "What's the weather in Paris right now?"),
        ("currency", "How many euros are 250 USD?"),
        ("chitchat", "Hi! What's your favourite colour?"),
    ]

    for label, q in prompts:
        log(f"[{label}] USER: {q}")
        t0 = time.time()
        try:
            result = agent.run_sync(q)
            elapsed = time.time() - t0
            log(f"[{label}] MODEL ({elapsed:.1f}s): {result.output}")
        except Exception as e:
            elapsed = time.time() - t0
            log(f"[{label}] FAILED ({elapsed:.1f}s): {type(e).__name__}: {e}")
        log("")


if __name__ == "__main__":
    sys.exit(main())
