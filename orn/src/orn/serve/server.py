"""FastAPI app exposing an ORN function-calling checkpoint as an
OpenAI-compatible LLM endpoint.

Endpoints:
- `POST /v1/chat/completions` — accepts the OpenAI ChatCompletions request
  shape (messages + optional tools), translates to ChatML+Glaive, runs the
  model, parses the GlaiveAI tool-call format, returns OpenAI-shaped
  `tool_calls` or `content`.
- `GET  /v1/models` — stub model list with the loaded checkpoint id.
- `GET  /health` — liveness probe.

The translation seam (`serve.glaive_format`) is the load-bearing piece. The
ORN was function-call-fine-tuned on GlaiveAI's `<functioncall> {...}
<|endoftext|>` format; this server is what makes the resulting model look
like a normal OpenAI-compatible backend to clients like Pydantic AI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from orn.serve import DEFAULT_FC_CHECKPOINT
from orn.serve.glaive_format import openai_to_chatml_glaive, parse_glaive_output
from orn.serve.mock_model import MockORNModel
from orn.serve.openai_types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    FunctionCall,
    ModelCard,
    ModelList,
    ResponseChoice,
    ToolCall,
    Usage,
)


def _make_orn_backend(checkpoint: str, device: str = "auto", dtype: str = "auto"):
    from orn.serve.orn_backend import ORNBackend
    return ORNBackend(checkpoint, device=device, dtype=dtype)


def build_app(backend: str, checkpoint: str | None,
              device: str = "auto", dtype: str = "auto") -> FastAPI:
    """Instantiate the FastAPI app with the requested backend wired in.

    Args:
        backend: `mock` for the canned-response backend (no GPU, no model),
            or `hf` for the real ORN inference backend.
        checkpoint: local path or HF repo id; ignored for the mock backend.
        device: `auto` | `cuda` | `mps` | `cpu`.
        dtype: `auto` | `fp32` | `fp16` | `bf16`.
    """
    app = FastAPI(title="ORN FC Server", version="0.1.0")

    if backend == "mock":
        model = MockORNModel(latency_ms=50)
    elif backend == "orn":
        model = _make_orn_backend(checkpoint or DEFAULT_FC_CHECKPOINT,
                                  device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown backend: {backend!r}; expected 'mock' or 'orn'")

    app.state.model = model
    app.state.model_id = checkpoint or DEFAULT_FC_CHECKPOINT

    @app.get("/v1/models")
    def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=app.state.model_id)])

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        if req.stream:
            raise HTTPException(status_code=400, detail="streaming not supported in this scaffold")

        messages_raw: list[dict[str, Any]] = [m.model_dump(exclude_none=True) for m in req.messages]
        tools_raw = [t.model_dump() for t in req.tools] if req.tools else None

        prompt = openai_to_chatml_glaive(messages_raw, tools=tools_raw)

        raw_text = app.state.model.generate(
            prompt=prompt,
            tools_schema=tools_raw,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )

        parsed = parse_glaive_output(raw_text)

        if "tool_call" in parsed:
            tc = parsed["tool_call"]
            tool_call = ToolCall(
                function=FunctionCall(
                    name=tc["name"],
                    arguments=tc.get("arguments") or json.dumps(tc.get("arguments_obj") or {}),
                )
            )
            choice = ResponseChoice(
                message=ChatMessage(role="assistant", content=None, tool_calls=[tool_call]),
                finish_reason="tool_calls",
            )
        else:
            choice = ResponseChoice(
                message=ChatMessage(role="assistant", content=parsed.get("content", "")),
                finish_reason="stop",
            )

        return ChatCompletionResponse(model=app.state.model_id, choices=[choice], usage=Usage())

    @app.get("/health")
    def health():
        return JSONResponse({"status": "ok", "backend": backend, "model": app.state.model_id})

    return app


def main():
    ap = argparse.ArgumentParser(prog="orn-serve")
    ap.add_argument("--backend", choices=["mock", "orn"], default="mock",
                    help="`mock` (no GPU, canned replies) or `orn` (real ORN inference)")
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_FC_CHECKPOINT,
                    help="local .pt path or HF repo id")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    import uvicorn
    app = build_app(backend=args.backend, checkpoint=args.checkpoint,
                    device=args.device, dtype=args.dtype)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
