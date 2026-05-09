"""End-to-end smoke test of the serve scaffold against the mock backend.

No model load, no GPU; just verifies that:
- the FastAPI app builds with `--backend mock`
- a `/v1/chat/completions` POST round-trips through the translation seam
- the response carries an OpenAI-shaped `tool_calls` block when the prompt
  matches the mock's "weather" trigger

If this test fails the translation seam (`openai_to_chatml_glaive` → mock
→ `parse_glaive_output` → OpenAI response shape) is broken; the real HF
backend would be broken too.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from orn.serve.server import build_app


def _client():
    app = build_app(backend="mock", checkpoint=None)
    return TestClient(app)


def test_health():
    resp = _client().get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["backend"] == "mock"


def test_models_list():
    resp = _client().get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) >= 1


def test_chat_completions_weather_tool_call():
    """The mock backend returns a Glaive-format `<functioncall>` for prompts
    containing 'weather'. Verify the server parses and returns OpenAI tool_calls."""
    body = {
        "model": "orn-fc",
        "messages": [
            {"role": "user", "content": "What's the weather in Paris right now?"}
        ],
        "tools": [{
            "type": "function", "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        }],
        "max_tokens": 80,
        "temperature": 0.0,
    }
    resp = _client().post("/v1/chat/completions", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    msg = data["choices"][0]["message"]
    assert msg["tool_calls"], f"expected tool_calls, got {msg!r}"
    fn = msg["tool_calls"][0]["function"]
    assert fn["name"] == "get_weather"


def test_chat_completions_chat_mode():
    """Without tools, mock backend should fall through to plain text content."""
    body = {
        "model": "orn-fc",
        "messages": [{"role": "user", "content": "Tell me about octopuses."}],
        "max_tokens": 50,
        "temperature": 0.0,
    }
    resp = _client().post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    msg = resp.json()["choices"][0]["message"]
    # Mock returns plain content for non-weather, non-tool prompts
    assert msg.get("content") or msg.get("tool_calls")


def test_streaming_rejected():
    """Streaming isn't implemented in this scaffold; the server should refuse."""
    body = {
        "model": "orn-fc",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    resp = _client().post("/v1/chat/completions", json=body)
    assert resp.status_code == 400
