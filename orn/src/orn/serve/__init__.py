"""OpenAI-compatible serving scaffold for ORN function-calling checkpoints.

This subpackage exposes a 600M ORN as a normal LLM agent backend behind
`POST /v1/chat/completions`. The server translates GlaiveAI's emitted tool-call
format (`<functioncall> {...} <|endoftext|>`) into OpenAI's `tool_calls` shape
on the way out, and renders incoming tool definitions into the ChatML+Glaive
prompt format the model was trained on. With that translation in place,
[Pydantic AI](https://ai.pydantic.dev) drives the model with no model-specific
client code.

See the `Running it locally` section of `drafts/post_training_a_600m_orn.md`
for the inference recipe (KV cache, fp16, fused B@Wk projection, MLX port,
int4 quantization) and the laptop performance numbers.

Public modules:
- `openai_types`            Pydantic models for the OpenAI ChatCompletions request/response
- `glaive_format`           pure prompt-render + output-parse functions (no GPU)
- `mock_model`              drop-in fake backend for local dev / translation-layer tests
- `orn_backend`             real ORN loader: prefill + cached decode, fp16/bf16/fp32
- `server`                  FastAPI app: `/v1/chat/completions`, `/v1/models`, `/health`
- `agent_demo`              single-command Pydantic AI agent demo (in-process server)
- `benchmark`               single-prompt timing benchmark (in-process server, direct call)
- `constrained_decoding`    forward stub for `outlines`-based constrained decoding
"""

DEFAULT_FC_CHECKPOINT = "mr-saoirse/orn-v3-3b-fc-sft"
