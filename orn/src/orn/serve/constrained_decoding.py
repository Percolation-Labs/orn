from __future__ import annotations

from typing import Any


def constrained_generate(
    model: Any,
    prompt: str,
    tools_schema: list[dict[str, Any]] | None,
    max_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    """
    Forward stub for outlines-based constrained decoding.

    Plan when wired up against the real ORN backend:

    1. Extract `name` enum from tools_schema:
           names = [t["function"]["name"] for t in tools_schema]
       Build a JSON schema where `name` is `enum: names`.

    2. Build a per-name argument schema. Outlines accepts a JSON schema as
       string and produces a regex/FSM. The parameter shape lives at
       tools_schema[i]["function"]["parameters"]. Glaive emits the args as
       a SINGLE-QUOTED JSON string, not nested JSON; constrain the inner
       string with `outlines.generate.json(model, schema)` and wrap the
       result with the literal `<functioncall> {"name": "...", "arguments":
       '...'} <|endoftext|>` template.

    3. Use a two-stage decode:
           a. Free-decode the first N tokens to detect intent (functioncall
              vs natural language) by sampling a single classification token
              after `<|im_start|>assistant\n`.
           b. If functioncall, switch to outlines-constrained decode.
              Otherwise free-decode until <|im_end|> / <|endoftext|>.

    4. Library calls (when implemented):
           import outlines
           regex_fsm = outlines.fsm.json_schema.build_regex_from_schema(schema_str)
           generator = outlines.generate.regex(model, regex_fsm)
           text = generator(prompt, max_tokens=max_tokens)

       Note: outlines wraps the model differently for HF Transformers vs raw
       PyTorch. The ORN is custom, so we'd implement the
       `outlines.models.transformers.Transformers`-style adapter manually.
       Logit processor: at each step, mask logits to the FSM-allowed token
       set; outlines provides this via `RegexLogitsProcessor`.

    5. Two failure modes this is designed to fix:
           a. Tool-name confabulation (`get_weather_color`,
              `calculate_excel_calculate_47`). Solved by the `name` enum.
           b. Schema enum violation (`unit: "metric"`). Solved by the per-
              parameter enum constraint inside the args schema.
    """
    raise NotImplementedError(
        "constrained_generate is a forward stub. Wire up `outlines` only on "
        "the GPU box (Vast 4090) once the HF backend is plumbed in server.py."
    )
