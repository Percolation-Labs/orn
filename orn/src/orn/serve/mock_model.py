from __future__ import annotations

import json
import time
from typing import Any


class MockORNModel:
    def __init__(self, latency_ms: int = 50) -> None:
        self.latency_ms = latency_ms

    def generate(
        self,
        prompt: str,
        tools_schema: list[dict[str, Any]] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ) -> str:
        time.sleep(self.latency_ms / 1000.0)

        if self._has_recent_tool_response(prompt):
            return "Based on the tool result, here is the answer for you. <|endoftext|>"

        last_user = self._last_user_turn(prompt)
        text = (last_user or "").lower()

        tool_names: list[str] = []
        if tools_schema:
            for t in tools_schema:
                fn = t.get("function", t)
                n = fn.get("name")
                if n:
                    tool_names.append(n)

        if "weather" in text and "get_weather" in tool_names:
            location = "Paris" if "paris" in text else ("Tokyo" if "tokyo" in text else "London")
            unit = "celsius" if "celsius" in text or "c " in text else "fahrenheit"
            args = json.dumps({"location": location, "unit": unit})
            return f"<functioncall> {{\"name\": \"get_weather\", \"arguments\": '{args}'}} <|endoftext|>"

        if ("convert" in text or "yen" in text or "usd" in text or "eur" in text) and "convert_currency" in tool_names:
            args = json.dumps({"amount": 250, "from_currency": "USD", "to_currency": "EUR"})
            return f"<functioncall> {{\"name\": \"convert_currency\", \"arguments\": '{args}'}} <|endoftext|>"

        if "favourite colour" in text or "favorite color" in text:
            return "I'm just a language model — I don't have personal preferences. <|endoftext|>"

        if "book" in text and "flight" in text:
            return (
                "I'm sorry, but I'm unable to assist with booking flights. "
                "My current capabilities are limited to the available tools. <|endoftext|>"
            )

        if "octopus" in text:
            return (
                "Octopuses have three hearts and blue, copper-based blood. "
                "Two hearts pump blood to the gills; the third pumps to the body. <|endoftext|>"
            )

        return "I don't have a tool that can handle that request. <|endoftext|>"

    @staticmethod
    def _has_recent_tool_response(prompt: str) -> bool:
        last_user = prompt.rfind("<|im_start|>user")
        last_tool = prompt.rfind("<|im_start|>tool")
        return last_tool > last_user

    @staticmethod
    def _last_user_turn(prompt: str) -> str | None:
        marker = "<|im_start|>user\n"
        end = "<|im_end|>"
        last = None
        cursor = 0
        while True:
            i = prompt.find(marker, cursor)
            if i < 0:
                break
            j = prompt.find(end, i + len(marker))
            if j < 0:
                last = prompt[i + len(marker):]
                break
            last = prompt[i + len(marker):j]
            cursor = j + len(end)
        return last
