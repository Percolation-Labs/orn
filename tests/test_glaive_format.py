from __future__ import annotations

import json

import pytest

from orn.serve.glaive_format import (
    glaive_system,
    openai_to_chatml_glaive,
    parse_glaive_output,
)


WEATHER_TOOL = {
    "type": "function",
    "function": {
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
    },
}

CURRENCY_TOOL = {
    "type": "function",
    "function": {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string"},
                "to_currency": {"type": "string"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
}


def test_glaive_system_no_tools():
    assert glaive_system(None) == "You are a helpful assistant."
    assert glaive_system([]) == "You are a helpful assistant."


def test_glaive_system_with_tool():
    s = glaive_system([WEATHER_TOOL])
    assert "You are a helpful assistant with access to the following functions" in s
    assert '"name": "get_weather"' in s


def test_openai_to_chatml_glaive_basic():
    msgs = [{"role": "user", "content": "What's the weather in Paris? Use celsius."}]
    prompt = openai_to_chatml_glaive(msgs, tools=[WEATHER_TOOL])
    assert prompt.startswith("<|im_start|>system\n")
    assert "<|im_start|>user\nWhat's the weather in Paris? Use celsius.<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert '"name": "get_weather"' in prompt


def test_openai_to_chatml_glaive_preserves_user_system():
    msgs = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Hi"},
    ]
    prompt = openai_to_chatml_glaive(msgs, tools=[WEATHER_TOOL])
    assert "Be terse." in prompt
    assert "with access to the following functions" in prompt


def test_parse_canonical_glaive_weather():
    text = "<functioncall> {\"name\": \"get_weather\", \"arguments\": '{\"location\": \"Paris\", \"unit\": \"celsius\"}'} <|endoftext|>"
    out = parse_glaive_output(text)
    assert "tool_call" in out
    assert out["tool_call"]["name"] == "get_weather"
    assert out["tool_call"]["arguments_obj"] == {"location": "Paris", "unit": "celsius"}
    assert out["dialect"] == "glaive"


def test_parse_canonical_glaive_currency():
    text = "<functioncall> {\"name\": \"convert_currency\", \"arguments\": '{\"amount\": 250, \"from_currency\": \"USD\", \"to_currency\": \"EUR\"}'} <|endoftext|>"
    out = parse_glaive_output(text)
    assert out["tool_call"]["name"] == "convert_currency"
    assert out["tool_call"]["arguments_obj"]["amount"] == 250


def test_parse_tool_call_tag_dialect_leakage():
    text = "<tool_call>{\"name\": \"get_weather_color\", \"arguments\": {}}</tool_call> <|endoftext|>"
    out = parse_glaive_output(text)
    assert "tool_call" in out
    assert out["tool_call"]["name"] == "get_weather_color"
    assert out["dialect"] == "tool_call_tag"


def test_parse_natural_language_refusal():
    text = "I'm sorry, but I'm unable to assist with booking flights. <|endoftext|>"
    out = parse_glaive_output(text)
    assert "content" in out
    assert "tool_call" not in out
    assert out["content"].startswith("I'm sorry")


def test_parse_strips_endoftext_and_imend():
    text = "Hello world<|im_end|>"
    out = parse_glaive_output(text)
    assert out["content"] == "Hello world"


def test_parse_handles_calculator_failure_mode():
    text = "<functioncall> {\"name\": \"calculate_excel_calculate_47\", \"arguments\": '{\"expression\": \"47+112+1024\"}'} <|endoftext|>"
    out = parse_glaive_output(text)
    assert out["tool_call"]["name"] == "calculate_excel_calculate_47"
    assert out["tool_call"]["arguments_obj"] == {"expression": "47+112+1024"}


def test_parse_doesnt_crash_on_garbage():
    text = "<functioncall> total nonsense {{{ <|endoftext|>"
    out = parse_glaive_output(text)
    assert "content" in out or "tool_call" in out


def test_parse_email_multiline_arguments():
    text = (
        "<functioncall> {\"name\": \"send_email\", \"arguments\": '{"
        "\"to\": \"Bob@example.com\","
        "\"subject\": \"Lunch\","
        "\"body\": \"Free at 1pm tomorrow.\""
        "}'} <|endoftext|>"
    )
    out = parse_glaive_output(text)
    assert out["tool_call"]["name"] == "send_email"
    assert out["tool_call"]["arguments_obj"]["to"] == "Bob@example.com"


def test_round_trip_assistant_tool_call_render():
    msgs = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "Paris"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 14}"},
        {"role": "user", "content": "Thanks"},
    ]
    prompt = openai_to_chatml_glaive(msgs, tools=[WEATHER_TOOL])
    assert "<functioncall>" in prompt
    assert "<|im_start|>tool" in prompt
