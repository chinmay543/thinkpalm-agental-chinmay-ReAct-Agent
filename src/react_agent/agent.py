"""ReAct-style agent loop using OpenAI chat completions + tool calls."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from react_agent.tools import TOOL_SCHEMAS, dispatch_tool, get_memory_store

SYSTEM_PROMPT = """You are a helpful multi-tool assistant. You must answer the user's question accurately.

Use tools when needed:
- calculator: for any arithmetic or numeric expression.
- web_search: for current events, facts you are unsure about, or anything requiring up-to-date information.
- memory_read / memory_write: to recall or persist user preferences, names, or facts across conversations.

Think step by step: decide which tools to call, interpret observations, and give a clear final answer in natural language.
If a tool fails, explain briefly and try another approach if appropriate."""


def run_agent(
    user_message: str,
    *,
    model: str | None = None,
    max_rounds: int = 8,
) -> str:
    """
    Run the agent until the model returns a message without tool calls or max_rounds is hit.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your environment or a .env file."
        )

    client = OpenAI(api_key=api_key)
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    memory = get_memory_store()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        choice = response.choices[0]
        msg = choice.message

        # Append assistant message (including tool_calls if any)
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if msg.content:
            assistant_msg["content"] = msg.content
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            return msg.content or ""

        for tc in msg.tool_calls:
            name = tc.function.name
            args = tc.function.arguments or "{}"
            result = dispatch_tool(name, args, memory=memory)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    return "Stopped: maximum tool rounds reached without a final answer."
