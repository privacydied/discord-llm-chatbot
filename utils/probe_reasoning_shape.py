"""Dump the raw completion shape for the reasoning-leak case. [REH]

Answers: does the provider put chain-of-thought in a separate `reasoning`
field, or inline in `content`? The remedy differs entirely.

Usage: uv run python utils/probe_reasoning_shape.py
"""

import asyncio
import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv("/volume1/py/discord-llm-chatbot/.env")

from bot.openai_backend import _make_openai_async_client  # noqa: E402
from bot.tools import get_registry  # noqa: E402

MODEL = os.getenv("OPENAI_TEXT_MODEL")

HISTORY_RESULT = """=== BEGIN UNVERIFIED EXTERNAL CONTENT ===
[3 posts ago] frank - 2026-08-15 12:07 UTC
lunch?

[4 posts ago] dave - 2026-08-15 12:06 UTC
the deploy finished, all green

[5 posts ago] erin - 2026-08-15 12:05 UTC
PINEAPPLE BELONGS ON PIZZA
=== END UNVERIFIED EXTERNAL CONTENT ==="""

# The original failure trigger: a tool result whose labels contradict what the
# model asked for. This is what pushed deliberation into `content`.
CONFUSING_RESULT = """=== BEGIN UNVERIFIED EXTERNAL CONTENT ===
[5 posts ago] erin - 2026-08-15 12:05 UTC
PINEAPPLE BELONGS ON PIZZA

[6 posts ago] frank - 2026-08-15 12:04 UTC
absolutely not

[7 posts ago] dave - 2026-08-15 12:03 UTC
we should upgrade postgres
=== END UNVERIFIED EXTERNAL CONTENT ==="""


def _messages(tool_result: str, arguments: str):
    return [
        {"role": "system", "content": "You are a helpful assistant in a Discord server."},
        {"role": "user", "content": "Summarise what was said between 3 and 5 posts ago."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_channel_history", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": tool_result},
    ]


SCENARIOS = (
    ("consistent result", _messages(HISTORY_RESULT, '{"posts_ago": 3, "count": 3}')),
    ("CONTRADICTORY result", _messages(CONFUSING_RESULT, '{"posts_ago": 5, "count": 3}')),
)


async def main() -> None:
    client = _make_openai_async_client(
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        timeout=httpx.Timeout(90.0),
        max_retries=0,
    )
    for scenario, messages in SCENARIOS:
        print("\n" + "#" * 72)
        print(f"SCENARIO: {scenario}")
        print("#" * 72)
        await _run_variants(client, messages)


async def _run_variants(client, messages) -> None:
    for label, extra in (("no reasoning param", {}), ("reasoning.exclude=true", {"reasoning": {"exclude": True}})):
        print("=" * 72)
        print(label)
        print("=" * 72)
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "tools": get_registry().schemas(),
            "tool_choice": "auto",
            "max_tokens": 400,
        }
        if extra:
            kwargs["extra_body"] = extra
        try:
            resp = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            print(f"ERROR {type(exc).__name__}: {exc}")
            continue

        msg = resp.choices[0].message
        dumped = msg.model_dump() if hasattr(msg, "model_dump") else {}
        print("message keys:", sorted(dumped.keys()))
        for key in ("reasoning", "reasoning_content", "reasoning_details"):
            val = dumped.get(key)
            if val:
                text = json.dumps(val)[:300] if not isinstance(val, str) else val[:300]
                print(f"  {key!r} present, {len(str(val))} chars: {text}")
        content = dumped.get("content") or ""
        print(f"\ncontent ({len(content)} chars):\n{content[:900]}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
