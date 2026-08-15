"""Live check: reproduce the reasoning leak, then verify recovery. [REH]

Drives the real detection and recovery code in bot/tools/inference.py against
the configured model, using the conversation state that reproduced the leak.

Usage: uv run python utils/check_reasoning_recovery.py
"""

import asyncio
import os

import httpx
from dotenv import load_dotenv

load_dotenv("/volume1/py/discord-llm-chatbot/.env")

from bot.openai_backend import _make_openai_async_client  # noqa: E402
from bot.tools import get_registry  # noqa: E402
from bot.tools import inference  # noqa: E402

MODEL = os.getenv("OPENAI_TEXT_MODEL")

# Tool output whose labels contradict the arguments the model chose. This is
# what pushed the model into unbounded deliberation.
CONTRADICTORY_RESULT = """=== BEGIN UNVERIFIED EXTERNAL CONTENT ===
[5 posts ago] erin - 2026-08-15 12:05 UTC
PINEAPPLE BELONGS ON PIZZA

[6 posts ago] frank - 2026-08-15 12:04 UTC
absolutely not

[7 posts ago] dave - 2026-08-15 12:03 UTC
we should upgrade postgres
=== END UNVERIFIED EXTERNAL CONTENT ==="""

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant in a Discord server."},
    {"role": "user", "content": "Summarise what was said between 3 and 5 posts ago."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_channel_history", "arguments": '{"posts_ago": 5, "count": 3}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": CONTRADICTORY_RESULT},
]


async def main() -> None:
    client = _make_openai_async_client(
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        timeout=httpx.Timeout(90.0),
        max_retries=0,
    )
    print(f"model: {MODEL}\n")

    response = await client.chat.completions.create(
        model=MODEL,
        messages=MESSAGES,
        tools=get_registry().schemas(),
        tool_choice="auto",
        max_tokens=400,
    )
    content, reasoning = inference._extract(response)
    leaked = inference._is_reasoning_leak(content, reasoning)

    print(f"content  : {len(content)} chars")
    print(f"reasoning: {len(reasoning)} chars")
    print(f"LEAK DETECTED: {leaked}")
    print("-" * 70)
    print("what the user WOULD have seen without the fix:")
    print((content[:400] + "…") if content else "(empty)")
    print("-" * 70)

    if not leaked and content:
        print("no leak this run; the model concluded on its own:")
        print(content)
        return

    recovered = await inference._force_answer(client, MODEL, MESSAGES, 400, {})
    print("what the user sees WITH the fix:")
    print(recovered if recovered else "(None — falls back to the normal text flow)")


if __name__ == "__main__":
    asyncio.run(main())
