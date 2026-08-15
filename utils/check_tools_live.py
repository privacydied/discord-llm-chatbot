"""Live end-to-end check of the tool loop against the configured model. [REH]

Exercises the real OpenRouter model with real tool schemas, using a fake
Discord channel so no guild is needed.

Usage: uv run python utils/check_tools_live.py
"""

import asyncio
import os
from datetime import UTC, datetime

from dotenv import load_dotenv

load_dotenv("/volume1/py/discord-llm-chatbot/.env")

from bot.tools import ToolContext  # noqa: E402
from bot.tools.inference import run_tool_conversation  # noqa: E402


class _Author:
    def __init__(self, name):
        self.display_name = name
        self.name = name


class _Msg:
    def __init__(self, content, author, minute):
        self.content = content
        self.author = _Author(author)
        self.created_at = datetime(2026, 8, 15, 12, minute, tzinfo=UTC)


class _Channel:
    """Newest-first, like discord.py."""

    def __init__(self, messages):
        self._messages = messages

    def history(self, limit=None, before=None):
        msgs = self._messages[:limit]

        async def _gen():
            for m in msgs:
                yield m

        return _gen()


def _ctx(cfg):
    history = [
        _Msg("and that's why I switched to nginx", "dave", 9),
        _Msg("has anyone tried the new caddy release?", "erin", 8),
        _Msg("lunch?", "frank", 7),
        _Msg("the deploy finished, all green", "dave", 6),
        _Msg("PINEAPPLE BELONGS ON PIZZA", "erin", 5),
        _Msg("absolutely not", "frank", 4),
        _Msg("we should upgrade postgres", "dave", 3),
        _Msg("agreed, 17 has been solid", "erin", 2),
        _Msg("ok I'll open a ticket", "frank", 1),
        _Msg("done: TICKET-4821", "dave", 0),
    ]
    current = _Msg("current message", "me", 10)
    current.channel = _Channel(history)
    return ToolContext(message=current, bot=None, config=cfg)


CASES = [
    "What time is it right now?",
    "What did the message 5 posts ago say?",
    "Summarise what was said between 3 and 5 posts ago.",
]


async def main() -> None:
    cfg = {
        "TOOLS_ENABLED": True,
        "TOOLS_MAX_ITERATIONS": 3,
        "TOOLS_TIMEOUT_S": 60.0,
        "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_TEXT_MODEL": os.getenv("OPENAI_TEXT_MODEL"),
        "MAX_RESPONSE_TOKENS": 400,
    }
    print("model:", cfg["OPENAI_TEXT_MODEL"])
    print("base :", cfg["OPENAI_API_BASE"])
    print("=" * 70)

    for question in CASES:
        print(f"\nQ: {question}")
        try:
            answer = await run_tool_conversation(
                prompt=question,
                ctx=_ctx(cfg),
                system_prompt="You are a helpful assistant in a Discord server. Use the tools available to you.",
                cfg=cfg,
            )
        except Exception as exc:
            print(f"   EXCEPTION: {type(exc).__name__}: {exc}")
            continue
        print(f"A: {answer if answer else '(None — fell back to normal flow)'}")


if __name__ == "__main__":
    asyncio.run(main())
