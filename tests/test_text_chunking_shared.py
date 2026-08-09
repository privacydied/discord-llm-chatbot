"""The shared splitter is the single source of truth for Discord's size limit.

There used to be two independent Discord chunkers with different policies, so the
guarantee depended on which code path a reply took. These tests pin the contract
and the fact that both entry points resolve to the same implementation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from bot.action import BotAction
from bot.core.bot import _DISCORD_MAX_CONTENT_LEN, LLMBot
from bot.core.text_chunking import DISCORD_MAX_CONTENT_LEN, split_for_discord


class TestSplitterContract:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "short",
            "A" * DISCORD_MAX_CONTENT_LEN,
            "A" * (DISCORD_MAX_CONTENT_LEN + 1),
            "A" * (DISCORD_MAX_CONTENT_LEN * 3 + 7),
            "para\n\n" * 900,
            "no-boundaries-at-all" * 400,
            "```py\n" + ("code line\n" * 500) + "```",
            "émoji 🎥 " * 500,
        ],
    )
    def test_reassembly_is_byte_exact(self, text: str) -> None:
        """ "".join(parts) == text -- splitters that re-join on newlines mutate content."""
        assert "".join(split_for_discord(text)) == text

    @pytest.mark.parametrize(
        "text",
        [
            "A" * (DISCORD_MAX_CONTENT_LEN + 1),
            "para\n\n" * 900,
            "no-boundaries-at-all" * 400,
            "émoji 🎥 " * 500,
        ],
    )
    def test_every_part_fits_the_limit(self, text: str) -> None:
        assert all(len(part) <= DISCORD_MAX_CONTENT_LEN for part in split_for_discord(text))

    def test_no_part_is_empty(self) -> None:
        parts = split_for_discord("\n\n".join(["x" * 100] * 200))
        assert all(part for part in parts), "an empty part would be a rejected message"

    def test_limit_leaves_headroom_under_discords_hard_cap(self) -> None:
        """A mention prefix is prepended AFTER splitting, so parts can't sit at 2000."""
        assert DISCORD_MAX_CONTENT_LEN < 2000

    def test_none_and_non_string_are_tolerated(self) -> None:
        assert split_for_discord(None) == []  # type: ignore[arg-type]
        assert split_for_discord("") == []


class TestSingleSourceOfTruth:
    def test_bot_delegates_to_the_shared_splitter(self) -> None:
        text = "A" * (DISCORD_MAX_CONTENT_LEN + 500)
        assert LLMBot._chunk_message_content(MagicMock(), text) == split_for_discord(text)

    def test_bot_constant_is_the_shared_constant(self) -> None:
        assert _DISCORD_MAX_CONTENT_LEN is DISCORD_MAX_CONTENT_LEN

    def test_dead_competing_chunker_is_gone(self) -> None:
        """bot.utils.send_in_chunks split at 2000 with no headroom and re-joined on
        newlines. It had no callers; keeping it invited a second, divergent policy.
        """
        import bot.utils

        assert not hasattr(bot.utils, "send_in_chunks")
        assert not hasattr(bot.utils, "send_chunks")


class TestOverflowFileGate:
    """full_response.txt must not duplicate content already delivered in parts.

    The attachment threshold (2000) sat above the chunk threshold (1950), so it fired
    on every long reply -- users got the batched messages *and* a redundant text dump.
    """

    async def test_chunked_reply_attaches_no_overflow_file(self) -> None:
        bot = LLMBot(command_prefix="!", intents=discord.Intents.none(), config={})
        bot.enhanced_context_manager = None

        sends: list[dict] = []

        class _Typing:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

        channel = MagicMock()
        channel.typing = lambda: _Typing()

        async def _record(content=None, **kwargs):
            sends.append({"content": content or "", "files": kwargs.get("files")})
            return MagicMock(id=len(sends))

        channel.send = AsyncMock(side_effect=_record)

        message = MagicMock()
        message.id = 77
        message.content = "hi"
        message.reference = None
        message.channel = channel
        message.author = MagicMock(id=111, bot=False, mention="<@111>")
        message.guild = MagicMock(id=222)

        async def _record_reply(**kwargs):
            return await _record(**kwargs)

        message.reply = AsyncMock(side_effect=_record_reply)

        content = "A" * (DISCORD_MAX_CONTENT_LEN + 600)
        await bot._execute_action(message, BotAction(content=content, embeds=[]))

        assert len(sends) >= 2, "long reply must be batched"
        attached = [s for s in sends if s["files"]]
        assert attached == [], "chunked delivery must not also attach full_response.txt"
