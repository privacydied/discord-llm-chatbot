"""bot.core.output enforces Discord's 2000-char limit and ordered delivery.

Before this, safe_send/safe_reply/safe_edit only sanitized -- despite the module
docstring claiming to be the outbound boundary -- so any caller handing them
unbounded LLM output, transcript text or exception strings produced a hard HTTP 400
(JSON error 50035).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from bot.core.output import safe_edit, safe_reply, safe_send
from bot.core.text_chunking import DISCORD_MAX_CONTENT_LEN, split_for_discord


class _Recorder:
    """Records send order and flags any overlapping (concurrent) sends."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.overlaps: list[str] = []
        self._in_flight = 0

    async def send(self, content=None, **_kwargs):
        self._in_flight += 1
        if self._in_flight > 1:
            self.overlaps.append((content or "")[:20])
        self.parts.append(content or "")
        await asyncio.sleep(0)  # a concurrent caller would interleave here
        self._in_flight -= 1
        return MagicMock(id=len(self.parts))


@pytest.fixture(autouse=True)
def _no_inter_part_delay(monkeypatch):
    """Keep the 0.3s rate-limit gap from slowing the suite."""
    monkeypatch.setattr("bot.core.output._INTER_PART_DELAY_S", 0)


@pytest.fixture(autouse=True)
def _identity_sanitizer(monkeypatch):
    """Isolate chunking from sanitization."""
    monkeypatch.setattr("bot.core.output.sanitize_public_text", lambda text: text)


def _oversize() -> str:
    return ("A" * (DISCORD_MAX_CONTENT_LEN + 500)) + "\n\ntail"


class TestSafeSend:
    async def test_short_content_sends_once(self) -> None:
        rec = _Recorder()
        destination = MagicMock()
        destination.send = AsyncMock(side_effect=rec.send)

        await safe_send(destination, "short")

        assert rec.parts == ["short"]

    async def test_oversize_content_is_split_and_ordered(self) -> None:
        rec = _Recorder()
        destination = MagicMock()
        destination.send = AsyncMock(side_effect=rec.send)
        content = _oversize()

        await safe_send(destination, content)

        assert len(rec.parts) >= 2, "oversize content must be batched"
        assert rec.parts == split_for_discord(content), "parts must be in splitter order"
        assert "".join(rec.parts) == content, "content must survive reassembly byte-exact"
        assert all(len(p) <= DISCORD_MAX_CONTENT_LEN for p in rec.parts)
        assert rec.overlaps == [], "sends must not overlap, or Discord may reorder them"

    async def test_attachments_ride_only_on_the_first_part(self) -> None:
        """A file repeated per part would post the same attachment several times."""
        calls: list[dict] = []

        async def _send(content=None, **kwargs):
            calls.append({"content": content, **kwargs})
            return MagicMock()

        destination = MagicMock()
        destination.send = AsyncMock(side_effect=_send)
        sentinel_file = MagicMock(name="file")

        await safe_send(destination, _oversize(), file=sentinel_file)

        assert calls[0]["file"] is sentinel_file
        assert len(calls) >= 2
        for call in calls[1:]:
            assert "file" not in call or call.get("file") is None

    async def test_empty_content_still_sends_embeds(self) -> None:
        """An embed-only payload must not be dropped by the split path."""
        rec = _Recorder()
        destination = MagicMock()
        destination.send = AsyncMock(side_effect=rec.send)

        await safe_send(destination, None, embed=discord.Embed(title="alpha"))

        assert len(rec.parts) == 1

    async def test_returns_last_message(self) -> None:
        destination = MagicMock()
        destination.send = AsyncMock(side_effect=_Recorder().send)

        result = await safe_send(destination, _oversize())

        assert result is not None


class TestSafeReply:
    async def test_only_first_part_is_a_reply(self) -> None:
        """Continuations go to the channel so the user isn't pinged once per part."""
        rec = _Recorder()
        message = MagicMock()
        message.reply = AsyncMock(side_effect=rec.send)
        message.channel = MagicMock()
        message.channel.send = AsyncMock(side_effect=rec.send)
        content = _oversize()

        await safe_reply(message, content)

        assert message.reply.await_count == 1
        assert message.channel.send.await_count == len(split_for_discord(content)) - 1
        assert rec.parts == split_for_discord(content)
        assert rec.overlaps == []


class TestSafeEdit:
    async def test_oversize_edit_appends_followups_in_order(self) -> None:
        rec = _Recorder()
        message = MagicMock()
        message.channel = MagicMock()
        message.channel.send = AsyncMock(side_effect=rec.send)
        edited = MagicMock()

        async def _edit(content=None, **_kwargs):
            rec.parts.append(content or "")
            return edited

        message.edit = AsyncMock(side_effect=_edit)
        content = _oversize()

        result = await safe_edit(message, content)

        expected = split_for_discord(content)
        assert rec.parts == expected, "edit gets part 1, follow-ups get the rest in order"
        assert result is edited, "callers hold the edited message handle, not the last follow-up"

    async def test_short_edit_makes_no_followups(self) -> None:
        message = MagicMock()
        message.channel = MagicMock()
        message.channel.send = AsyncMock()
        message.edit = AsyncMock(return_value=MagicMock())

        await safe_edit(message, "short")

        message.channel.send.assert_not_awaited()
