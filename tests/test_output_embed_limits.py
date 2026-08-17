"""bot.core.output enforces Discord's embed size limits at the send boundary.

Before this, `_sanitize_embeds` only ran leakage sanitization -- a >4096-char
description, a >1024-char field value, more than 25 fields, or fields summing
past Discord's 6000-char aggregate cap all still reached `destination.send()`
verbatim and came back as HTTP 400 / error 50035 ("Invalid Form Body").
Truncation existed only as per-caller duplicates (rag_commands.py,
admin_alert_manager.py); this pins the boundary itself as safe regardless of
what a caller builds.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from bot.core.output import (
    _EMBED_DESCRIPTION_LIMIT,
    _EMBED_FIELD_VALUE_LIMIT,
    _EMBED_MAX_FIELDS,
    _EMBED_TOTAL_LIMIT,
    safe_send,
)


@pytest.fixture(autouse=True)
def _identity_sanitizer(monkeypatch):
    monkeypatch.setattr("bot.core.output.sanitize_public_text", lambda text: text)


def _embed_total_len(embed: discord.Embed) -> int:
    total = len(embed.title or "") + len(embed.description or "")
    if embed.footer and embed.footer.text:
        total += len(embed.footer.text)
    for f in embed.fields or []:
        total += len(f.name or "") + len(f.value or "")
    return total


class TestEmbedSizeEnforcement:
    async def test_oversize_description_is_truncated(self) -> None:
        sent: dict = {}

        async def _send(content=None, **kwargs):
            sent.update(kwargs)
            return MagicMock()

        destination = MagicMock()
        destination.send = AsyncMock(side_effect=_send)

        embed = discord.Embed(title="t", description="D" * (_EMBED_DESCRIPTION_LIMIT + 500))
        await safe_send(destination, "hi", embed=embed)

        assert len(sent["embed"].description) <= _EMBED_DESCRIPTION_LIMIT

    async def test_oversize_field_value_is_truncated(self) -> None:
        sent: dict = {}

        async def _send(content=None, **kwargs):
            sent.update(kwargs)
            return MagicMock()

        destination = MagicMock()
        destination.send = AsyncMock(side_effect=_send)

        embed = discord.Embed(title="t")
        embed.add_field(name="f", value="V" * (_EMBED_FIELD_VALUE_LIMIT + 200), inline=False)
        await safe_send(destination, "hi", embed=embed)

        assert all(len(f.value) <= _EMBED_FIELD_VALUE_LIMIT for f in sent["embed"].fields)

    async def test_too_many_fields_is_capped(self) -> None:
        sent: dict = {}

        async def _send(content=None, **kwargs):
            sent.update(kwargs)
            return MagicMock()

        destination = MagicMock()
        destination.send = AsyncMock(side_effect=_send)

        embed = discord.Embed(title="t")
        for i in range(_EMBED_MAX_FIELDS + 10):
            embed.add_field(name=f"f{i}", value=f"v{i}", inline=True)
        await safe_send(destination, "hi", embed=embed)

        assert len(sent["embed"].fields) <= _EMBED_MAX_FIELDS

    async def test_aggregate_total_is_enforced_even_when_each_field_is_individually_legal(self) -> None:
        """25 fields at ~300 chars each are each under the 1024 per-field limit
        but sum well past the 6000 aggregate cap -- this is the case per-field
        truncation alone cannot catch.
        """
        sent: dict = {}

        async def _send(content=None, **kwargs):
            sent.update(kwargs)
            return MagicMock()

        destination = MagicMock()
        destination.send = AsyncMock(side_effect=_send)

        embed = discord.Embed(title="t")
        for i in range(_EMBED_MAX_FIELDS):
            embed.add_field(name=f"field {i}", value="x" * 300, inline=False)
        assert _embed_total_len(embed) > _EMBED_TOTAL_LIMIT, "fixture must actually exceed the aggregate cap"

        await safe_send(destination, "hi", embed=embed)

        assert _embed_total_len(sent["embed"]) <= _EMBED_TOTAL_LIMIT

    async def test_legal_embed_is_untouched(self) -> None:
        sent: dict = {}

        async def _send(content=None, **kwargs):
            sent.update(kwargs)
            return MagicMock()

        destination = MagicMock()
        destination.send = AsyncMock(side_effect=_send)

        embed = discord.Embed(title="fine", description="also fine")
        embed.add_field(name="f", value="v", inline=True)
        await safe_send(destination, "hi", embed=embed)

        assert sent["embed"].title == "fine"
        assert sent["embed"].description == "also fine"
        assert sent["embed"].fields[0].value == "v"

    async def test_embeds_plural_list_is_also_enforced(self) -> None:
        sent: dict = {}

        async def _send(content=None, **kwargs):
            sent.update(kwargs)
            return MagicMock()

        destination = MagicMock()
        destination.send = AsyncMock(side_effect=_send)

        big = discord.Embed(title="t", description="D" * (_EMBED_DESCRIPTION_LIMIT + 500))
        await safe_send(destination, "hi", embeds=[big])

        assert len(sent["embeds"][0].description) <= _EMBED_DESCRIPTION_LIMIT
