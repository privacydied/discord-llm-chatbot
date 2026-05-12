"""Tests for bot.core.output — explicit safe Discord outbound helpers."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bot.core.output import (
    safe_send,
    safe_reply,
    safe_edit,
    _maybe_sanitize_text,
    _sanitize_embeds,
)


class TestMaybeSanitizeText:
    """Unit tests for the internal _maybe_sanitize_text helper."""

    def test_empty_passthrough(self):
        assert _maybe_sanitize_text("") == ""
        assert _maybe_sanitize_text("   ") == "   "

    def test_none_passthrough(self):
        assert _maybe_sanitize_text(None) is None  # type: ignore[arg-type]

    def test_calls_sanitizer(self):
        with patch("bot.core.output.sanitize_public_text") as mock_sanitize:
            mock_sanitize.return_value = "sanitized"
            result = _maybe_sanitize_text("  raw  ")
            mock_sanitize.assert_called_once_with("  raw  ")
            assert result == "sanitized"


class TestSanitizeEmbeds:
    """Unit tests for the internal _sanitize_embeds helper."""

    def test_no_embed_returns_none_none(self):
        assert _sanitize_embeds(None, None) == (None, None)

    def test_embed_only(self):
        mock_embed = MagicMock()
        with patch("bot.core.output.sanitize_embed_for_public") as mock_san:
            mock_san.return_value = mock_embed
            sanitized_embed, sanitized_embeds = _sanitize_embeds(mock_embed, None)
        assert sanitized_embed is mock_embed
        assert sanitized_embeds is None

    def test_embeds_list(self):
        with patch("bot.core.output.sanitize_embed_collection_for_public") as mock_col:
            mock_col.return_value = [MagicMock()]
            sanitized_embed, sanitized_embeds = _sanitize_embeds(
                None, [MagicMock()]
            )
        assert sanitized_embed is None
        assert sanitized_embeds is not None
        assert len(sanitized_embeds) == 1

    def test_both_embed_and_embeds_normalizes_to_embeds(self):
        mock_embed = MagicMock()
        mock_embed2 = MagicMock()
        with (
            patch("bot.core.output.sanitize_embed_for_public") as mock_one,
            patch("bot.core.output.sanitize_embed_collection_for_public") as mock_col,
        ):
            mock_one.return_value = mock_embed
            mock_col.return_value = [mock_embed2]
            sanitized_embed, sanitized_embeds = _sanitize_embeds(
                mock_embed, [mock_embed2]
            )
        assert sanitized_embed is None
        assert sanitized_embeds == [mock_embed, mock_embed2]


@pytest.mark.asyncio
class TestSafeSend:
    """Integration-like tests for safe_send with mocked Discord."""

    async def test_safe_send_sanitizes_content(self):
        dest = AsyncMock()
        dest.send = AsyncMock()
        with patch("bot.core.output.sanitize_public_text", return_value="safe"):
            await safe_send(dest, content="  raw  ")
        dest.send.assert_awaited_once()
        call_args = dest.send.call_args
        assert call_args[0][0] == "safe" or call_args.kwargs.get("content") == "safe"

    async def test_safe_send_with_none_content(self):
        dest = AsyncMock()
        dest.send = AsyncMock()
        await safe_send(dest, content=None)
        dest.send.assert_awaited_once()


@pytest.mark.asyncio
class TestSafeReply:
    """Integration-like tests for safe_reply with mocked Discord."""

    async def test_safe_reply_sanitizes_content(self):
        msg = MagicMock()
        msg.reply = AsyncMock()
        with patch("bot.core.output.sanitize_public_text", return_value="safe"):
            await safe_reply(msg, content="  raw  ")
        msg.reply.assert_awaited_once()
        call_args = msg.reply.call_args
        # content is passed as first positional arg
        assert call_args.args[0] == "safe"


@pytest.mark.asyncio
class TestSafeEdit:
    """Integration-like tests for safe_edit with mocked Discord."""

    async def test_safe_edit_sanitizes_content(self):
        msg = MagicMock()
        msg.edit = AsyncMock()
        with patch("bot.core.output.sanitize_public_text", return_value="safe"):
            await safe_edit(msg, content="  raw  ")
        msg.edit.assert_awaited_once()
