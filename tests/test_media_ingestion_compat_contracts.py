from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.action import BotAction
from bot.media_ingestion import MediaIngestionManager, MediaIngestionResult
import bot.media_ingestion as media_ingestion


def _make_bot():
    bot = MagicMock()
    bot.config = {}
    bot.context_manager = AsyncMock()
    bot.context_manager.get_context_string.return_value = ""
    bot.system_prompts = {}
    bot.enhanced_context_manager = None
    bot.router = MagicMock()
    return bot


def test_media_ingestion_exposes_patchable_module_seams() -> None:
    for name in (
        "hear_infer_from_url",
        "brain_infer",
        "contextual_brain_infer_simple",
        "see_infer",
    ):
        assert hasattr(media_ingestion, name), f"missing module symbol: {name}"
        assert callable(getattr(media_ingestion, name)), f"symbol not callable: {name}"


@pytest.mark.asyncio
async def test_fallback_nonawaitable_router_path_uses_brain_fallback() -> None:
    bot = _make_bot()
    bot.router._invoke_text_flow = MagicMock(return_value=None)
    manager = MediaIngestionManager(bot)
    fallback_result = MediaIngestionResult(
        success=True,
        content="fallback article text",
        fallback_triggered=True,
        source_type="scrape",
    )
    message = MagicMock()
    message.id = 1
    expected = BotAction(content="brain fallback response")

    with patch(
        "bot.media_ingestion.brain_infer", new=AsyncMock(return_value=expected)
    ) as brain_mock:
        result = await manager._create_bot_action_from_fallback(
            fallback_result, message
        )

    assert result == expected
    brain_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_fallback_nonawaitable_router_path_returns_bot_action_directly() -> None:
    bot = _make_bot()
    routed = BotAction(content="router text flow")
    bot.router._invoke_text_flow = MagicMock(return_value=routed)
    manager = MediaIngestionManager(bot)
    fallback_result = MediaIngestionResult(
        success=True,
        content="fallback article text",
        fallback_triggered=True,
        source_type="scrape",
    )
    message = MagicMock()
    message.id = 2

    result = await manager._create_bot_action_from_fallback(fallback_result, message)

    assert result == routed
