"""Paywall thin-content cascade test for the Router web_scrape path. [PAY]

Regression test for: a paywalled page returns HTTP 200 with only a short
teaser (e.g. 122 chars). process_url treats that as success, so the tiered
extractor (Tier C reader proxy) was never reached via the error/empty
fallbacks -- the bot answered "paywall blocked it". The fix retries through
web_extractor.extract() (which cascades A->B->C) when process_url content is
thin. This test exercises that branch in _handle_general_url.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from bot.modality import InputItem
from bot.router import Router

_TIMES_URL = (
    "https://www.thetimes.com/life-style/sex-relationships/"
    "article/why-men-wont-date-sklh6t2xg"
)


@pytest.fixture
def mock_bot():
    bot = MagicMock(spec=discord.Client)
    bot.user = MagicMock()
    bot.user.id = 12345
    bot.config = {}
    bot.tts_manager = AsyncMock()
    bot.brain = AsyncMock()
    bot.loop = AsyncMock()
    return bot


@pytest.fixture
def router(mock_bot):
    return Router(bot=mock_bot, logger=MagicMock(spec=logging.Logger))

_TIMES_URL = (
    "https://www.thetimes.com/life-style/sex-relationships/"
    "article/why-men-wont-date-sklh6t2xg"
)


@pytest.mark.asyncio
async def test_handle_general_url_cascades_to_reader_on_thin_process_url(
    router,
) -> None:
    thin_payload = {
        "text": "Short teaser only. Subscribe to read the full article.",
        "screenshot_path": None,
    }
    full_article = (
        "# Ghosting, pressure, the cost of dinner -- I know why men won't date\n\n"
        "Male friends I know say they feel cornered by women wanting to settle down."
    )

    with patch(
        "bot.router.process_url", new=AsyncMock(return_value=thin_payload)
    ), patch(
        "bot.url_classifier.classify_url",
        new=AsyncMock(return_value=MagicMock(bucket=MagicMock(name="OTHER"))),
    ), patch(
        "bot.router.web_extractor"
    ) as mock_ex:
        mock_ex.extract = AsyncMock(
            return_value=MagicMock(
                success=True,
                tier_used="C",
                canonical_url=_TIMES_URL,
                text=full_article,
                error=None,
            )
        )
        # _finalize_extraction wraps the tiered result; make it return the body.
        router._finalize_extraction = AsyncMock(return_value=full_article)

        item = InputItem(source_type="url", payload=_TIMES_URL, order_index=0)
        result = await router._handle_general_url(item, message=None)

    assert result == full_article
    mock_ex.extract.assert_awaited_once()
