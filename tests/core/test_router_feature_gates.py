from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import bot.router as router_module
from bot.router import ResponseMessage, Router


@pytest.fixture(autouse=True)
def disable_router_pdf_support(monkeypatch) -> None:
    monkeypatch.setattr(router_module, "PDF_SUPPORT", False, raising=False)


@pytest.fixture
def router_bot():
    bot = SimpleNamespace()
    bot.config = {
        "VISION_ENABLED": False,
        "BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO": False,
    }
    bot.tts_manager = SimpleNamespace()
    bot.user = SimpleNamespace(id=999)
    bot.vision_orchestrator = None
    bot.loop = None
    return bot


@pytest.fixture
def router(router_bot):
    return Router(router_bot)


@pytest.fixture
def guild_message():
    return SimpleNamespace(
        id=1,
        content="<@999> !img a robot in space",
        guild=SimpleNamespace(id=123),
        channel=SimpleNamespace(id=456),
        author=SimpleNamespace(bot=False, id=42),
        attachments=[],
    )


@pytest.mark.asyncio
async def test_feature_gate_blocks_image_generation_before_command_handler(router, guild_message, monkeypatch) -> None:
    router._handle_img_command = AsyncMock()
    monkeypatch.setattr(
        "bot.router.is_server_feature_enabled",
        lambda guild_id, feature: feature != "image_generation",
    )

    result = await router.dispatch_message(guild_message)

    assert isinstance(result, ResponseMessage)
    assert "disabled on this server" in result.text.lower()
    router._handle_img_command.assert_not_called()


@pytest.mark.asyncio
async def test_feature_gate_blocks_disabled_vision_before_attachment_flow(router, monkeypatch) -> None:
    router._flows["process_attachments"] = AsyncMock()
    message = SimpleNamespace(
        id=2,
        content="",
        guild=SimpleNamespace(id=123),
        channel=SimpleNamespace(id=456),
        author=SimpleNamespace(bot=False, id=42),
        attachments=[SimpleNamespace(content_type="image/png", filename="screenshot.png")],
    )
    monkeypatch.setattr(
        "bot.router.is_server_feature_enabled",
        lambda guild_id, feature: feature != "vision",
    )

    result = await router.dispatch_message(message)

    assert isinstance(result, ResponseMessage)
    assert "vision is disabled" in result.text.lower()
    router._flows["process_attachments"].assert_not_called()


@pytest.mark.asyncio
async def test_feature_gate_blocks_disabled_stt_before_attachment_flow(router, monkeypatch) -> None:
    router._flows["process_attachments"] = AsyncMock()
    message = SimpleNamespace(
        id=3,
        content="",
        guild=SimpleNamespace(id=123),
        channel=SimpleNamespace(id=456),
        author=SimpleNamespace(bot=False, id=42),
        attachments=[SimpleNamespace(content_type="audio/wav", filename="clip.wav")],
    )
    monkeypatch.setattr(
        "bot.router.is_server_feature_enabled",
        lambda guild_id, feature: feature != "stt",
    )

    result = await router.dispatch_message(message)

    assert isinstance(result, ResponseMessage)
    assert "audio/video transcription is disabled" in result.text.lower()
    router._flows["process_attachments"].assert_not_called()


@pytest.mark.asyncio
async def test_feature_gate_blocks_disabled_web_extraction_before_url_flow(router, monkeypatch) -> None:
    router._flows["process_url"] = AsyncMock()
    message = SimpleNamespace(
        id=4,
        content="check this https://example.com/article",
        guild=SimpleNamespace(id=123),
        channel=SimpleNamespace(id=456),
        author=SimpleNamespace(bot=False, id=42),
        attachments=[],
    )
    monkeypatch.setattr(
        "bot.router.is_server_feature_enabled",
        lambda guild_id, feature: feature != "web_extraction",
    )

    result = await router.dispatch_message(message)

    assert isinstance(result, ResponseMessage)
    assert "web extraction is disabled" in result.text.lower()
    router._flows["process_url"].assert_not_called()


@pytest.mark.asyncio
async def test_feature_gate_blocks_disabled_x_extraction_before_url_flow(router, monkeypatch) -> None:
    router._flows["process_url"] = AsyncMock()

    class _Typing:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    message = SimpleNamespace(
        id=5,
        content="check this https://x.com/u/status/1",
        guild=SimpleNamespace(id=123),
        channel=SimpleNamespace(id=456, typing=_Typing),
        author=SimpleNamespace(bot=False, id=42),
        attachments=[],
    )
    monkeypatch.setattr(
        "bot.router.is_server_feature_enabled",
        lambda guild_id, feature: feature != "x_twitter_extraction",
    )
    monkeypatch.setattr(router, "_is_twitter_status_url", lambda url: True)

    result = await router.dispatch_message(message)

    assert isinstance(result, ResponseMessage)
    assert "x/twitter extraction is disabled" in result.text.lower()
    router._flows["process_url"].assert_not_called()
