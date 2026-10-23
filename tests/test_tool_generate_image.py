"""generate_image tool: natural requests reach the image pipeline [CA][REH]."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from bot.tools import ToolContext
from bot.tools.registry import ALLOWED_TOOL_NAMES, execute_tool, get_registry


def _ctx(router):
    return ToolContext(message=SimpleNamespace(id=1), bot=SimpleNamespace(router=router), config={})


def test_registered_and_offered():
    assert "generate_image" in ALLOWED_TOOL_NAMES
    assert "generate_image" in {s["function"]["name"] for s in get_registry().schemas()}


async def test_starts_job_with_prompt_and_shape():
    router = SimpleNamespace(start_image_generation=AsyncMock(return_value=(True, "abcd1234")))
    result = await execute_tool("generate_image", {"prompt": "  a 32 year old  man in carhartt  ", "shape": "portrait"}, _ctx(router))
    assert result.ok and "will be posted" in result.content
    args, kwargs = router.start_image_generation.await_args
    assert args[0] == "a 32 year old man in carhartt"
    assert (kwargs["width"], kwargs["height"]) == (832, 1216)


async def test_refusal_is_reported_to_model():
    router = SimpleNamespace(start_image_generation=AsyncMock(return_value=(False, "image generation is turned off on this server")))
    result = await execute_tool("generate_image", {"prompt": "a cat"}, _ctx(router))
    assert not result.ok and "turned off" in (result.error or "")


async def test_needs_prompt_and_router():
    assert not (await execute_tool("generate_image", {"prompt": " "}, _ctx(SimpleNamespace()))).ok
    assert not (await execute_tool("generate_image", {"prompt": "a cat"}, ToolContext())).ok


async def test_router_start_respects_server_toggle(monkeypatch):
    from bot.router import Router

    router = Router.__new__(Router)
    router._vision_orchestrator = None
    monkeypatch.setattr("bot.router.is_server_feature_enabled", lambda gid, feat: False)
    message = SimpleNamespace(id=5, guild=SimpleNamespace(id=9), author=SimpleNamespace(id=2), channel=SimpleNamespace(id=3))
    started, detail = await router.start_image_generation("a cat", message)
    assert not started and "turned off" in detail
