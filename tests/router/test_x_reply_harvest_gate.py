from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import bot.router as router_mod
from bot.modality import ImageRef, InputModality
from bot.router import BotAction, Router


class DummyBot:
    def __init__(self) -> None:
        self.user = SimpleNamespace(id=12345, mention="<@12345>")
        self.config = {
            "HYBRID_FORCE_PERCEPTION_ON_REPLY": True,
            "VISION_REPLY_IMAGE_HARVEST": True,
        }
        self.tts_manager = None
        self.loop = None
        self.system_prompts = {"vl_prompt": None}
        self.enhanced_context_manager = None


@pytest.mark.asyncio
async def test_harvested_reply_x_url_skips_reply_perception_gate(monkeypatch) -> None:
    bot = DummyBot()
    router = Router(bot)

    ref_msg = MagicMock()
    ref_msg.id = 9001
    ref_msg.content = "https://x.com/user/status/1234567890123456789"
    ref_msg.attachments = []
    ref_msg.embeds = []

    msg = MagicMock()
    msg.id = 9002
    msg.content = "<@12345> thoughts?"
    msg.attachments = []
    msg.embeds = []
    msg.mentions = [bot.user]
    msg.reference = SimpleNamespace(message_id=ref_msg.id, resolved=ref_msg)
    msg.channel = MagicMock()
    msg.channel.fetch_message = AsyncMock(return_value=ref_msg)
    msg.author = SimpleNamespace(id=111, bot=False)

    monkeypatch.setattr(router_mod, "collect_input_items", lambda _m: [])

    def _fake_collect_images(m):
        if getattr(m, "id", None) == ref_msg.id:
            return [ImageRef(url="https://pbs.twimg.com/media/test.jpg")]
        return []

    monkeypatch.setattr(router_mod, "collect_image_urls_from_message", _fake_collect_images)

    async def _fake_map_item_to_modality(item):
        payload = str(getattr(item, "payload", ""))
        if "x.com/" in payload or "twitter.com/" in payload:
            return InputModality.GENERAL_URL
        return InputModality.SINGLE_IMAGE

    monkeypatch.setattr(router_mod, "map_item_to_modality", _fake_map_item_to_modality)

    class _RetryManager:
        async def run_with_fallback(self, *, modality, coro_factory, per_item_budget):
            return SimpleNamespace(
                success=True,
                total_time=0.01,
                result=f"ok:{modality}",
                attempts=1,
                fallback_occurred=False,
                error=None,
            )

    monkeypatch.setattr(router_mod, "get_retry_manager", _RetryManager)

    router._prioritized_vision_route = AsyncMock(return_value=None)
    router._run_perception_notes = AsyncMock(return_value=("notes", None))
    router._invoke_text_flow = AsyncMock(return_value=BotAction(content="done"))

    action = await router._process_multimodal_message_internal(msg, "ctx")

    router._run_perception_notes.assert_not_awaited()
    router._invoke_text_flow.assert_awaited_once()
    assert isinstance(action, BotAction)
    assert action.content == "done"

    prompt_arg = router._invoke_text_flow.await_args.args[0]
    assert "ok:" in prompt_arg
