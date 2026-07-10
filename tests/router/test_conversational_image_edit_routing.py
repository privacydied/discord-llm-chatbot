"""End-to-end routing tests: an addressed message with an image + an edit
instruction must resolve the image and call the img2img path - not VL
analysis, not text-to-image - while analysis questions keep going to VL
unchanged. Mirrors the mocking recipe in test_x_reply_harvest_gate.py.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import bot.router as router_mod
from bot.modality import ImageRef, InputModality
from bot.router import BotAction, Router
from bot.router_components import ResolvedEditImage


class DummyBot:
    def __init__(self) -> None:
        self.user = SimpleNamespace(id=12345, mention="<@12345>")
        self.config = {
            "HYBRID_FORCE_PERCEPTION_ON_REPLY": True,
            "VISION_REPLY_IMAGE_HARVEST": True,
            "VISION_ENABLED": True,
            "VISION_CONVERSATIONAL_EDIT_ENABLED": True,
        }
        self.tts_manager = None
        self.loop = None
        self.system_prompts = {"vl_prompt": None}
        self.enhanced_context_manager = None


def _wire_common_mocks(monkeypatch, *, image_owner_id):
    """Common plumbing so _process_multimodal_message_internal reaches the
    routing decision point without touching unrelated subsystems."""
    monkeypatch.setattr(router_mod, "collect_input_items", lambda _m: [])

    def _fake_collect_images(m):
        if getattr(m, "id", None) == image_owner_id:
            return [ImageRef(url="https://cdn.example/photo.png", content_type="image/png")]
        return []

    monkeypatch.setattr(router_mod, "collect_image_urls_from_message", _fake_collect_images)

    async def _fake_map_item_to_modality(item):
        return InputModality.SINGLE_IMAGE

    monkeypatch.setattr(router_mod, "map_item_to_modality", _fake_map_item_to_modality)

    class _RetryManager:
        async def run_with_fallback(self, *, modality, coro_factory, per_item_budget):
            return SimpleNamespace(success=True, total_time=0.01, result="ok", attempts=1, fallback_occurred=False, error=None)

    monkeypatch.setattr(router_mod, "get_retry_manager", _RetryManager)


def _reply_message(bot, *, content: str):
    ref_msg = MagicMock()
    ref_msg.id = 9001
    ref_msg.content = ""
    ref_msg.attachments = []
    ref_msg.embeds = []

    msg = MagicMock()
    msg.id = 9002
    msg.content = content
    msg.attachments = []
    msg.embeds = []
    msg.mentions = [bot.user]
    msg.reference = SimpleNamespace(message_id=ref_msg.id, resolved=ref_msg)
    msg.channel = MagicMock()
    msg.channel.fetch_message = AsyncMock(return_value=ref_msg)
    msg.author = SimpleNamespace(id=111, bot=False)
    msg.guild = None
    return msg, ref_msg


def _attachment_message(bot, *, content: str):
    msg = MagicMock()
    msg.id = 9003
    msg.content = content
    msg.attachments = ["fake_attachment"]
    msg.embeds = []
    msg.mentions = [bot.user]
    msg.reference = None
    msg.channel = MagicMock()
    msg.channel.fetch_message = AsyncMock()
    msg.author = SimpleNamespace(id=111, bot=False)
    msg.guild = None
    return msg


@pytest.mark.asyncio
async def test_reply_image_with_edit_instruction_routes_to_edit_not_vl(monkeypatch) -> None:
    bot = DummyBot()
    router = Router(bot)
    router._vision_orchestrator = MagicMock()  # non-None sentinel; job path is mocked below

    msg, ref_msg = _reply_message(bot, content="<@12345> give this man a beard")
    _wire_common_mocks(monkeypatch, image_owner_id=ref_msg.id)

    resolved = ResolvedEditImage(data=b"img-bytes", content_type="image/png", source="reply")
    monkeypatch.setattr(router_mod, "resolve_edit_source_image", AsyncMock(return_value=resolved))

    router._prioritized_vision_route = AsyncMock(return_value=None)
    router._run_perception_notes = AsyncMock(return_value=("notes", None))
    router._invoke_text_flow = AsyncMock(return_value=BotAction(content="should not be used"))
    router._run_conversational_edit_job = AsyncMock(return_value=BotAction(content="", files=["fake-file"]))

    action = await router._process_multimodal_message_internal(msg, "ctx")

    router._run_conversational_edit_job.assert_awaited_once()
    call_args = router._run_conversational_edit_job.await_args.args
    assert call_args[0] is msg
    assert "beard" in call_args[1]
    assert call_args[2] is resolved

    router._run_perception_notes.assert_not_awaited()
    router._invoke_text_flow.assert_not_awaited()
    assert action.files == ["fake-file"]


@pytest.mark.asyncio
async def test_attachment_on_message_with_edit_instruction_routes_to_edit(monkeypatch) -> None:
    bot = DummyBot()
    router = Router(bot)
    router._vision_orchestrator = MagicMock()

    msg = _attachment_message(bot, content="<@12345> remove the background")
    _wire_common_mocks(monkeypatch, image_owner_id=msg.id)

    resolved = ResolvedEditImage(data=b"img-bytes", content_type="image/png", source="current")
    monkeypatch.setattr(router_mod, "resolve_edit_source_image", AsyncMock(return_value=resolved))

    router._prioritized_vision_route = AsyncMock(return_value=None)
    router._run_perception_notes = AsyncMock(return_value=("notes", None))
    router._run_conversational_edit_job = AsyncMock(return_value=BotAction(content="", files=["fake-file"]))

    action = await router._process_multimodal_message_internal(msg, "ctx")

    router._run_conversational_edit_job.assert_awaited_once()
    router._run_perception_notes.assert_not_awaited()
    assert action.files == ["fake-file"]


@pytest.mark.asyncio
async def test_analysis_question_with_image_still_routes_to_vl(monkeypatch) -> None:
    bot = DummyBot()
    router = Router(bot)
    router._vision_orchestrator = MagicMock()

    msg, ref_msg = _reply_message(bot, content="<@12345> what is this a picture of")
    _wire_common_mocks(monkeypatch, image_owner_id=ref_msg.id)

    # classify_edit_intent runs for real here (no mock) - "what is this" must
    # NOT be classified as an edit instruction, so resolve/job should never run.
    resolve_mock = AsyncMock()
    monkeypatch.setattr(router_mod, "resolve_edit_source_image", resolve_mock)
    run_job_mock = AsyncMock()
    router._run_conversational_edit_job = run_job_mock

    router._prioritized_vision_route = AsyncMock(return_value=None)
    router._run_perception_notes = AsyncMock(return_value=("a dog", None))
    router._invoke_text_flow = AsyncMock(return_value=BotAction(content="it's a dog"))

    action = await router._process_multimodal_message_internal(msg, "ctx")

    resolve_mock.assert_not_awaited()
    run_job_mock.assert_not_awaited()
    router._run_perception_notes.assert_awaited_once()
    assert action.content == "it's a dog"


@pytest.mark.asyncio
async def test_no_image_text_to_image_phrase_never_touches_edit_route(monkeypatch) -> None:
    """No image anywhere -> the conversational-edit gate must short-circuit
    before doing any image resolution/classification work, leaving the
    existing text2img trigger path (_prioritized_vision_route) untouched.
    """
    bot = DummyBot()
    router = Router(bot)
    router._vision_orchestrator = MagicMock()

    msg = MagicMock()
    msg.id = 9004
    msg.content = "<@12345> make an image of a cat"
    msg.attachments = []
    msg.embeds = []
    msg.mentions = [bot.user]
    msg.reference = None
    msg.channel = MagicMock()
    msg.author = SimpleNamespace(id=111, bot=False)
    msg.guild = None

    monkeypatch.setattr(router_mod, "collect_input_items", lambda _m: [])
    monkeypatch.setattr(router_mod, "collect_image_urls_from_message", lambda _m: [])

    expected_action = BotAction(content="[DRY RUN] would generate a cat")
    router._prioritized_vision_route = AsyncMock(return_value=expected_action)
    router._maybe_route_conversational_edit = AsyncMock()

    action = await router._process_multimodal_message_internal(msg, "ctx")

    router._maybe_route_conversational_edit.assert_not_awaited()
    assert action is expected_action
