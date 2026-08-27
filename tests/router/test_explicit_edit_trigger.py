"""Unit tests for the explicit @Bot edit: <prompt> mention trigger
(parse_explicit_edit_trigger) and its integration into
_maybe_route_conversational_edit. [CA]
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import bot.router as router_mod
from bot.router import BotAction, Router
from bot.router_components import (
    EditIntentResult,
    ExplicitEditInvocation,
    ResolvedEditImage,
    parse_explicit_edit_trigger,
)


# ---------------------------------------------------------------------------
# parse_explicit_edit_trigger — pure parsing
# ---------------------------------------------------------------------------


class TestParseExplicitEditTrigger:
    @pytest.mark.parametrize(
        ("text", "expected_prompt"),
        [
            ("edit: make this guy chinese", "make this guy chinese"),
            ("edit make this guy chinese", "make this guy chinese"),
            ("EDIT: make him asian", "make him asian"),
            ("Edit: remove background", "remove background"),
            ("edit:make him tall", "make him tall"),
            ("edit     add a hat", "add a hat"),
            ("  edit: fix the lighting  ", "fix the lighting"),
        ],
    )
    def test_valid_forms_parsed(self, text, expected_prompt):
        result = parse_explicit_edit_trigger(text)
        assert result is not None
        assert result.prompt == expected_prompt

    @pytest.mark.parametrize(
        "text",
        [
            "",  # empty
            "   ",  # whitespace only
            "edit",  # bare keyword, no prompt
            "edit:",  # colon but no prompt
            "edit ",  # trailing space, no prompt
            "edited by someone",  # "edit" is prefix of longer word
            "editor picks",  # substring
            "credit me please",  # contains "edit" inside "credit"
            "i updated the reddit post",  # "edit" inside "reddit"
            "the prefix is wrong",  # "fix" is not "edit"
            "make him chinese",  # valid heuristic but not explicit trigger
            "hello world",  # unrelated
        ],
    )
    def test_non_matching_forms_return_none(self, text):
        assert parse_explicit_edit_trigger(text) is None


# ---------------------------------------------------------------------------
# _maybe_route_conversational_edit — integration: explicit trigger fires
# ---------------------------------------------------------------------------


class DummyBot:
    def __init__(self) -> None:
        self.user = SimpleNamespace(id=12345, mention="<@12345>")
        self.config = {
            "HYBRID_FORCE_PERCEPTION_ON_REPLY": True,
            "VISION_REPLY_IMAGE_HARVEST": True,
            "VISION_ENABLED": True,
            "VISION_CONVERSATIONAL_EDIT_ENABLED": True,
            "MAX_ATTACHMENT_SIZE_MB": 25,
        }
        self.tts_manager = None
        self.loop = None
        self.system_prompts = {"vl_prompt": None}
        self.enhanced_context_manager = None


def _make_message(content, *, attachments=True, reference=None, mentions_bot=True):
    msg = MagicMock()
    msg.id = 9001
    msg.content = content
    msg.attachments = ["fake"] if attachments else []
    msg.embeds = []
    msg.mentions = [SimpleNamespace(id=12345)] if mentions_bot else []
    msg.reference = reference
    msg.channel = MagicMock()
    msg.channel.fetch_message = AsyncMock()
    msg.author = SimpleNamespace(id=111, bot=False)
    msg.guild = None
    return msg


@pytest.mark.asyncio
async def test_explicit_edit_trigger_routes_to_edit_job(monkeypatch) -> None:
    """@Bot edit: <prompt> must fire the edit route even without a keyword match."""
    bot = DummyBot()
    router = Router(bot)
    router._vision_orchestrator = MagicMock()

    msg = _make_message("edit: make this guy chinese -steps 10")

    resolved = ResolvedEditImage(data=b"img", content_type="image/png", source="current")
    monkeypatch.setattr(router_mod, "resolve_edit_source_image", AsyncMock(return_value=resolved))
    monkeypatch.setattr(
        router_mod,
        "parse_explicit_edit_trigger",
        lambda t: ExplicitEditInvocation(prompt="make this guy chinese -steps 10"),
    )
    router._run_conversational_edit_job = AsyncMock(
        return_value=BotAction(content="", files=["edited.png"])
    )

    # authored_text after mention strip = "edit: make this guy chinese -steps 10"
    action = await router._maybe_route_conversational_edit(msg, "edit: make this guy chinese -steps 10")

    router._run_conversational_edit_job.assert_awaited_once()
    # The prompt passed to the job must be just the text AFTER "edit:"
    call_args = router._run_conversational_edit_job.await_args.args
    assert call_args[1] == "make this guy chinese -steps 10"


@pytest.mark.asyncio
async def test_explicit_edit_trigger_without_colon_routes(monkeypatch) -> None:
    """@Bot edit <prompt> (no colon) also fires."""
    bot = DummyBot()
    router = Router(bot)
    router._vision_orchestrator = MagicMock()

    msg = _make_message("edit make him a superhero")

    resolved = ResolvedEditImage(data=b"img", content_type="image/png", source="current")
    monkeypatch.setattr(router_mod, "resolve_edit_source_image", AsyncMock(return_value=resolved))
    router._run_conversational_edit_job = AsyncMock(
        return_value=BotAction(content="", files=["edited.png"])
    )

    action = await router._maybe_route_conversational_edit(msg, "edit make him a superhero")

    router._run_conversational_edit_job.assert_awaited_once()
    call_args = router._run_conversational_edit_job.await_args.args
    assert call_args[1] == "make him a superhero"


@pytest.mark.asyncio
async def test_explicit_trigger_skips_heuristic(monkeypatch) -> None:
    """The explicit trigger must fire even when classify_edit_intent returns False."""
    bot = DummyBot()
    router = Router(bot)
    router._vision_orchestrator = MagicMock()

    msg = _make_message("edit: something totally neutral")

    resolved = ResolvedEditImage(data=b"img", content_type="image/png", source="current")
    monkeypatch.setattr(router_mod, "resolve_edit_source_image", AsyncMock(return_value=resolved))
    # Heuristic would return False for this text — but explicit fires regardless.
    monkeypatch.setattr(
        router_mod,
        "classify_edit_intent",
        lambda *a, **kw: EditIntentResult(is_edit=False),
    )
    router._run_conversational_edit_job = AsyncMock(
        return_value=BotAction(content="", files=["edited.png"])
    )

    action = await router._maybe_route_conversational_edit(msg, "edit: something totally neutral")

    router._run_conversational_edit_job.assert_awaited_once()


@pytest.mark.asyncio
async def test_explicit_trigger_with_no_image_returns_none(monkeypatch) -> None:
    """Explicit trigger but no resolvable image -> None (falls through)."""
    bot = DummyBot()
    router = Router(bot)
    router._vision_orchestrator = MagicMock()

    msg = _make_message("edit: make him tall", attachments=False)

    monkeypatch.setattr(router_mod, "resolve_edit_source_image", AsyncMock(return_value=None))
    router._run_conversational_edit_job = AsyncMock()

    result = await router._maybe_route_conversational_edit(msg, "edit: make him tall")

    assert result is None
    router._run_conversational_edit_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_heuristic_still_works_when_explicit_does_not_match(monkeypatch) -> None:
    """When the text doesn't match the explicit form, heuristic classification
    is used as before (regression guard). [REH]"""
    bot = DummyBot()
    router = Router(bot)
    router._vision_orchestrator = MagicMock()

    msg = _make_message("give this man a beard")

    resolved = ResolvedEditImage(data=b"img", content_type="image/png", source="current")
    monkeypatch.setattr(router_mod, "resolve_edit_source_image", AsyncMock(return_value=resolved))
    router._run_conversational_edit_job = AsyncMock(
        return_value=BotAction(content="", files=["edited.png"])
    )

    action = await router._maybe_route_conversational_edit(msg, "give this man a beard")

    router._run_conversational_edit_job.assert_awaited_once()
    # The full text is the prompt for heuristic triggers
    call_args = router._run_conversational_edit_job.await_args.args
    assert call_args[1] == "give this man a beard"


@pytest.mark.asyncio
async def test_no_image_no_keyword_returns_none(monkeypatch) -> None:
    """Plain chat with no edit trigger at all -> None."""
    bot = DummyBot()
    router = Router(bot)
    router._vision_orchestrator = MagicMock()

    msg = _make_message("hello bot how are you")

    router._run_conversational_edit_job = AsyncMock()

    result = await router._maybe_route_conversational_edit(msg, "hello bot how are you")

    assert result is None
    router._run_conversational_edit_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_trigger_uses_invocation_type_in_log(monkeypatch) -> None:
    """The structured log records invocation=explicit for the mention trigger."""
    bot = DummyBot()
    router = Router(bot)
    router._vision_orchestrator = MagicMock()

    msg = _make_message("edit: make him tall")

    resolved = ResolvedEditImage(data=b"img", content_type="image/png", source="current")
    monkeypatch.setattr(router_mod, "resolve_edit_source_image", AsyncMock(return_value=resolved))
    router._run_conversational_edit_job = AsyncMock(
        return_value=BotAction(content="", files=["edited.png"])
    )

    with patch.object(router.logger, "info") as mock_info:
        await router._maybe_route_conversational_edit(msg, "edit: make him tall")

    # Check that the log was emitted with invocation=explicit
    log_calls = [c for c in mock_info.call_args_list if c.args and "edit_route.fired" in c.args]
    assert log_calls, "edit_route.fired log not emitted"
    extra = log_calls[0].kwargs.get("extra", {})
    assert "invocation=explicit" in extra.get("detail", "")
