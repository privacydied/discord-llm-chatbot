"""Router.dispatch_prompt: answer a wrapped prompt despite the earlier dispatch. [CA]"""

from types import SimpleNamespace

from bot.router import Router


async def test_dispatch_prompt_clears_dedupe_swaps_content_and_restores():
    seen = {}

    async def fake_dispatch(message):
        seen["content"] = message.content
        seen["deduped"] = message.id in router._processed_recent_ts
        seen["gate_hint"] = hints.get(message.id)
        return "answer"

    hints = {}
    router = SimpleNamespace(_processed_recent_ts={42: 1.0}, dispatch_message=fake_dispatch, record_gate_hint=hints.__setitem__)
    message = SimpleNamespace(id=42, content="!speak what time is it")

    result = await Router.dispatch_prompt(router, message, "what time is it")

    assert result == "answer"
    assert seen == {"content": "what time is it", "deduped": False, "gate_hint": True}
    assert message.content == "!speak what time is it"  # restored


async def test_dispatch_prompt_restores_content_when_dispatch_raises():
    async def boom(_message):
        raise RuntimeError("model down")

    router = SimpleNamespace(_processed_recent_ts={}, dispatch_message=boom, record_gate_hint=lambda *_: None)
    message = SimpleNamespace(id=7, content="!speak hi")
    try:
        await Router.dispatch_prompt(router, message, "hi")
    except RuntimeError:
        pass
    assert message.content == "!speak hi"


# -- through the real Router: bot.py dispatches "!speak ..." first, then the cog --------


def _real_router(process_text):
    import logging
    from unittest.mock import AsyncMock, MagicMock

    bot = MagicMock()
    bot.config = {
        "OWNER_IDS": [],
        "REPLY_TRIGGERS": ["dm", "mention", "reply", "bot_threads", "owner", "command_prefix"],
        "REQUIRE_MENTION_IN_GUILDS": True,
        "DM_REQUIRE_MENTION": False,
        "COMMAND_PREFIX": "!",
        "AMBIENT_REPLY_ENABLED": False,
        "VISION_ENABLED": False,  # MagicMock intent results would read as "use vision"
    }
    bot.user = SimpleNamespace(id=9999)
    router = Router(bot=bot, flow_overrides={"process_text": process_text}, logger=MagicMock(spec=logging.Logger))
    router._detect_direct_vision_triggers = MagicMock(return_value=False)
    router._try_ambient_reply = MagicMock(return_value=False)
    # The mock-bot test shortcut would answer before the gate; force the production path.
    router._compat_dispatch_for_tests = AsyncMock(return_value=None)
    return router


def _guild_message(content):
    from unittest.mock import MagicMock

    import discord

    message = MagicMock()
    message.id = 77
    message.channel = MagicMock(spec=discord.TextChannel)
    message.guild = SimpleNamespace(id=42)
    message.author = SimpleNamespace(id=1111, bot=False)
    message.content = content
    message.mentions = []
    message.reference = None
    message.attachments = []
    return message


async def test_speak_prompt_reaches_reply_pipeline_through_real_gate_and_dedupe():
    """Gate, dedupe and scope run for real; the multimodal entry every @mention reply
    goes through is stubbed to capture what it receives (MagicMock messages stall it)."""
    from bot.action import BotAction

    prompts = []

    async def process_text(content, context_str, message, perception_notes=None):
        return BotAction(content="unused")

    async def pipeline(message, context_str):
        prompts.append(message.content)
        return BotAction(content="paris")

    router = _real_router(process_text)
    router._process_multimodal_message_internal = pipeline
    message = _guild_message("!speak what's the capital of france")
    first = await router.dispatch_message(message)  # bot.py's first pass: command -> cog
    assert first is None or first.meta.get("delegated_to_cog")
    action = await router.dispatch_prompt(message, "what's the capital of france")
    assert isinstance(action, BotAction) and action.content == "paris"
    assert prompts == ["what's the capital of france"]
    assert message.content == "!speak what's the capital of france"


async def test_mentionless_prompt_without_command_is_still_blocked():
    """Control: the gate hint is what lets the prompt through, not a gate bypass."""
    from bot.action import BotAction

    async def process_text(content, context_str, message, perception_notes=None):
        return BotAction(content="should not happen")

    router = _real_router(process_text)
    message = _guild_message("what's the capital of france")
    assert router._should_process_message(message) is False
    assert await router.dispatch_message(message) is None
