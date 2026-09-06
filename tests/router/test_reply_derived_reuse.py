"""Regression tests: reply to processed X media must reuse stored derived results.

Production fault: a reply to a previously transcribed X video re-ran the full
media pipeline (syndication + yt-dlp + Whisper + Playwright + Wayback) because
four reply-harvest blocks re-expanded the parent's URLs/embeds into items and
nothing consulted the stored turn first.

- reuse-hit run: handler call count == 0, strict STT mock untouched, follow-up
  routes text-only.
- control (no stored result): exactly one handler call, canonical single URL.
- before/after: with reuse+prune disabled the same fixture harvests 4 items
  (the reported production shape); with the fix it harvests 0/1.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.fernet import Fernet

from bot.action import BotAction
from bot.memory.enhanced_context_manager import EnhancedContextManager
from bot.modality import InputItem

STATUS_ID = "123456789"
STATUS_URL = f"https://x.com/userA/status/{STATUS_ID}"
STATUS_ALIAS = f"https://twitter.com/userA/status/{STATUS_ID}?s=20"
STATUS_CANON = f"https://x.com/i/status/{STATUS_ID}"
PROFILE_URL = "https://x.com/userA"
PREVIEW_URL = "https://jf.x.com/images/media-preview/xyz.jpg"
VIDEO_URL = "https://video.twimg.com/ext_tw_video/abc.mp4"
TRANSCRIPT = "hello from the video, friends"


class FakeAuthor:
    def __init__(self, id: int, name: str, bot: bool = False) -> None:
        self.id = id
        self.name = name
        self.display_name = name
        self.bot = bot


class FakeChannel:
    def __init__(self, channel_id: int) -> None:
        self.id = channel_id


class FakeGuild:
    def __init__(self, gid: int) -> None:
        self.id = gid


class FakeMessage:
    _next_id = 5000

    def __init__(self, content, author, channel, guild=None, *, msg_id=None, reference=None, mentions=None, attachments=None, embeds=None):
        if msg_id is None:
            FakeMessage._next_id += 1
            msg_id = FakeMessage._next_id
        self.id = msg_id
        self.author = author
        self.content = content
        self.created_at = datetime.now(UTC)
        self.channel = channel
        self.guild = guild
        self.reference = reference
        self.mentions = mentions or []
        self.attachments = attachments or []
        self.embeds = embeds or []
        self.jump_url = f"https://discordapp.com/channels/0/{channel.id}/{msg_id}"


def make_ref_message(user, channel, guild):
    """Parent shaped like the production report: content URL + rich embed with
    canonical URL, direct video, author profile, and jf.x.com preview image."""
    embed = SimpleNamespace(
        type="rich",
        url=STATUS_URL,
        video=SimpleNamespace(url=VIDEO_URL),
        author=SimpleNamespace(url=PROFILE_URL),
        image=SimpleNamespace(url=PREVIEW_URL),
        thumbnail=None,
        provider=SimpleNamespace(name="Twitter"),
    )
    return FakeMessage(
        f"check this out {STATUS_URL}",
        user,
        channel,
        guild,
        msg_id=1546192435338813460,
        mentions=[],
        embeds=[embed],
    )


def make_ecm(tmp_path):
    bot = SimpleNamespace(user=FakeAuthor(999, "ronin", bot=True), get_user=lambda uid: None)
    return EnhancedContextManager(
        bot,
        filepath=str(tmp_path / "enhanced_context.json"),
        history_window=10,
        encryption_key=Fernet.generate_key(),
    )


def make_router(ecm):
    from bot.router import Router

    bot = SimpleNamespace(
        user=FakeAuthor(999, "ronin", bot=True),
        config={},
        tts_manager=None,
        loop=None,
        enhanced_context_manager=ecm,
    )
    return Router(bot=bot, flow_overrides={}, logger=logging.getLogger("test-reuse"))


async def run_multimodal(router, message, handler_fake):
    """Drive the real harvest+dispatch internals with network handlers faked.

    Returns (invoke_contents, handler_calls, elapsed_s).
    """
    invoke_contents: list[str] = []

    async def fake_invoke(content, msg, context_str, perception_notes=None):
        invoke_contents.append(content.compose() if hasattr(content, "compose") else content)
        return BotAction(content="ok")

    t0 = time.perf_counter()
    with (
        patch.object(router, "_handle_item_with_provider", new=handler_fake),
        patch.object(router, "_invoke_text_flow", new=fake_invoke),
        patch.object(router, "_maybe_add_news_digest", new=AsyncMock(side_effect=lambda t, m, c: c)),
        patch.object(router, "_maybe_answer_with_tools", new=AsyncMock(return_value=None)),
        patch("bot.router.hear_infer_from_url", new=AsyncMock(side_effect=AssertionError("STT must not run on reuse path"))),
    ):
        await router._process_multimodal_message_internal(message, "")
    return invoke_contents, handler_fake, time.perf_counter() - t0


@pytest.mark.asyncio
async def test_reply_reuses_transcript_with_zero_stt(tmp_path) -> None:
    """TEST 1: previously transcribed X video + later reply -> no reprocessing."""
    ecm = make_ecm(tmp_path)
    router = make_router(ecm)
    user, guild, channel = FakeAuthor(1, "alice"), FakeGuild(10), FakeChannel(100)
    ref = make_ref_message(user, channel, guild)

    await ecm.append_message(ref, role="user")
    await ecm.attach_derived_notes(ref, [{"kind": "x_video", "label": STATUS_URL, "text": TRANSCRIPT}])
    await ecm.append_message(FakeMessage(f"Transcription: {TRANSCRIPT}", router.bot.user, channel, guild), role="bot")

    reply = FakeMessage(
        "what did he say in it?",
        user,
        channel,
        guild,
        reference=SimpleNamespace(message_id=ref.id, resolved=None),
    )

    async def _fetch(mid):
        raise AssertionError("gateway fetch must not be needed")

    reply.channel.fetch_message = _fetch  # type: ignore[attr-defined]

    assert await router._check_reply_derived_reuse(reply) is True
    assert router._reply_reuse_hit(reply) is True

    handler = AsyncMock(side_effect=AssertionError("no media item may be processed on reuse path"))
    invoke_contents, _, elapsed = await run_multimodal(router, reply, handler)

    handler.assert_not_called()
    assert len(invoke_contents) == 1
    assert "x.com" not in invoke_contents[0]
    assert "what did he say in it?" in invoke_contents[0]
    print(f"\nreuse-hit follow-up Jiabei: {elapsed * 1000:.1f}ms, handler calls=0")


@pytest.mark.asyncio
async def test_no_stored_result_processes_exactly_once(tmp_path) -> None:
    """TEST 2: no stored derived result -> fallback processes canonical URL once."""
    ecm = make_ecm(tmp_path)
    router = make_router(ecm)
    user, guild, channel = FakeAuthor(2, "bob"), FakeGuild(10), FakeChannel(100)
    ref = make_ref_message(user, channel, guild)
    await ecm.append_message(ref, role="user")  # raw turn only, no derived notes

    reply = FakeMessage(
        "please transcribe this",
        user,
        channel,
        guild,
        reference=SimpleNamespace(message_id=ref.id, resolved=None),
    )

    async def _fetch(mid):
        return ref

    reply.channel.fetch_message = _fetch  # type: ignore[attr-defined]

    assert await router._check_reply_derived_reuse(reply) is False

    seen: list[str] = []

    async def fake_handler(item, modality, provider_config, message=None):
        seen.append(str(item.payload))
        return f"PROCESSED::{item.payload}"

    handler = AsyncMock(side_effect=fake_handler)
    invoke_contents, _, elapsed = await run_multimodal(router, reply, handler)

    assert len(seen) == 1, f"expected exactly one media item, got {seen}"
    assert seen[0] == STATUS_CANON
    assert len(invoke_contents) == 1
    print(f"\nfallback follow-up Jiabei: {elapsed * 1000:.1f}ms, handler calls=1")


@pytest.mark.asyncio
async def test_before_shape_harvests_four_items(tmp_path) -> None:
    """Before/after: with reuse+prune disabled the fixture yields the reported
    4 URL items; the fix reduces the same fixture to 0 (reuse) or 1 (fallback)."""
    ecm = make_ecm(tmp_path)
    router = make_router(ecm)
    user, guild, channel = FakeAuthor(3, "cara"), FakeGuild(10), FakeChannel(100)
    ref = make_ref_message(user, channel, guild)
    await ecm.append_message(ref, role="user")

    reply = FakeMessage(
        "please transcribe this",
        user,
        channel,
        guild,
        reference=SimpleNamespace(message_id=ref.id, resolved=None),
    )

    async def _fetch(mid):
        return ref

    reply.channel.fetch_message = _fetch  # type: ignore[attr-defined]

    seen: list[str] = []

    async def fake_handler(item, modality, provider_config, message=None):
        seen.append(str(item.payload))
        return f"PROCESSED::{item.payload}"

    handler = AsyncMock(side_effect=fake_handler)
    with patch.object(router, "_prune_redundant_reply_media_items", return_value={}):
        await run_multimodal(router, reply, handler)

    assert len(seen) == 4, f"expected the reported 4-URL shape, got {seen}"
    assert PREVIEW_URL in seen and PROFILE_URL in seen


@pytest.mark.asyncio
async def test_prune_collapses_to_one_status_entity(tmp_path) -> None:
    """TEST 3: embed preview + canonical status + twitter alias + profile ->
    one logical X status/media entity."""
    ecm = make_ecm(tmp_path)
    router = make_router(ecm)
    user, guild, channel = FakeAuthor(4, "dan"), FakeGuild(10), FakeChannel(100)
    msg = FakeMessage("reply", user, channel, guild, reference=SimpleNamespace(message_id=1, resolved=None))
    items = [
        InputItem(source_type="url", payload=STATUS_URL, order_index=1),
        InputItem(source_type="url", payload=STATUS_ALIAS, order_index=2),
        InputItem(source_type="url", payload=STATUS_CANON, order_index=3),
        InputItem(source_type="url", payload=PREVIEW_URL, order_index=4),
        InputItem(source_type="url", payload=PROFILE_URL, order_index=5),
        InputItem(source_type="url", payload=VIDEO_URL, order_index=6),
        InputItem(source_type="url", payload="https://x.com/other/status/1546192435338813460", order_index=7),
    ]
    stats = router._prune_redundant_reply_media_items(msg, items)
    payloads = [str(it.payload) for it in items]
    assert payloads == [STATUS_CANON, "https://x.com/i/status/1546192435338813460"], payloads
    assert stats["pruned_aliases"] == 2
    assert stats["pruned_profiles"] == 1
    assert stats["pruned_previews"] == 1
    assert stats["pruned_media_artifacts"] == 1
    assert stats["canonicalized"] == 2


@pytest.mark.asyncio
async def test_jf_preview_never_general_url() -> None:
    """TEST 4: jf.x.com media-preview artifact routes to image, never tiered
    web extraction."""
    from bot.modality import InputModality, map_item_to_modality

    item = InputItem(source_type="url", payload=PREVIEW_URL, order_index=1)
    assert await map_item_to_modality(item) == InputModality.SINGLE_IMAGE


@pytest.mark.asyncio
async def test_reuse_survives_persistence_restart(tmp_path) -> None:
    """TEST 5: referenced derived state survives the persisted round-trip the
    recent-turn subsystem promises (encrypted file reload)."""
    from bot.memory.enhanced_context_manager import EnhancedContextManager

    key = Fernet.generate_key()
    path = str(tmp_path / "enhanced_context.json")
    mkbot = lambda ecm_path=None: SimpleNamespace(user=FakeAuthor(999, "ronin", bot=True), get_user=lambda uid: None)
    ecm = EnhancedContextManager(mkbot(), filepath=path, history_window=10, encryption_key=key)
    user, guild, channel = FakeAuthor(5, "erin"), FakeGuild(10), FakeChannel(100)
    ref = make_ref_message(user, channel, guild)
    await ecm.append_message(ref, role="user")
    await ecm.attach_derived_notes(ref, [{"kind": "x_video", "label": STATUS_URL, "text": TRANSCRIPT}])
    assert await ecm.flush_if_dirty() is True

    ecm2 = EnhancedContextManager(mkbot(), filepath=path, history_window=10, encryption_key=key)
    router2 = make_router(ecm2)
    reply = FakeMessage(
        "what did he say?",
        user,
        channel,
        guild,
        reference=SimpleNamespace(message_id=ref.id, resolved=None),
    )
    assert await router2._check_reply_derived_reuse(reply) is True
    turn = ecm2.get_turn_by_message_id(ref.id)
    assert turn is not None and any(TRANSCRIPT in n["text"] for n in turn["derived"])
