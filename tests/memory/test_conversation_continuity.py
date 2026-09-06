"""Regression tests for conversational reference continuity.

Production bug: after the bot transcribes an X/Twitter video (or describes an
image, extracts a URL, ...), a follow-up in the SAME conversation such as
"The video attached" is answered as though no media was ever seen
("I can't view videos directly...").

Invariant under test:
    If the bot perceived/extracted/transcribed something during a recent
    conversational turn, later turns in that conversation can refer back to
    that result naturally.

These tests exercise the real prompt-assembly path used by the text backend
(`contextual_brain_infer_simple` + a real `EnhancedContextManager`) and
capture the exact prompt handed to the text model. They are written against
the fixed behavior and MUST fail on the old code.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from bot.contextual_brain import contextual_brain_infer_simple
from bot.memory.enhanced_context_manager import EnhancedContextManager

X_URL = "https://x.com/example/status/123"
TRANSCRIPT = "hello from the video, friends and countrymen"
RAG_BLOCK = "=== Relevant Knowledge ===\nMuseums in Paris are lovely.\n=== End Knowledge ==="


# --------------------------------------------------------------------------
# Fakes (mirror tests/memory/test_mention_context.py conventions)
# --------------------------------------------------------------------------


class FakeAuthor:
    def __init__(self, id: int, name: str, bot: bool = False) -> None:
        self.id = id
        self.name = name
        self.display_name = name
        self.bot = bot


class FakeAttachment:
    def __init__(self, filename: str, content_type: str, url: str) -> None:
        self.filename = filename
        self.content_type = content_type
        self.url = url


class FakeChannel:
    def __init__(self, channel_id: int) -> None:
        self.id = channel_id

    def __repr__(self) -> str:  # pragma: no cover
        return f"<FakeChannel {self.id}>"


class FakeGuild:
    def __init__(self, gid: int) -> None:
        self.id = gid


class FakeMessage:
    _next_id = 1000

    def __init__(
        self,
        content: str,
        author: FakeAuthor,
        channel: FakeChannel,
        guild: FakeGuild | None = None,
        *,
        msg_id: int | None = None,
        reference: SimpleNamespace | None = None,
        mentions: list | None = None,
        attachments: list | None = None,
    ) -> None:
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
        self.jump_url = f"https://discordapp.com/channels/0/{channel.id}/{msg_id}"


def make_ecm(tmp_path, history_window: int = 10) -> EnhancedContextManager:
    bot = SimpleNamespace(user=FakeAuthor(999, "ronin", bot=True), get_user=lambda uid: None)
    return EnhancedContextManager(
        bot,
        filepath=str(tmp_path / "enhanced_context.json"),
        history_window=history_window,
        encryption_key=Fernet.generate_key(),
    )


def make_bot(ecm: EnhancedContextManager):
    bot_user = FakeAuthor(999, "ronin", bot=True)
    return SimpleNamespace(user=bot_user, enhanced_context_manager=ecm, get_user=lambda uid: None)


async def run_turn(
    bot,
    message: FakeMessage,
    prompt: str,
    *,
    extra_context: str | None = None,
    retrieved_context: str | None = None,
) -> str:
    """Run the real prompt assembly; return the exact prompt given to the text model."""
    captured: dict[str, str] = {}

    async def fake_brain_infer(prompt_text, context="", system_prompt=None):
        captured["prompt"] = prompt_text
        result = MagicMock()
        result.content = "model reply"
        return result

    kwargs: dict[str, Any] = {}
    if extra_context is not None:
        kwargs["extra_context"] = extra_context
    if retrieved_context is not None:
        kwargs["retrieved_context"] = retrieved_context
    with patch("bot.contextual_brain.brain_infer", side_effect=fake_brain_infer):
        await contextual_brain_infer_simple(message, prompt, bot, **kwargs)
    return captured["prompt"]


async def record_x_video_turn(ecm, user, channel, guild, bot_user) -> FakeMessage:
    """Simulate turn 1: user posts an X video URL, bot transcribes it."""
    msg1 = FakeMessage(
        f"@ronin please transcribe {X_URL}",
        user,
        channel,
        guild,
        mentions=[bot_user],
    )
    await ecm.append_message(msg1, role="user")
    await ecm.attach_derived_notes(
        msg1,
        [{"kind": "x_video", "label": X_URL, "text": f"tweet text: check this out\ntranscript: {TRANSCRIPT}"}],
    )
    bot_reply = FakeMessage(f"Transcription: {TRANSCRIPT}", bot_user, channel, guild)
    await ecm.append_message(bot_reply, role="bot")
    return msg1


# --------------------------------------------------------------------------
# TEST 1: X video follow-up continuity (the reported production bug)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_x_video_followup_sees_transcript(tmp_path) -> None:
    ecm = make_ecm(tmp_path)
    bot = make_bot(ecm)
    user, guild, channel = FakeAuthor(1, "alice"), FakeGuild(10), FakeChannel(100)

    await record_x_video_turn(ecm, user, channel, guild, bot.user)

    # A few messages later in the SAME conversation:
    msg2 = FakeMessage("@ronin The video attached", user, channel, guild, mentions=[bot.user])
    await ecm.append_message(msg2, role="user")

    # A RAG/memory block is present (as in production) -- it must NOT evict
    # the recent conversation from the prompt.
    prompt = await run_turn(bot, msg2, "The video attached", retrieved_context=RAG_BLOCK)

    assert X_URL in prompt, "media identity lost from next-turn context"
    assert TRANSCRIPT in prompt, "derived transcript lost from next-turn context"
    assert "model reply" not in prompt  # sanity: we inspect the INPUT prompt
    assert "Transcription:" in prompt, "prior assistant response lost from next-turn context"


# --------------------------------------------------------------------------
# TEST 2: loose image follow-up ("translate that")
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_followup_sees_vl_result(tmp_path) -> None:
    ecm = make_ecm(tmp_path)
    bot = make_bot(ecm)
    user, guild, channel = FakeAuthor(2, "bob"), FakeGuild(10), FakeChannel(100)

    msg1 = FakeMessage(
        "@ronin what does this say?",
        user,
        channel,
        guild,
        mentions=[bot.user],
        attachments=[FakeAttachment("sign.png", "image/png", "https://cdn.discord/sign.png")],
    )
    await ecm.append_message(msg1, role="user")
    await ecm.attach_derived_notes(msg1, [{"kind": "image", "label": "sign.png", "text": "VL description: a stop sign reading STOP"}])
    await ecm.append_message(FakeMessage("It says STOP", bot.user, channel, guild), role="bot")

    msg2 = FakeMessage("@ronin translate that into Arabic", user, channel, guild, mentions=[bot.user])
    await ecm.append_message(msg2, role="user")

    prompt = await run_turn(bot, msg2, "translate that into Arabic", retrieved_context=RAG_BLOCK)
    assert "STOP" in prompt


# --------------------------------------------------------------------------
# TEST 3: explicit Discord reply carries reference identity
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_reply_resolves_referenced_turn(tmp_path) -> None:
    ecm = make_ecm(tmp_path)
    bot = make_bot(ecm)
    user, guild, channel = FakeAuthor(3, "cara"), FakeGuild(10), FakeChannel(100)

    msg1 = await record_x_video_turn(ecm, user, channel, guild, bot.user)

    reply = FakeMessage(
        "@ronin what did he say in it?",
        user,
        channel,
        guild,
        mentions=[bot.user],
        reference=SimpleNamespace(message_id=msg1.id),
    )
    await ecm.append_message(reply, role="user")

    # Referenced identity must be captured on the stored turn ...
    stored = ecm.get_turn_by_message_id(reply.id)
    assert stored is not None
    assert stored["referenced_message_id"] == str(msg1.id)

    # ... and the exact referenced turn (with derived media text) must resolve.
    target = ecm.get_turn_by_message_id(msg1.id)
    assert target is not None
    assert any(TRANSCRIPT in note["text"] for note in target["derived"])

    prompt = await run_turn(bot, reply, "what did he say in it?", retrieved_context=RAG_BLOCK)
    assert TRANSCRIPT in prompt


# --------------------------------------------------------------------------
# TEST 10: router shared boundary -- aggregator/evidence mapping + record
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_record_boundary_maps_all_modalities(tmp_path) -> None:
    """Route parity at the shared boundary: every modality's result maps to
    the same turn-note representation and joins the stored turn."""
    import logging
    from unittest.mock import MagicMock

    from bot.memory.turn_notes import build_note, note_from_aggregator_result, note_from_evidence_part
    from bot.router import Router

    ecm = make_ecm(tmp_path)
    bot = MagicMock()
    bot.enhanced_context_manager = ecm
    router = Router(bot=bot, flow_overrides={}, logger=logging.getLogger("test"))
    user, guild, channel = FakeAuthor(12, "ian"), FakeGuild(10), FakeChannel(100)

    msg = FakeMessage(f"@ronin transcribe {X_URL}", user, channel, guild)
    await ecm.append_message(msg, role="user")

    agg_notes = [
        note_from_aggregator_result("VIDEO_URL", X_URL, f"transcript {TRANSCRIPT}"),
        note_from_aggregator_result("AUDIO_VIDEO_FILE", "clip.mp4", "audio words"),
        note_from_aggregator_result("SINGLE_IMAGE", "pic.png", "vl seen"),
        note_from_aggregator_result("GENERAL_URL", "https://example.com", "article text"),
        note_from_aggregator_result("PDF_DOCUMENT", "doc.pdf", "doc text"),
    ]
    assert all(n is not None for n in agg_notes)
    assert {n["kind"] for n in agg_notes} == {"video", "audio", "image", "url", "document"}
    # SCREENSHOT_URL maps to the screenshot kind (covered in T7's shared
    # representation test; omitted here to respect the per-turn note cap).
    assert note_from_aggregator_result("SCREENSHOT_URL", "https://example.com", "shot seen")["kind"] == "screenshot"
    # Failure placeholders must not pollute conversation state.
    assert note_from_aggregator_result("VIDEO_URL", X_URL, "❌ Failed: boom") is None
    assert note_from_aggregator_result("VIDEO_URL", X_URL, "   ") is None

    ev_notes = [
        note_from_evidence_part("[TRANSCRIPT: clip.mp4]\nspoken words"),
        note_from_evidence_part("[DOCUMENT: doc.pdf]\npaper words"),
        note_from_evidence_part("[IMAGE ANALYSIS]\nseen things"),
    ]
    assert [n["kind"] for n in ev_notes] == ["audio", "document", "image"]

    assert await router._record_turn_derived(msg, agg_notes) is True
    assert await router._record_turn_derived(msg, ev_notes) is True
    assert await router._record_turn_derived(msg, []) is False
    assert await router._record_turn_derived(msg, None) is False

    # Per-turn note cap holds: 5 + 3 notes attached, a 9th is refused.
    turn = ecm.get_turn_by_message_id(msg.id)
    assert turn is not None and len(turn["derived"]) == 8
    assert await router._record_turn_derived(msg, [build_note("url", "extra", "ninth")]) is False
    assert len(ecm.get_turn_by_message_id(msg.id)["derived"]) == 8

    history = ecm.format_context_string(ecm.get_context_for_user(msg))
    for snippet in (TRANSCRIPT, "audio words", "vl seen", "article text", "doc text", "spoken words", "paper words", "seen things"):
        assert snippet in history


# --------------------------------------------------------------------------
# TEST 11: router archive fallback when the gateway fetch misses
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_archive_fallback_on_gateway_miss(tmp_path) -> None:
    import logging
    from unittest.mock import MagicMock, patch

    import discord

    from bot.router import Router

    ecm = make_ecm(tmp_path)
    bot = MagicMock()
    bot.enhanced_context_manager = ecm
    router = Router(bot=bot, flow_overrides={}, logger=logging.getLogger("test"))
    user, guild = FakeAuthor(13, "june"), FakeGuild(10)

    class DeadChannel(FakeChannel):
        async def fetch_message(self, mid: int):
            raise discord.NotFound(SimpleNamespace(), "gone")

    channel = DeadChannel(100)
    msg = FakeMessage("@ronin what did he say?", user, channel, guild, reference=SimpleNamespace(message_id=424201, resolved=None))
    router._dispatch_metadata[msg.id] = {}

    record = {
        "content": "@ronin transcribe https://x.com/example/status/999",
        "attachments_json": [{"filename": "v.mp4", "content_type": "video/mp4", "url": "https://cdn/x/v.mp4", "proxy_url": "", "size": 1}],
        "author_id": "7",
        "author_username": "dave",
        "author_display_name": "dave",
        "author_is_bot": False,
        "created_at": "2026-09-06T10:00:00+00:00",
        "jump_url": "https://discordapp.com/channels/0/100/424201",
        "reply_to_message_id": None,
    }
    with patch("bot.server_archive.service.get_archived_message", new=AsyncMock(return_value=record)):
        resolved = await router._fetch_referenced_message(msg)

    assert resolved is not None
    assert "x.com/example/status/999" in resolved.content
    assert resolved.attachments and resolved.attachments[0].url == "https://cdn/x/v.mp4"
    assert router._dispatch_metadata[msg.id].get("reference_resolve") == "archive"


# --------------------------------------------------------------------------
# TEST 12: reply-chain local block renders stored derived media text
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reply_chain_block_includes_derived_media(tmp_path) -> None:
    from bot.memory.mention_context import REPLY_CASE, _package

    ecm = make_ecm(tmp_path)
    bot_user = FakeAuthor(999, "ronin", bot=True)
    bot = SimpleNamespace(user=bot_user, enhanced_context_manager=ecm)
    user, guild, channel = FakeAuthor(14, "kim"), FakeGuild(10), FakeChannel(100)

    msg1 = FakeMessage(f"@ronin transcribe {X_URL}", user, channel, guild, mentions=[bot_user])
    await ecm.append_message(msg1, role="user")
    await ecm.attach_derived_notes(msg1, [{"kind": "x_video", "label": X_URL, "text": TRANSCRIPT}])

    trigger = FakeMessage(
        "@ronin what did he say?",
        user,
        channel,
        guild,
        mentions=[bot_user],
        reference=SimpleNamespace(message_id=msg1.id),
    )
    block = _package(bot, trigger, REPLY_CASE, msg1, [msg1, trigger])
    assert "/media" in block.joined_text
    assert TRANSCRIPT in block.joined_text
    assert X_URL in block.joined_text


# --------------------------------------------------------------------------
# TEST 4: archive fallback for a reply whose turn aged out of memory
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_archive_fallback_recovers_referenced_message(tmp_path) -> None:
    from bot.server_archive.models import (
        ArchiveChannel,
        ArchiveGuild,
        ArchiveMessage,
        ArchiveMessageBundle,
        ArchiveUser,
    )
    from bot.server_archive.store import ServerArchiveStore

    store = ServerArchiveStore(tmp_path / "archive.db")
    await store.initialize()
    try:
        bundle = ArchiveMessageBundle(
            guild=ArchiveGuild(guild_id="10", name="test-guild"),
            channel=ArchiveChannel(channel_id="100", guild_id="10", name="general"),
            author=ArchiveUser(user_id="7", username="dave", display_name="dave"),
            message=ArchiveMessage(
                message_id="424201",
                guild_id="10",
                channel_id="100",
                thread_id=None,
                author_id="7",
                content="@ronin transcribe https://x.com/example/status/999",
                clean_content="transcribe https://x.com/example/status/999",
            ),
        )
        await store.upsert_bundles([bundle])

        recovered = await store.get_message_by_id("424201")
        assert recovered is not None
        assert "x.com/example/status/999" in recovered["content"]
        assert recovered["guild_id"] == "10"

        assert await store.get_message_by_id("no-such-message") is None
    finally:
        await store.close() if hasattr(store, "close") else None


# --------------------------------------------------------------------------
# TEST 5: no cross-channel bleed
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_cross_channel_bleed(tmp_path) -> None:
    ecm = make_ecm(tmp_path)
    bot = make_bot(ecm)
    user, guild = FakeAuthor(5, "erin"), FakeGuild(10)
    chan_a, chan_b = FakeChannel(101), FakeChannel(102)

    await record_x_video_turn(ecm, user, chan_a, guild, bot.user)

    msg_b = FakeMessage("@ronin The video attached", user, chan_b, guild, mentions=[bot.user])
    await ecm.append_message(msg_b, role="user")
    prompt = await run_turn(bot, msg_b, "The video attached", retrieved_context=RAG_BLOCK)

    assert X_URL not in prompt
    assert TRANSCRIPT not in prompt


# --------------------------------------------------------------------------
# TEST 6: another user speaking between turns must not evict continuity
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_other_user_between_turns_keeps_continuity(tmp_path) -> None:
    ecm = make_ecm(tmp_path)
    bot = make_bot(ecm)
    guild, channel = FakeGuild(10), FakeChannel(100)
    user_a, user_b = FakeAuthor(6, "amy"), FakeAuthor(7, "ben")

    await record_x_video_turn(ecm, user_a, channel, guild, bot.user)
    await ecm.append_message(FakeMessage("hey everyone, lunch?", user_b, channel, guild), role="user")

    followup = FakeMessage("@ronin what did he say at the end?", user_a, channel, guild, mentions=[bot.user])
    await ecm.append_message(followup, role="user")
    prompt = await run_turn(bot, followup, "what did he say at the end?", retrieved_context=RAG_BLOCK)

    assert TRANSCRIPT in prompt
    assert X_URL in prompt


# --------------------------------------------------------------------------
# TEST 7: route parity -- every modality feeds the same representation
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_modalities_share_one_context_representation(tmp_path) -> None:
    from bot.memory.turn_notes import build_note

    ecm = make_ecm(tmp_path)
    bot = make_bot(ecm)
    user, guild, channel = FakeAuthor(8, "finn"), FakeGuild(10), FakeChannel(100)

    msg = FakeMessage("@ronin process everything", user, channel, guild, mentions=[bot.user])
    await ecm.append_message(msg, role="user")
    kinds_texts = [
        ("x_video", "https://x.com/a/status/1", "tweet transcript alpha"),
        ("video", "clip.mp4", "STT transcript beta"),
        ("audio", "voice.ogg", "audio transcript gamma"),
        ("image", "pic.png", "VL description delta"),
        ("url", "https://example.com/page", "extracted article epsilon"),
        ("screenshot", "https://example.com", "screenshot analysis zeta"),
        ("document", "paper.pdf", "extracted document eta"),
    ]
    await ecm.attach_derived_notes(msg, [build_note(k, label, text) for k, label, text in kinds_texts])

    history = ecm.format_context_string(ecm.get_context_for_user(msg))
    for _, _, text in kinds_texts:
        assert text in history, f"modality result missing from shared context: {text!r}"


# --------------------------------------------------------------------------
# TEST 8: bounded context -- huge transcripts stay bounded but identifiable
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_large_transcript_is_bounded_but_identifiable(tmp_path) -> None:
    ecm = make_ecm(tmp_path)
    bot = make_bot(ecm)
    user, guild, channel = FakeAuthor(9, "gina"), FakeGuild(10), FakeChannel(100)

    msg = FakeMessage(f"@ronin transcribe {X_URL}", user, channel, guild, mentions=[bot.user])
    await ecm.append_message(msg, role="user")
    big = "word " * 6000  # ~30k chars
    await ecm.attach_derived_notes(msg, [{"kind": "x_video", "label": X_URL, "text": big}])

    stored = ecm.get_turn_by_message_id(msg.id)
    assert stored is not None
    note_text = stored["derived"][0]["text"]
    assert len(note_text) < len(big), "oversize derived text must be compacted"
    assert len(note_text) <= 2000

    history = ecm.format_context_string(ecm.get_context_for_user(msg))
    assert len(history) <= ecm.max_total_chars + 2000  # bounded by window caps
    assert X_URL in history, "media identity must survive compaction"
    assert "x_video" in history


# --------------------------------------------------------------------------
# TEST 9: restart round-trip retains identity + derived metadata
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_round_trip_retains_turn_metadata(tmp_path) -> None:
    key = Fernet.generate_key()
    bot = SimpleNamespace(user=FakeAuthor(999, "ronin", bot=True), get_user=lambda uid: None)
    path = str(tmp_path / "enhanced_context.json")
    ecm = EnhancedContextManager(bot, filepath=path, history_window=10, encryption_key=key)
    user, guild, channel = FakeAuthor(11, "hank"), FakeGuild(10), FakeChannel(100)

    msg1 = FakeMessage(f"@ronin transcribe {X_URL}", user, channel, guild, mentions=[bot.user])
    await ecm.append_message(msg1, role="user")
    await ecm.attach_derived_notes(msg1, [{"kind": "x_video", "label": X_URL, "text": TRANSCRIPT}])
    assert await ecm.flush_if_dirty() is True

    ecm2 = EnhancedContextManager(bot, filepath=path, history_window=10, encryption_key=key)
    revived = ecm2.get_turn_by_message_id(msg1.id)
    assert revived is not None, "turn identity lost across restart"
    assert revived["urls"] == [X_URL]
    assert any(TRANSCRIPT in n["text"] for n in revived["derived"])
    prompt = await run_turn(make_bot(ecm2), msg1, "The video attached", retrieved_context=RAG_BLOCK)
    assert TRANSCRIPT in prompt
