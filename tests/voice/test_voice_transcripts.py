"""Replies to the bot's own voice notes use the TTS transcript, never STT. [CA]"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bot.voice import transcripts as vt

BOT_ID = 1392310898311303259


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_TRANSCRIPTS_FILE", str(tmp_path / "vt.json"))
    vt.reset_cache_for_tests()
    yield
    vt.reset_cache_for_tests()


def _voice_note(mid=10, author=BOT_ID, flags=1 << 13, content="", atts=None):
    att = SimpleNamespace(content_type="audio/ogg", filename="voice-message.ogg", url="https://cdn/x.ogg")
    return SimpleNamespace(id=mid, author=SimpleNamespace(id=author), flags=SimpleNamespace(value=flags), content=content, attachments=atts if atts is not None else [att])


def test_remember_lookup_and_persist_across_restart():
    vt.remember(10, "  paris, obviously  ")
    assert vt.lookup(10) == "paris, obviously"
    vt.reset_cache_for_tests()  # simulate restart: reload from disk
    assert vt.message_text(_voice_note()) == "paris, obviously"


def test_message_text_prefers_real_content_and_ignores_blank():
    vt.remember(10, "spoken")
    assert vt.message_text(_voice_note(content="typed")) == "typed"
    vt.remember(11, "   ")
    assert vt.lookup(11) is None


def test_bounded(monkeypatch):
    monkeypatch.setattr(vt, "MAX_ENTRIES", 3)
    for i in range(5):
        vt.remember(i, f"t{i}")
    assert vt.lookup(0) is None and vt.lookup(4) == "t4"


def test_own_voice_note_detection():
    assert vt.is_own_voice_note(_voice_note(), BOT_ID)  # voice flag
    assert vt.is_own_voice_note(_voice_note(flags=0), BOT_ID)  # audio-file fallback
    assert not vt.is_own_voice_note(_voice_note(author=42), BOT_ID)  # a user's voice note still gets STT
    assert not vt.is_own_voice_note(_voice_note(flags=0, atts=[SimpleNamespace(content_type="image/png", filename="a.png")]), BOT_ID)


@pytest.mark.asyncio
async def test_reply_chain_context_uses_transcript():
    from bot.memory.mention_context import _package

    vt.remember(10, "the knicks are winning it all")
    user = SimpleNamespace(id=5, display_name="pry", bot=False)
    reply = SimpleNamespace(id=11, author=user, content="nah", created_at=None, attachments=[], jump_url="")
    note = _voice_note()
    note.author = SimpleNamespace(id=BOT_ID, display_name="ronin", bot=True)
    note.created_at = None
    note.jump_url = ""
    block = _package(SimpleNamespace(user=SimpleNamespace(id=BOT_ID)), SimpleNamespace(guild=None), "reply", None, [note, reply])
    assert "the knicks are winning it all" in str(block)
