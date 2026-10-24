"""Voice is opt-in per message: explicit asks get a voice note, "say this" stays verbatim. [CA]"""

import pytest

from bot.router_components.say_command import parse_say_command
from bot.router_components.voice_request import wants_voice_reply


@pytest.mark.parametrize(
    "text",
    [
        "can you reply in voice note",
        "send me a voice note",
        "drop a vn",
        "answer in voice",
        "tell me in a voice message why stalin > trotsky",
        "vn me your take",
        "say it out loud",
        "can u send that as a voice note",
        "explain it in a voice note",
        "gimme a voice message",
    ],
)
def test_voice_asks(text):
    assert wants_voice_reply(text)


@pytest.mark.parametrize(
    "text",
    ["can you say this", "say this: hello", "what's a voice note", "i hate voice notes", "don't send a voice note", "no voice notes pls", "his voice is mid", "the vn was funny"],
)
def test_not_voice_asks(text):
    assert not wants_voice_reply(text)


def test_say_this_is_the_verbatim_path_not_a_voice_ask():
    assert parse_say_command("say this: hello there") == (True, "hello there")
    assert not wants_voice_reply("say this: hello there")


def test_requested_message_is_spoken_with_global_voice_off(monkeypatch):
    from types import SimpleNamespace

    from bot.tts.state import TTSState, apply_tts_preference

    monkeypatch.setenv("TTS_GLOBAL_DEFAULT", "false")
    state = TTSState()
    msg = SimpleNamespace(id=42, guild=None, author=SimpleNamespace(id=7))
    plain = SimpleNamespace(content="normal reply", meta={}, files=None, audio_path=None)
    assert not apply_tts_preference(plain, msg, state=state)  # text by default
    state.request_voice_reply(42)
    asked = SimpleNamespace(content="spoken reply", meta={}, files=None, audio_path=None)
    assert apply_tts_preference(asked, msg, state=state) and asked.meta["requires_tts"]
