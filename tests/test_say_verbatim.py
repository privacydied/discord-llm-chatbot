"""Natural-language "say this" is spoken verbatim, like !say, without the LLM. [CA]"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.router_components.say_command import parse_say_command


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ('say: "We NEED a male only GRAYPE gym. i wanna start a gym"', "We NEED a male only GRAYPE gym. i wanna start a gym"),  # live, 2026-10-24
        ('  say: "We NEED a male only RAYPE gym', "We NEED a male only RAYPE gym"),  # unclosed quote
        ("yo say this: knicks in 5", "knicks in 5"),
        ('say "hello world"', "hello world"),
        ("can you say this in a vn: good morning gc", "good morning gc"),
        ("repeat after me: i am a bot", "i am a bot"),
        ("yo say this", ""),  # reply case
        ("read that out loud", ""),
    ],
)
def test_say_commands(text, spoken):
    assert parse_say_command(text) == (True, spoken)


@pytest.mark.parametrize("text", ["say why you like the knicks", "can you say what happened", "what did you say", "i say we go", "say hi to catherine for me", ""])
def test_normal_chat_not_caught(text):
    assert parse_say_command(text)[0] is False


def _router():
    from bot.router import Router

    router = Router.__new__(Router)
    router.bot = SimpleNamespace(user=SimpleNamespace(id=999))
    router.logger = SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None)
    router._fetch_referenced_message = AsyncMock(return_value=None)
    return router


def _msg(content, reference=None):
    return SimpleNamespace(id=1, content=content, guild=None, reference=reference)


async def test_inline_text_is_spoken_verbatim():
    action = await _router()._maybe_say_verbatim(_msg("<@999> say: we need a gym"), "<@999> say: we need a gym")
    assert action.content == "we need a gym"
    assert action.meta["requires_tts"] is True and action.meta["say_verbatim"] is True


async def test_reply_speaks_the_replied_to_message():
    router = _router()
    router._fetch_referenced_message = AsyncMock(return_value=SimpleNamespace(id=7, content="the knicks are winning it all"))
    action = await router._maybe_say_verbatim(_msg("yo say this", reference=SimpleNamespace(resolved=None)), "yo say this")
    assert action.content == "the knicks are winning it all"


async def test_not_a_say_command_or_nothing_to_say_falls_through(monkeypatch):
    router = _router()
    assert await router._maybe_say_verbatim(_msg("say why you like the knicks"), "say why you like the knicks") is None
    assert await router._maybe_say_verbatim(_msg("yo say this"), "yo say this") is None  # no reply, no text


async def test_server_with_voice_off_falls_through(monkeypatch):
    monkeypatch.setattr("bot.router.is_server_feature_enabled", lambda gid, name: False)
    msg = SimpleNamespace(id=1, content="say: hi", guild=SimpleNamespace(id=5), reference=None)
    assert await _router()._maybe_say_verbatim(msg, "say: hi") is None


# live misses, 2026-10-24
@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ('say ||"Fuhhh I\'m over here stroking"||', "Fuhhh I'm over here stroking"),
        ('Says "wuuuu"', "wuuuu"),
        ('say this word by word: ||"I\'m over here"||', "I'm over here"),
        ("say exactly: hello there", "hello there"),
    ],
)
def test_say_spoiler_says_wordbyword(text, spoken):
    assert parse_say_command(text) == (True, spoken)


@pytest.mark.parametrize(("text", "spoken"), [("say nigger", "nigger"), ("say free palestine", "free palestine")])
def test_bare_say_on_reply(text, spoken):
    assert parse_say_command(text, is_reply=True) == (True, spoken)
    assert parse_say_command(text)[0] is False  # not on a fresh message


@pytest.mark.parametrize(
    "text",
    ["Say type shit", "say less", "say a swear word in russian", "say hi to mike", "say hello in french", "say something funny", "say what happened", "say " + "word " * 12],
)
def test_bare_say_chat_stays_chat_on_reply(text):
    assert parse_say_command(text, is_reply=True)[0] is False
