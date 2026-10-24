"""Auto-translation of foreign-language tweet links (bot/embed_fix/translate.py)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import discord
import pytest

from bot.embed_fix import handler as h
from bot.embed_fix import translate
from bot.embed_fix.sites import find_fixes

TR_TWEET = "https://x.com/nefretbeyi/status/2109002841683525719"


def _fix(url: str):
    (fix,) = find_fixes(url)
    return fix


class _Resp:
    def __init__(self, status, payload):
        self.status, self._payload = status, payload

    async def __aenter__(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self, content_type=None):
        return self._payload


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "en"), ("en", "en"), (" EN ", "en"), ("de", "de"), ("", None), ("off", None), ("en-US", None)],
)
def test_parse_target(raw, expected):
    assert translate.parse_target(raw) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (TR_TWEET, "2109002841683525719"),
        ("https://twitter.com/i/web/status/42", "42"),
        ("https://x.com/a/status/42/photo/1", "42"),
        ("https://x.com/a/status/42/en", None),  # poster already chose a language
        ("https://x.com/a/status/42/original", None),
        ("https://www.instagram.com/reel/AB/", None),
    ],
)
def test_tweet_id_only_for_plain_tweet_links(url, expected):
    assert translate.tweet_id(_fix(url)) == expected


@pytest.mark.parametrize(
    ("lang", "expected"),
    [("tr", True), ("ja", True), ("pt-BR", True), ("en", False), ("EN", False), ("en-GB", False)] + [(code, False) for code in sorted(translate.NON_LANGUAGE_CODES)] + [(None, False), ("", False), (5, False)],
)
def test_needs_translation(lang, expected):
    assert translate.needs_translation(lang, "en") is expected


async def test_fetch_lang_reads_fxtwitter_api():
    session = MagicMock()
    session.get = MagicMock(return_value=_Resp(200, {"code": 200, "tweet": {"lang": "tr"}}))
    assert await translate.fetch_lang(session, "123") == "tr"
    assert session.get.call_args.args[0] == "https://api.fxtwitter.com/status/123"


BJP_TWEET = (  # api.fxtwitter.com/status/2113598812363374958, 2026-10-24: lang missing
    "⚠️ Ilhan Omar \n⚠️ Donald Lu\n⚠️ Jeremy Corbyn\n⚠️ Sunita Viswanath \n⚠️ Gunther Fehlinger \n⚠️ खालिस्तानी समर्थक... \n\nये महज कुछ नाम नहीं हैं, बल्कि कट्टर भारत-विरोधियों का पूरा नेटवर्क है, जिनके साथ Rahul"
)


@pytest.mark.parametrize(
    ("tweet", "expected"),
    [
        ({"lang": None, "text": BJP_TWEET}, translate.SCRIPT_GUESS),  # no lang, mostly Hindi -> translate
        ({"lang": "und", "text": "Путин заявил сегодня утром о новых мерах"}, translate.SCRIPT_GUESS),
        ({"text": "no language field but plain english text here"}, None),
        ({"lang": None, "text": "ok 👍"}, None),  # too little text to judge
        ({"lang": "hi", "text": BJP_TWEET}, "hi"),  # a real code always wins
    ],
)
async def test_fetch_lang_guesses_script_when_lang_missing(tweet, expected):
    session = MagicMock()
    session.get = MagicMock(return_value=_Resp(200, {"code": 200, "tweet": tweet}))
    lang = await translate.fetch_lang(session, "123")
    assert lang == expected or (expected is None and not translate.needs_translation(lang, "en"))


@pytest.mark.parametrize(
    "resp",
    [_Resp(404, {}), _Resp(200, {"code": 404}), _Resp(200, ["x"]), _Resp(200, aiohttp.ClientConnectionError("down")), _Resp(200, TimeoutError())],
)
async def test_fetch_lang_failures_return_none(resp):
    session = MagicMock()
    session.get = MagicMock(return_value=resp)
    assert await translate.fetch_lang(session, "123") is None


async def test_apply_appends_target_only_to_foreign_tweets(monkeypatch):
    langs = {"1": "tr", "2": "en", "3": None}
    monkeypatch.setattr(translate, "fetch_lang", AsyncMock(side_effect=lambda session, tid: langs[tid]))
    fixes = find_fixes("https://x.com/a/status/1 https://x.com/b/status/2 https://x.com/c/status/3 https://x.com/d/status/4/ja https://instagram.com/reel/AB")
    out = await translate.apply(fixes, "en")
    assert [f.rewritten for f in out] == [
        "https://vxtwitter.com/a/status/1/en",
        "https://vxtwitter.com/b/status/2",
        "https://vxtwitter.com/c/status/3",
        "https://vxtwitter.com/d/status/4/ja",
        "https://www.oginstagram.com/reel/AB/",
    ]
    assert translate.fetch_lang.await_count == 3


async def test_apply_disabled_makes_no_lookups(monkeypatch):
    monkeypatch.setattr(translate, "fetch_lang", AsyncMock(return_value="tr"))
    fixes = find_fixes(TR_TWEET)
    assert await translate.apply(fixes, None) == fixes
    translate.fetch_lang.assert_not_awaited()


async def test_handler_sends_translated_link(monkeypatch):
    monkeypatch.setattr(h.translate, "fetch_lang", AsyncMock(return_value="tr"))
    reply = AsyncMock()
    monkeypatch.setattr(h, "safe_reply", reply)
    monkeypatch.setattr(h, "is_server_feature_enabled", lambda guild_id, feature: True)
    monkeypatch.setattr(h, "SUPPRESS_RECHECK_DELAYS_S", ())
    perms = discord.Permissions(send_messages=True, embed_links=True, read_message_history=True, manage_messages=True)
    channel = MagicMock()
    channel.permissions_for = MagicMock(return_value=perms)
    channel.is_nsfw = MagicMock(return_value=False)
    msg = MagicMock(id=1, content=f"look {TR_TWEET}", channel=channel)
    msg.author = SimpleNamespace(bot=False, id=2)
    msg.guild = SimpleNamespace(id=3, me=SimpleNamespace(id=4))
    msg.edit = AsyncMock()

    sent = await h.maybe_fix_embeds(msg, {})
    assert reply.await_args.args[1] == "https://vxtwitter.com/nefretbeyi/status/2109002841683525719/en"
    assert [f.rewritten for f in sent] == [reply.await_args.args[1]]

    await h.maybe_fix_embeds(msg, {"EMBED_FIX_TRANSLATE_TO": ""})
    assert reply.await_args.args[1] == "https://vxtwitter.com/nefretbeyi/status/2109002841683525719"
