"""Tests for the news digest: intent detection, headline fetch, router hook.
[CA][REH][IV].
"""

from __future__ import annotations

import logging
import re

import pytest

from bot.news import headlines as hl
from bot.news.cooldown import UserCooldown
from bot.news.headlines import Headline, _build_params, _quote_topic, _to_headline, render_digest
from bot.news.intent import DAYS_TODAY, DAYS_WEEK, detect_news_intent

# --------------------------------------------------------------------------
# Intent: positives
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "what's happening in the world today?",
        "whats going on in the world today",
        "what's happening in the news?",
        "any news today?",
        "what's in the news",
        "give me the headlines",
        "what are the headlines this week",
        "catch me up",
        "anything happening around the world right now",
        "what's new in the world today",
        "current events please",
        "what's the latest in the news",
    ],
)
def test_detects_news_questions(text):
    assert detect_news_intent(text) is not None, text


# --------------------------------------------------------------------------
# Intent: negatives (false positives are the expensive failure)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "what's happening with my deploy?",
        "what's going on with this code?",
        "whats happening with my build",
        "what's up",
        "hello there",
        "can you summarise this for me",
        "what do you think about rust",
        "what's going on here?",
        "explain how the router works",
        "why did my tests fail today",
        "",
        None,
    ],
)
def test_ignores_non_news_questions(text):
    assert detect_news_intent(text) is None, text


def test_ignores_overlong_text():
    assert detect_news_intent("what's happening in the news today? " + "x" * 400) is None


def test_personal_context_beats_news_wording():
    """'my project' must win even when news words are present."""
    assert detect_news_intent("any news on my PR today?") is None
    assert detect_news_intent("what's the latest news on my deploy") is None


# --------------------------------------------------------------------------
# Intent: time windows and topics
# --------------------------------------------------------------------------


def test_today_window():
    assert detect_news_intent("what's happening in the world today").days == DAYS_TODAY


def test_week_window():
    assert detect_news_intent("what's been happening in the news this week").days == DAYS_WEEK


def test_general_query_has_no_topic():
    query = detect_news_intent("what's happening in the world today")
    assert query.topic is None
    assert query.is_general


def test_topic_extraction():
    query = detect_news_intent("any news about climate change")
    assert query is not None
    assert query.topic == "climate change"
    assert not query.is_general


def test_topic_strips_trailing_time_word():
    query = detect_news_intent("any news about artificial intelligence today")
    assert query.topic == "artificial intelligence"
    assert query.days == DAYS_TODAY


def test_scope_words_never_become_topics():
    for text in ("what's happening in the world today", "what's going on in the news right now"):
        assert detect_news_intent(text).topic is None, text


def test_absurdly_long_topic_is_dropped():
    query = detect_news_intent("any news about " + " ".join(["word"] * 12))
    assert query is not None
    assert query.topic is None


# --------------------------------------------------------------------------
# Guardian query construction
# --------------------------------------------------------------------------


def test_quote_topic_wraps_multiword():
    assert _quote_topic("artificial intelligence") == '"artificial intelligence"'


def test_quote_topic_leaves_single_word():
    assert _quote_topic("ukraine") == "ukraine"


def test_quote_topic_leaves_prequoted():
    assert _quote_topic('"already quoted"') == '"already quoted"'


def test_topic_search_ranks_by_relevance():
    """Ordering a topic search by date returns same-day noise, not the subject."""
    params = _build_params("k", "climate change", days=7, limit=5)
    assert params["order-by"] == "relevance"
    assert params["q"] == '"climate change"'
    assert "section" not in params


def test_general_search_ranks_by_date_and_filters_sections():
    params = _build_params("k", None, days=1, limit=5)
    assert params["order-by"] == "newest"
    assert "q" not in params
    assert "world" in params["section"]


def test_from_date_is_iso_and_page_size_clamped():
    params = _build_params("k", None, days=3, limit=999)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", params["from-date"])
    assert params["page-size"] == hl.MAX_LIMIT


# --------------------------------------------------------------------------
# Result parsing
# --------------------------------------------------------------------------


def test_to_headline_prefers_trailtext():
    result = {
        "webUrl": "https://example.com/a",
        "webTitle": "Fallback",
        "sectionName": "World news",
        "webPublicationDate": "2026-08-15T10:00:00Z",
        "fields": {"headline": "Real headline", "trailText": "The summary.", "bodyText": "Body."},
    }
    headline = _to_headline(result)
    assert headline.title == "Real headline"
    assert headline.summary == "The summary."
    assert headline.section == "World news"


def test_to_headline_falls_back_to_body():
    result = {"webUrl": "https://example.com/a", "webTitle": "T", "fields": {"bodyText": "B" * 900}}
    headline = _to_headline(result)
    assert headline.title == "T"
    assert len(headline.summary) <= hl.SUMMARY_CHARS


def test_to_headline_rejects_missing_url_or_title():
    assert _to_headline({"webTitle": "T"}) is None
    assert _to_headline({"webUrl": "https://example.com/a"}) is None


def test_parse_results_rejects_api_error():
    assert hl._parse_results({"response": {"status": "error"}}, None) == []


def test_parse_results_skips_bad_rows():
    payload = {"response": {"status": "ok", "results": [{"webTitle": "no url"}, {"webUrl": "u", "webTitle": "t"}]}}
    assert len(hl._parse_results(payload, None)) == 1


async def test_fetch_headlines_without_key_returns_empty():
    assert await hl.fetch_headlines(None, cfg={}) == []


# --------------------------------------------------------------------------
# Digest rendering
# --------------------------------------------------------------------------


def _headline(title="T", url="https://example.com/a"):
    return Headline(title=title, url=url, section="World news", published="2026-08-15T00:00:00Z", summary="S")


def test_render_digest_includes_titles_and_urls():
    out = render_digest([_headline("Alpha"), _headline("Beta")], None, days=1)
    assert "Alpha" in out and "Beta" in out
    assert "https://example.com/a" in out
    assert "last 24 hours" in out
    assert "newest first" in out


def test_render_digest_labels_topic_and_relevance_order():
    out = render_digest([_headline()], "climate change", days=7)
    assert 'on "climate change"' in out
    assert "most relevant first" in out
    assert "last 7 days" in out


def test_render_digest_empty_is_blank():
    assert render_digest([], None, days=1) == ""


# --------------------------------------------------------------------------
# Cooldown
# --------------------------------------------------------------------------


def test_cooldown_blocks_second_immediate_call():
    cd = UserCooldown()
    assert cd.allow(1, 60.0)
    assert not cd.allow(1, 60.0)


def test_cooldown_is_per_user():
    cd = UserCooldown()
    assert cd.allow(1, 60.0)
    assert cd.allow(2, 60.0)


def test_cooldown_zero_interval_always_allows():
    cd = UserCooldown()
    assert cd.allow(1, 0)
    assert cd.allow(1, 0)


def test_cooldown_allows_anonymous():
    cd = UserCooldown()
    assert cd.allow(None, 60.0)
    assert cd.allow(None, 60.0)


def test_cooldown_reset():
    cd = UserCooldown()
    assert cd.allow(1, 60.0)
    cd.reset(1)
    assert cd.allow(1, 60.0)


# --------------------------------------------------------------------------
# Router hook
# --------------------------------------------------------------------------


class _StubAuthor:
    def __init__(self, uid):
        self.id = uid


class _StubMessage:
    def __init__(self, uid=1):
        self.author = _StubAuthor(uid)


class _StubRouter:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("test.router.news")


def _hook(config):
    from bot.router import Router

    stub = _StubRouter(config)
    return lambda *a, **k: Router._maybe_add_news_digest(stub, *a, **k)


@pytest.fixture(autouse=True)
def _reset_shared_cooldown():
    from bot.news.cooldown import digest_cooldown

    digest_cooldown.reset()
    yield
    digest_cooldown.reset()


async def test_hook_ignores_non_news_text(monkeypatch):
    called = False

    async def _fetch(*a, **k):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(hl, "fetch_headlines", _fetch)
    result = await _hook({})("what's wrong with my deploy?", _StubMessage(), "ctx")
    assert result == "ctx"
    assert not called


async def test_hook_attaches_digest(monkeypatch):
    async def _fetch(topic, cfg=None, days=1, limit=8):
        return [_headline("Big story")]

    monkeypatch.setattr(hl, "fetch_headlines", _fetch)
    result = await _hook({})("what's happening in the world today", _StubMessage(), "ctx")
    assert "Big story" in result
    assert result.startswith("ctx")


async def test_hook_returns_context_unchanged_on_empty(monkeypatch):
    async def _fetch(topic, cfg=None, days=1, limit=8):
        return []

    monkeypatch.setattr(hl, "fetch_headlines", _fetch)
    assert await _hook({})("any news today", _StubMessage(), "ctx") == "ctx"


async def test_hook_survives_fetch_failure(monkeypatch):
    async def _boom(topic, cfg=None, days=1, limit=8):
        raise RuntimeError("upstream down")

    monkeypatch.setattr(hl, "fetch_headlines", _boom)
    assert await _hook({})("any news today", _StubMessage(), "ctx") == "ctx"


async def test_hook_respects_kill_switch(monkeypatch):
    called = False

    async def _fetch(*a, **k):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(hl, "fetch_headlines", _fetch)
    result = await _hook({"NEWS_DIGEST_ENABLED": False})("any news today", _StubMessage(), "ctx")
    assert result == "ctx"
    assert not called


async def test_hook_enforces_cooldown(monkeypatch):
    calls = 0

    async def _fetch(topic, cfg=None, days=1, limit=8):
        nonlocal calls
        calls += 1
        return [_headline()]

    monkeypatch.setattr(hl, "fetch_headlines", _fetch)
    hook = _hook({"NEWS_DIGEST_COOLDOWN_S": 300.0})
    await hook("any news today", _StubMessage(7), "")
    await hook("any news today", _StubMessage(7), "")
    assert calls == 1, "second request within the cooldown must not hit the API"


async def test_hook_cooldown_is_per_user(monkeypatch):
    calls = 0

    async def _fetch(topic, cfg=None, days=1, limit=8):
        nonlocal calls
        calls += 1
        return [_headline()]

    monkeypatch.setattr(hl, "fetch_headlines", _fetch)
    hook = _hook({"NEWS_DIGEST_COOLDOWN_S": 300.0})
    await hook("any news today", _StubMessage(1), "")
    await hook("any news today", _StubMessage(2), "")
    assert calls == 2


async def test_hook_handles_empty_context(monkeypatch):
    async def _fetch(topic, cfg=None, days=1, limit=8):
        return [_headline("Solo")]

    monkeypatch.setattr(hl, "fetch_headlines", _fetch)
    result = await _hook({})("any news today", _StubMessage(), "")
    assert "Solo" in result
    assert not result.startswith("\n")


# --------------------------------------------------------------------------
# Config registration
# --------------------------------------------------------------------------


def _getter(values):
    def get(key, default=None):
        return values.get(key, default)

    return get


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("NEWS_DIGEST_ENABLED", True),
        ("NEWS_DIGEST_LIMIT", 8),
        ("NEWS_DIGEST_COOLDOWN_S", 30.0),
        ("NEWS_DIGEST_TIMEOUT_S", 12.0),
    ],
)
def test_digest_config_defaults(key, expected):
    from bot.config._base import _build_config

    assert _build_config(_getter({}))[key] == expected


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("NEWS_DIGEST_LIMIT", "15", 15),
        ("NEWS_DIGEST_COOLDOWN_S", "5.5", 5.5),
        ("NEWS_DIGEST_TIMEOUT_S", "20.0", 20.0),
        ("NEWS_DIGEST_ENABLED", "false", False),
    ],
)
def test_digest_config_reads_env(key, raw, expected):
    from bot.config._base import _build_config

    assert _build_config(_getter({key: raw}))[key] == expected


# --------------------------------------------------------------------------
# Cog wiring
# --------------------------------------------------------------------------


def test_news_cog_is_registered_in_loader():
    """A cog absent from module_definitions never loads at runtime."""
    from pathlib import Path

    source = Path("bot/core/bot.py").read_text()
    assert '("news_commands", "NewsCommands")' in source


def test_news_embed_respects_discord_limits():
    from bot.commands.news_commands import MAX_FIELDS, _build_embed

    many = [_headline(title="T" * 400, url="https://example.com/" + "u" * 400) for _ in range(20)]
    embed = _build_embed(many, "topic", days=7)
    assert len(embed.fields) == MAX_FIELDS
    for field in embed.fields:
        assert len(field.name) <= 256
        assert len(field.value) <= 1024
