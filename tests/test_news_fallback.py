"""Tests for the licensed news fallback chain.
[CA][REH][IV].

Covers stub detection, provider URL mapping, feed parsing, the orchestrator's
accept/reject rules, config registration, and the router helper that ties them
together.
"""

from __future__ import annotations

import logging

import pytest

from bot.news import feed_parse, service, thin_content
from bot.news.providers.guardian import GuardianNewsProvider, content_id_from_url
from bot.news.providers.rss import discover_feeds, well_known_feeds
from bot.news.types import NewsArticle

# --------------------------------------------------------------------------
# thin_content
# --------------------------------------------------------------------------

LONG_ARTICLE = "The council met on Tuesday to debate the proposal. " * 40


def test_empty_text_is_thin():
    verdict = thin_content.assess("")
    assert verdict.is_thin
    assert verdict.reason == "empty"


def test_none_text_is_thin():
    assert thin_content.assess(None).is_thin


def test_short_text_is_thin():
    verdict = thin_content.assess("Headline only.")
    assert verdict.is_thin
    assert verdict.reason == "below_min_chars"


def test_long_article_is_not_thin():
    verdict = thin_content.assess(LONG_ARTICLE)
    assert not verdict.is_thin
    assert verdict.reason == "ok"
    assert bool(verdict) is False


def test_subscription_marker_in_short_body_is_thin():
    body = "Big story broken today. " * 40 + " Subscribe to continue reading this article."
    verdict = thin_content.assess(body)
    assert verdict.is_thin
    assert verdict.reason.startswith("subscription_marker:")


def test_marker_in_long_article_is_not_thin():
    """A footer promo on a genuinely long article must not trigger recovery."""
    body = LONG_ARTICLE * 3 + " Already a subscriber? Sign in."
    assert len(body) > thin_content.MARKER_MAX_CHARS
    assert not thin_content.assess(body).is_thin


def test_min_chars_is_configurable():
    text = "word " * 50  # 250 chars
    assert thin_content.assess(text, min_chars=1000).is_thin
    assert not thin_content.assess(text, min_chars=100).is_thin


# --------------------------------------------------------------------------
# Guardian URL -> content id
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.theguardian.com/world/2024/jan/01/some-slug", "world/2024/jan/01/some-slug"),
        ("https://amp.theguardian.com/world/2024/jan/01/some-slug/amp", "world/2024/jan/01/some-slug"),
        ("https://www.theguardian.com/politics/2023/dec/25/a/b/c", "politics/2023/dec/25/a/b/c"),
    ],
)
def test_content_id_from_article_url(url, expected):
    assert content_id_from_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://www.theguardian.com/world",  # section front, no article body
        "https://www.theguardian.com/world/2024",
        "https://www.bbc.co.uk/news/uk-123",  # different publisher
        "not-a-url",
    ],
)
def test_content_id_rejects_non_articles(url):
    assert content_id_from_url(url) is None


def test_guardian_disabled_without_key():
    provider = GuardianNewsProvider(None)
    assert not provider.enabled
    assert not provider.supports("https://www.theguardian.com/world/2024/jan/01/x")


def test_guardian_supports_with_key():
    provider = GuardianNewsProvider("k")
    assert provider.enabled
    assert provider.supports("https://www.theguardian.com/world/2024/jan/01/x")
    assert not provider.supports("https://example.com/story")


async def test_guardian_fetch_without_key_returns_none():
    assert await GuardianNewsProvider("").fetch("https://www.theguardian.com/world/2024/jan/01/x") is None


def test_guardian_to_article_from_payload():
    provider = GuardianNewsProvider("k")
    payload = {
        "response": {
            "status": "ok",
            "content": {
                "webUrl": "https://www.theguardian.com/world/2024/jan/01/x",
                "webTitle": "Fallback title",
                "fields": {
                    "bodyText": "Full licensed body text.",
                    "headline": "Real headline",
                    "byline": "A Reporter",
                    "firstPublicationDate": "2024-01-01T00:00:00Z",
                },
            },
        }
    }
    article = provider._to_article(payload, "https://example.com")
    assert article is not None
    assert article.body == "Full licensed body text."
    assert article.title == "Real headline"
    assert article.author == "A Reporter"
    assert article.provider == "guardian"


def test_guardian_to_article_rejects_error_status():
    provider = GuardianNewsProvider("k")
    assert provider._to_article({"response": {"status": "error"}}, "u") is None
    assert provider._to_article({"response": {"status": "ok", "content": {}}}, "u") is None


# --------------------------------------------------------------------------
# Feed parsing
# --------------------------------------------------------------------------

RSS_XML = b"""<?xml version="1.0"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"
     xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <item>
      <title>Other story</title>
      <link>https://example.com/news/other</link>
      <description>Nope.</description>
    </item>
    <item>
      <title>Target story</title>
      <link>https://example.com/news/target</link>
      <dc:creator>Jane Doe</dc:creator>
      <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
      <content:encoded>&lt;p&gt;The &lt;b&gt;full&lt;/b&gt; body.&lt;/p&gt;</content:encoded>
    </item>
  </channel>
</rss>
"""

ATOM_XML = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Atom story</title>
    <link rel="alternate" href="https://example.com/news/atom"/>
    <author><name>Atom Author</name></author>
    <updated>2024-01-01T00:00:00Z</updated>
    <content>&lt;p&gt;Atom body text.&lt;/p&gt;</content>
  </entry>
</feed>
"""


def test_parse_rss_extracts_content_encoded():
    entries = feed_parse.parse_feed(RSS_XML)
    assert len(entries) == 2
    target = feed_parse.find_entry(entries, "https://example.com/news/target")
    assert target is not None
    assert target.body == "The full body."
    assert target.author == "Jane Doe"
    assert target.title == "Target story"


def test_parse_atom_entry():
    entries = feed_parse.parse_feed(ATOM_XML)
    assert len(entries) == 1
    assert entries[0].body == "Atom body text."
    assert entries[0].author == "Atom Author"


def test_find_entry_ignores_cosmetic_url_differences():
    entries = feed_parse.parse_feed(RSS_XML)
    for variant in (
        "https://www.example.com/news/target",
        "http://example.com/news/target/",
        "https://example.com/news/target?utm_source=x",
        "https://amp.example.com/news/target/amp",
    ):
        assert feed_parse.find_entry(entries, variant) is not None, variant


def test_find_entry_misses_different_article():
    entries = feed_parse.parse_feed(RSS_XML)
    assert feed_parse.find_entry(entries, "https://example.com/news/unrelated") is None


def test_parse_feed_tolerates_garbage():
    assert feed_parse.parse_feed(b"") == []
    assert feed_parse.parse_feed(b"not xml at all") == []


def test_parse_feed_blocks_entity_expansion():
    """A billion-laughs payload must not expand. [SFT]"""
    bomb = b"""<?xml version="1.0"?>
    <!DOCTYPE rss [
      <!ENTITY a "AAAAAAAAAA">
      <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">
      <!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">
    ]>
    <rss><channel><item><link>u</link><description>&c;</description></item></channel></rss>
    """
    entries = feed_parse.parse_feed(bomb)
    # Either rejected outright or parsed with the entity left unexpanded.
    for entry in entries:
        assert "AAAAAAAAAA" not in entry.body


# --------------------------------------------------------------------------
# Feed discovery
# --------------------------------------------------------------------------


def test_discover_feeds_from_link_tags():
    html = """
    <html><head>
      <link rel="alternate" type="application/rss+xml" href="/feed.xml">
      <link rel="alternate" type="application/atom+xml" href="https://cdn.example.com/atom">
      <link rel="alternate" type="text/html" href="/nope">
      <link rel="stylesheet" href="/style.css">
    </head></html>
    """
    feeds = discover_feeds(html, "https://example.com/news/story")
    assert feeds == ["https://example.com/feed.xml", "https://cdn.example.com/atom"]


def test_discover_feeds_handles_empty_html():
    assert discover_feeds("", "https://example.com") == []


def test_well_known_feeds_uses_origin():
    feeds = well_known_feeds("https://example.com/a/b/c?x=1")
    assert "https://example.com/feed" in feeds
    assert all(f.startswith("https://example.com/") for f in feeds)


def test_well_known_feeds_rejects_bad_url():
    assert well_known_feeds("not-a-url") == []


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


class _FakeProvider:
    def __init__(self, name, article=None, exc=None):
        self.name = name
        self._article = article
        self._exc = exc
        self.calls = 0

    def supports(self, url):
        return True

    async def fetch(self, url):
        self.calls += 1
        if self._exc:
            raise self._exc
        return self._article


def _article(body, provider="fake"):
    return NewsArticle(url="https://example.com/a", body=body, provider=provider)


async def test_resolve_returns_first_good_article(monkeypatch):
    good = _FakeProvider("good", _article(LONG_ARTICLE))
    second = _FakeProvider("second", _article(LONG_ARTICLE))
    monkeypatch.setattr(service, "build_providers", lambda cfg: [good, second])

    result = await service.resolve_article("https://example.com/a", cfg={})
    assert result is not None
    assert result.provider == "fake"
    assert second.calls == 0, "must stop at the first usable article"


async def test_resolve_skips_thin_provider_body(monkeypatch):
    thin = _FakeProvider("thin", _article("too short"))
    good = _FakeProvider("good", _article(LONG_ARTICLE))
    monkeypatch.setattr(service, "build_providers", lambda cfg: [thin, good])

    result = await service.resolve_article("https://example.com/a", cfg={})
    assert result is not None
    assert good.calls == 1


async def test_resolve_survives_provider_exception(monkeypatch):
    boom = _FakeProvider("boom", exc=RuntimeError("provider down"))
    good = _FakeProvider("good", _article(LONG_ARTICLE))
    monkeypatch.setattr(service, "build_providers", lambda cfg: [boom, good])

    result = await service.resolve_article("https://example.com/a", cfg={})
    assert result is not None


async def test_resolve_returns_none_when_all_fail(monkeypatch):
    monkeypatch.setattr(service, "build_providers", lambda cfg: [_FakeProvider("a"), _FakeProvider("b")])
    assert await service.resolve_article("https://example.com/a", cfg={}) is None


async def test_resolve_respects_kill_switch(monkeypatch):
    provider = _FakeProvider("never", _article(LONG_ARTICLE))
    monkeypatch.setattr(service, "build_providers", lambda cfg: [provider])

    result = await service.resolve_article("https://example.com/a", cfg={"NEWS_FALLBACK_ENABLED": False})
    assert result is None
    assert provider.calls == 0


async def test_resolve_rejects_empty_url():
    assert await service.resolve_article("", cfg={}) is None


def test_build_providers_omits_guardian_without_key():
    names = [p.name for p in service.build_providers({"NEWS_RSS_ENABLED": True})]
    assert "guardian" not in names
    assert "rss" in names


def test_build_providers_includes_guardian_with_key():
    names = [p.name for p in service.build_providers({"GUARDIAN_API_KEY": "k"})]
    assert names[0] == "guardian", "licensed full-text source must be tried first"


def test_build_providers_respects_rss_toggle():
    names = [p.name for p in service.build_providers({"NEWS_RSS_ENABLED": False})]
    assert names == []


# --------------------------------------------------------------------------
# NewsArticle rendering
# --------------------------------------------------------------------------


def test_capped_body_truncates_on_word_boundary():
    article = _article("word " * 500)
    capped = article.capped_body(max_chars=100)
    assert len(capped) <= 101
    assert capped.endswith("…")


def test_capped_body_leaves_short_body_intact():
    article = _article("short body")
    assert article.capped_body(max_chars=100) == "short body"


def test_to_message_includes_provenance():
    article = NewsArticle(
        url="https://example.com/a",
        body="Body.",
        title="T",
        author="A",
        provider="guardian",
    )
    msg = article.to_message()
    assert "Title: T" in msg
    assert "Author: A" in msg
    assert "guardian" in msg
    assert "Body." in msg


# --------------------------------------------------------------------------
# Config registration (guards against the b719ad1 class of bug)
# --------------------------------------------------------------------------


def _getter(values):
    def get(key, default=None):
        return values.get(key, default)

    return get


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("NEWS_MIN_ARTICLE_CHARS", "1500", 1500),
        ("NEWS_MAX_BODY_CHARS", "9000", 9000),
        ("NEWS_FETCH_TIMEOUT_S", "12.5", 12.5),
        ("GUARDIAN_API_KEY", "abc123", "abc123"),
        ("URL_PROCESS_TIMEOUT_S", "40.0", 40.0),
        ("WEB_EXTRACT_TIMEOUT_S", "45.0", 45.0),
    ],
)
def test_news_config_keys_reach_config(key, raw, expected):
    from bot.config._base import _build_config

    cfg = _build_config(_getter({key: raw}))
    assert cfg[key] == expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("NEWS_FALLBACK_ENABLED", True),
        ("NEWS_RSS_ENABLED", True),
        ("NEWS_MIN_ARTICLE_CHARS", 800),
        ("NEWS_MAX_BODY_CHARS", 6000),
        ("NEWS_FETCH_TIMEOUT_S", 8.0),
        ("URL_PROCESS_TIMEOUT_S", 25.0),
        ("WEB_EXTRACT_TIMEOUT_S", 30.0),
    ],
)
def test_news_config_defaults_when_unset(key, expected):
    from bot.config._base import _build_config

    cfg = _build_config(_getter({}))
    assert cfg[key] == expected


def test_news_config_strips_inline_comment():
    from bot.config._base import _build_config

    cfg = _build_config(_getter({"GUARDIAN_API_KEY": "abc  # my key"}))
    assert cfg["GUARDIAN_API_KEY"] == "abc"


def test_news_config_falls_back_on_malformed_number():
    from bot.config._base import _build_config

    cfg = _build_config(_getter({"NEWS_MIN_ARTICLE_CHARS": "not-a-number"}))
    assert cfg["NEWS_MIN_ARTICLE_CHARS"] == 800


# --------------------------------------------------------------------------
# Router helper
# --------------------------------------------------------------------------


class _StubRouter:
    """Minimal stand-in carrying only what _finalize_web_content touches."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("test.router")


def _bind(config):
    from bot.router import Router

    stub = _StubRouter(config)
    return lambda *a, **k: Router._finalize_web_content(stub, *a, **k), stub


async def test_router_passes_through_full_article(monkeypatch):
    from bot import news

    called = False

    async def _never(*a, **k):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(news, "resolve_article", _never)
    finalize, _ = _bind({})

    result = await finalize("https://example.com/a", LONG_ARTICLE)
    assert result is not None
    assert "Web content from https://example.com/a" in result
    assert not called, "a healthy article must not trigger the news chain"


async def test_router_recovers_thin_content_from_news(monkeypatch):
    from bot import news

    async def _resolve(url, cfg=None, min_chars=None, timeout_s=None):
        return NewsArticle(url=url, body=LONG_ARTICLE, title="Real", provider="guardian")

    monkeypatch.setattr(news, "resolve_article", _resolve)
    finalize, _ = _bind({})

    result = await finalize("https://example.com/a", "Subscribe to continue reading.")
    assert result is not None
    assert "via guardian" in result
    assert "Real" in result


async def test_router_labels_partial_when_news_misses(monkeypatch):
    from bot import news

    async def _resolve(url, cfg=None, min_chars=None, timeout_s=None):
        return None

    monkeypatch.setattr(news, "resolve_article", _resolve)
    finalize, _ = _bind({})

    stub_text = "Headline. Subscribe to continue reading."
    result = await finalize("https://example.com/a", stub_text)
    assert result is not None
    assert "Partial content only" in result
    assert stub_text in result, "must not silently drop what was extracted"


async def test_router_returns_none_when_nothing_available(monkeypatch):
    from bot import news

    async def _resolve(url, cfg=None, min_chars=None, timeout_s=None):
        return None

    monkeypatch.setattr(news, "resolve_article", _resolve)
    finalize, _ = _bind({})

    assert await finalize("https://example.com/a", "") is None
    assert await finalize("https://example.com/a", None) is None
