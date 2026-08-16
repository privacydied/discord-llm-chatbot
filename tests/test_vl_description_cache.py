"""Tests for VL description caching and the image-download User-Agent.
[PA][REH][CMV][SFT].

The cache module had no production callers before this, so these tests
validate the cache's own behaviour as well as our use of it.
"""

from __future__ import annotations

import pytest

from bot.single_flight_cache import CacheFamily, SingleFlightCache
from bot.tools.builtins import vision
from bot.tools.builtins.vision import cache_identity

# --------------------------------------------------------------------------
# Cache key identity
# --------------------------------------------------------------------------

DISCORD_A = "https://cdn.discordapp.com/attachments/1/2/cat.png?ex=aaa&is=bbb&hm=ccc&"
DISCORD_B = "https://cdn.discordapp.com/attachments/1/2/cat.png?ex=zzz&is=yyy&hm=xxx&"


def test_expiring_discord_signature_ignored():
    """Discord re-signs URLs; the same image must still hit the cache."""
    assert cache_identity(DISCORD_A) == cache_identity(DISCORD_B)


def test_different_images_keep_different_identities():
    other = "https://cdn.discordapp.com/attachments/1/2/dog.png?ex=aaa&is=bbb&hm=ccc"
    assert cache_identity(DISCORD_A) != cache_identity(other)


def test_meaningful_query_params_are_preserved():
    """Hosts use query params to pick a rendition — those must not be dropped."""
    small = "https://media.discordapp.net/x/y.png?format=png&size=512"
    large = "https://media.discordapp.net/x/y.png?format=png&size=4096"
    assert cache_identity(small) != cache_identity(large)


def test_query_order_does_not_matter():
    one = "https://example.com/i.png?b=2&a=1"
    two = "https://example.com/i.png?a=1&b=2"
    assert cache_identity(one) == cache_identity(two)


def test_fragment_ignored():
    assert cache_identity("https://example.com/i.png#top") == cache_identity("https://example.com/i.png")


def test_malformed_url_does_not_raise():
    assert cache_identity("not a url") is not None


# --------------------------------------------------------------------------
# Cache behaviour through _describe
# --------------------------------------------------------------------------


@pytest.fixture
def _fresh_cache(monkeypatch):
    """Give each test its own cache so entries cannot leak between them."""
    cache = SingleFlightCache({})
    monkeypatch.setattr("bot.single_flight_cache.get_cache", lambda config=None: cache)

    async def _allow(url):
        return None

    monkeypatch.setattr("bot.url_safety.validate_url_with_dns", _allow)
    return cache


def _count_calls(monkeypatch, result="a description", fail=False):
    calls = {"n": 0}

    async def _vl(url, question):
        calls["n"] += 1
        if fail:
            raise vision._VisionUnavailable("model down")
        return f"{result} for {question}"

    monkeypatch.setattr(vision, "_run_vl", _vl)
    return calls


async def test_second_identical_request_is_served_from_cache(_fresh_cache, monkeypatch):
    calls = _count_calls(monkeypatch)
    first = await vision._describe(DISCORD_A, "describe this", {})
    second = await vision._describe(DISCORD_A, "describe this", {})
    assert first == second
    assert calls["n"] == 1, "the second ask must not re-run inference"


async def test_resigned_discord_url_still_hits(_fresh_cache, monkeypatch):
    """The real-world case: same image, refreshed signature."""
    calls = _count_calls(monkeypatch)
    await vision._describe(DISCORD_A, "describe this", {})
    await vision._describe(DISCORD_B, "describe this", {})
    assert calls["n"] == 1


async def test_different_question_recomputes(_fresh_cache, monkeypatch):
    """A cached 'describe this' is the wrong answer to a specific question."""
    calls = _count_calls(monkeypatch)
    await vision._describe(DISCORD_A, "describe this", {})
    await vision._describe(DISCORD_A, "what colour is the car?", {})
    assert calls["n"] == 2


async def test_question_matching_is_case_insensitive(_fresh_cache, monkeypatch):
    calls = _count_calls(monkeypatch)
    await vision._describe(DISCORD_A, "Describe This", {})
    await vision._describe(DISCORD_A, "describe this", {})
    assert calls["n"] == 1


async def test_failures_are_never_cached(_fresh_cache, monkeypatch):
    """A transient model outage must not poison the entry for 24 hours."""
    calls = _count_calls(monkeypatch, fail=True)
    assert await vision._describe(DISCORD_A, "describe this", {}) is None
    assert await vision._describe(DISCORD_A, "describe this", {}) is None
    assert calls["n"] == 2, "each attempt must retry, not replay a cached failure"


async def test_success_after_failure_is_cached(_fresh_cache, monkeypatch):
    state = {"n": 0}

    async def _vl(url, question):
        state["n"] += 1
        if state["n"] == 1:
            raise vision._VisionUnavailable("first attempt fails")
        return "recovered description"

    monkeypatch.setattr(vision, "_run_vl", _vl)
    assert await vision._describe(DISCORD_A, "q", {}) is None
    assert await vision._describe(DISCORD_A, "q", {}) == "recovered description"
    assert await vision._describe(DISCORD_A, "q", {}) == "recovered description"
    assert state["n"] == 2


async def test_blocked_url_never_reaches_the_model(_fresh_cache, monkeypatch):
    from bot.url_safety import UrlSafetyError

    calls = _count_calls(monkeypatch)

    async def _block(url):
        raise UrlSafetyError("private address")

    monkeypatch.setattr("bot.url_safety.validate_url_with_dns", _block)
    assert await vision._describe("http://127.0.0.1/x.png", "q", {}) is None
    assert calls["n"] == 0


async def test_cache_outage_falls_back_to_direct_inference(monkeypatch):
    """The cache must never be load-bearing."""
    calls = _count_calls(monkeypatch)

    async def _allow(url):
        return None

    monkeypatch.setattr("bot.url_safety.validate_url_with_dns", _allow)

    def _boom(config=None):
        raise RuntimeError("cache exploded")

    monkeypatch.setattr("bot.single_flight_cache.get_cache", _boom)
    assert await vision._describe(DISCORD_A, "q", {}) is not None
    assert calls["n"] == 1


# --------------------------------------------------------------------------
# The cache module itself — first production user, so verify its contract
# --------------------------------------------------------------------------


async def test_family_registered_with_expected_ttl():
    cache = SingleFlightCache({})
    assert CacheFamily.VL_DESCRIPTION in cache.family_ttls
    assert cache.family_ttls[CacheFamily.VL_DESCRIPTION] == 86400.0


async def test_ttl_is_configurable():
    cache = SingleFlightCache({"VL_DESCRIPTION_TTL_S": 60})
    assert cache.family_ttls[CacheFamily.VL_DESCRIPTION] == 60.0


async def test_get_or_compute_reports_hit_flag():
    cache = SingleFlightCache({})

    async def _compute():
        return "value"

    first, hit_one = await cache.get_or_compute(CacheFamily.VL_DESCRIPTION, ["k"], _compute)
    second, hit_two = await cache.get_or_compute(CacheFamily.VL_DESCRIPTION, ["k"], _compute)
    assert (first, second) == ("value", "value")
    assert hit_one is False
    assert hit_two is True


async def test_negative_caching_disabled_means_nothing_stored():
    cache = SingleFlightCache({})
    attempts = {"n": 0}

    async def _fail():
        attempts["n"] += 1
        raise RuntimeError("nope")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cache.get_or_compute(CacheFamily.VL_DESCRIPTION, ["k"], _fail, negative_on_exception=False)
    assert attempts["n"] == 2


async def test_distinct_key_parts_do_not_collide():
    cache = SingleFlightCache({})

    async def _a():
        return "A"

    async def _b():
        return "B"

    first, _ = await cache.get_or_compute(CacheFamily.VL_DESCRIPTION, ["img", "q1"], _a)
    second, _ = await cache.get_or_compute(CacheFamily.VL_DESCRIPTION, ["img", "q2"], _b)
    assert (first, second) == ("A", "B")


async def test_disabled_cache_always_computes():
    cache = SingleFlightCache({"CACHE_SINGLE_FLIGHT_ENABLE": False})
    calls = {"n": 0}

    async def _compute():
        calls["n"] += 1
        return "v"

    await cache.get_or_compute(CacheFamily.VL_DESCRIPTION, ["k"], _compute)
    await cache.get_or_compute(CacheFamily.VL_DESCRIPTION, ["k"], _compute)
    assert calls["n"] == 2


# --------------------------------------------------------------------------
# Image download headers
# --------------------------------------------------------------------------


def test_image_download_sends_a_user_agent():
    from bot.openai_backend import IMAGE_DOWNLOAD_HEADERS

    ua = IMAGE_DOWNLOAD_HEADERS["User-Agent"]
    assert ua
    assert "python" not in ua.lower(), "the library default is what hosts reject"


def test_user_agent_identifies_honestly_rather_than_spoofing_a_browser():
    """Wikimedia rejects browser-spoofing UAs under its robot policy."""
    from bot.openai_backend import IMAGE_DOWNLOAD_HEADERS

    ua = IMAGE_DOWNLOAD_HEADERS["User-Agent"].lower()
    for spoof in ("mozilla", "chrome", "safari", "applewebkit", "gecko"):
        assert spoof not in ua, f"UA should not impersonate a browser (found {spoof!r})"
    assert "discord-llm-chatbot" in ua


def test_user_agent_is_overridable(monkeypatch):
    monkeypatch.setenv("IMAGE_DOWNLOAD_UA", "mybot/2.0 (+https://example.org)")
    import importlib

    import bot.openai_backend as backend

    importlib.reload(backend)
    try:
        assert backend.IMAGE_DOWNLOAD_HEADERS["User-Agent"] == "mybot/2.0 (+https://example.org)"
    finally:
        monkeypatch.delenv("IMAGE_DOWNLOAD_UA", raising=False)
        importlib.reload(backend)


def test_accept_header_prefers_images():
    from bot.openai_backend import IMAGE_DOWNLOAD_HEADERS

    assert "image/" in IMAGE_DOWNLOAD_HEADERS["Accept"]
