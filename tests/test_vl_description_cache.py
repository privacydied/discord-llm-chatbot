"""Tests for VL description caching and the image-download User-Agent.
[PA][REH][CMV][SFT].

The cache module had no production callers before this, so these tests
validate the cache's own behaviour as well as our use of it.
"""

from __future__ import annotations

import pytest

from bot.single_flight_cache import CacheFamily, SingleFlightCache
from bot.tools.builtins import vision
from bot.tools.builtins.vision import attachment_token, cache_identity, normalize_url

# --------------------------------------------------------------------------
# Cache key identity — keyed on (message_id, attachment_id)
# --------------------------------------------------------------------------

# Same attachment (id 2222), different expiring signature.
DISCORD_A = "https://cdn.discordapp.com/attachments/1111/2222/cat.png?ex=aaa&is=bbb&hm=ccc&"
DISCORD_B = "https://cdn.discordapp.com/attachments/1111/2222/cat.png?ex=zzz&is=yyy&hm=xxx&"


class _Ref:
    def __init__(self, url, filename="cat.png"):
        self.url = url
        self.filename = filename


class _Msg:
    def __init__(self, id_=99):
        self.id = id_


def test_attachment_id_extracted_from_discord_url():
    assert attachment_token(DISCORD_A) == "2222"


def test_attachment_token_falls_back_to_filename():
    token = attachment_token("https://example.com/some/path/img.png", "img.png")
    assert token == "img.png"


def test_attachment_token_falls_back_to_url_without_filename():
    token = attachment_token("https://example.com/some/path/img.png", None)
    assert token.startswith("https://example.com")


def test_identity_uses_message_and_attachment_ids():
    identity = cache_identity(_Msg(777), _Ref(DISCORD_A))
    assert identity == "m777:2222"


def test_resigned_url_yields_the_same_identity():
    """Discord re-signs URLs; the same attachment must still hit."""
    assert cache_identity(_Msg(5), _Ref(DISCORD_A)) == cache_identity(_Msg(5), _Ref(DISCORD_B))


def test_identity_survives_a_completely_rotated_url():
    """The whole point of message+attachment keying over URL keying."""
    rotated = "https://cdn.discordapp.com/attachments/1111/2222/cat.png?totally=different&params=here"
    assert cache_identity(_Msg(5), _Ref(DISCORD_A)) == cache_identity(_Msg(5), _Ref(rotated))


def test_different_attachments_differ():
    other = "https://cdn.discordapp.com/attachments/1111/3333/dog.png?ex=aaa"
    assert cache_identity(_Msg(5), _Ref(DISCORD_A)) != cache_identity(_Msg(5), _Ref(other))


def test_same_attachment_in_different_messages_differs():
    assert cache_identity(_Msg(1), _Ref(DISCORD_A)) != cache_identity(_Msg(2), _Ref(DISCORD_A))


def test_two_images_on_one_message_differ():
    """A message with several attachments must not collapse to one entry."""
    first = _Ref("https://cdn.discordapp.com/attachments/1/10/a.png", "a.png")
    second = _Ref("https://cdn.discordapp.com/attachments/1/11/b.png", "b.png")
    assert cache_identity(_Msg(5), first) != cache_identity(_Msg(5), second)


def test_external_embed_images_on_one_message_differ_by_filename():
    first = _Ref("https://example.com/a.png", "a.png")
    second = _Ref("https://example.com/b.png", "b.png")
    assert cache_identity(_Msg(5), first) != cache_identity(_Msg(5), second)


def test_falls_back_to_url_when_message_has_no_id():
    identity = cache_identity(object(), _Ref(DISCORD_A))
    assert identity.startswith("u:")


def test_malformed_url_does_not_raise():
    assert cache_identity(_Msg(1), _Ref("not a url", None)) is not None


# --- URL normalisation, still used as the fallback key ---------------------


def test_normalize_strips_expiring_params():
    assert normalize_url(DISCORD_A) == normalize_url(DISCORD_B)


def test_normalize_keeps_rendition_params():
    small = "https://media.discordapp.net/x/y.png?format=png&size=512"
    large = "https://media.discordapp.net/x/y.png?format=png&size=4096"
    assert normalize_url(small) != normalize_url(large)


def test_normalize_is_query_order_insensitive():
    assert normalize_url("https://example.com/i.png?b=2&a=1") == normalize_url("https://example.com/i.png?a=1&b=2")


def test_normalize_drops_fragment():
    assert normalize_url("https://example.com/i.png#top") == normalize_url("https://example.com/i.png")


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


IDENTITY = "m777:2222"


async def test_second_identical_request_is_served_from_cache(_fresh_cache, monkeypatch):
    calls = _count_calls(monkeypatch)
    first = await vision._describe(DISCORD_A, "describe this", {}, IDENTITY)
    second = await vision._describe(DISCORD_A, "describe this", {}, IDENTITY)
    assert first == second
    assert calls["n"] == 1, "the second ask must not re-run inference"


async def test_same_identity_hits_even_when_the_url_changed(_fresh_cache, monkeypatch):
    """The real-world case: same attachment, entirely different URL."""
    calls = _count_calls(monkeypatch)
    await vision._describe(DISCORD_A, "describe this", {}, IDENTITY)
    await vision._describe("https://cdn.discordapp.com/totally/other?x=1", "describe this", {}, IDENTITY)
    assert calls["n"] == 1


async def test_different_identity_recomputes(_fresh_cache, monkeypatch):
    calls = _count_calls(monkeypatch)
    await vision._describe(DISCORD_A, "describe this", {}, "m1:aaa")
    await vision._describe(DISCORD_A, "describe this", {}, "m2:bbb")
    assert calls["n"] == 2


async def test_url_fallback_still_caches_without_identity(_fresh_cache, monkeypatch):
    """Synthetic contexts with no message id must still benefit."""
    calls = _count_calls(monkeypatch)
    await vision._describe(DISCORD_A, "describe this", {})
    await vision._describe(DISCORD_B, "describe this", {})
    assert calls["n"] == 1


async def test_different_question_recomputes(_fresh_cache, monkeypatch):
    """A cached 'describe this' is the wrong answer to a specific question."""
    calls = _count_calls(monkeypatch)
    await vision._describe(DISCORD_A, "describe this", {}, IDENTITY)
    await vision._describe(DISCORD_A, "what colour is the car?", {}, IDENTITY)
    assert calls["n"] == 2


async def test_question_matching_is_case_insensitive(_fresh_cache, monkeypatch):
    calls = _count_calls(monkeypatch)
    await vision._describe(DISCORD_A, "Describe This", {}, IDENTITY)
    await vision._describe(DISCORD_A, "describe this", {}, IDENTITY)
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
# End to end through view_image, so the identity really is wired up
# --------------------------------------------------------------------------


class _E2EAuthor:
    display_name = "alice"
    name = "alice"


class _E2EMessage:
    def __init__(self, id_, url):
        self.id = id_
        self.author = _E2EAuthor()
        self.created_at = None
        self.content = ""
        self._url = url


class _E2EChannel:
    def __init__(self, messages):
        self._messages = messages

    def history(self, limit=None, before=None):
        messages = self._messages[:limit]

        async def _gen():
            for m in messages:
                yield m

        return _gen()


async def test_view_image_twice_runs_inference_once(_fresh_cache, monkeypatch):
    """The goldfish fix must not pay twice for the same picture."""
    from bot.tools import ToolContext
    from bot.tools.builtins.vision import view_image

    calls = _count_calls(monkeypatch)
    monkeypatch.setattr(vision, "_image_refs", lambda msg: [_Ref(msg._url)])

    posted = _E2EMessage(4242, DISCORD_A)
    current = _E2EMessage(1, "")
    current.channel = _E2EChannel([posted])
    ctx = ToolContext(message=current, bot=None, config={})

    first = await view_image(ctx, {})
    second = await view_image(ctx, {})
    assert first.ok and second.ok
    assert calls["n"] == 1, "the repeat look must be served from cache"


async def test_view_image_hits_cache_after_discord_resigns_the_url(_fresh_cache, monkeypatch):
    from bot.tools import ToolContext
    from bot.tools.builtins.vision import view_image

    calls = _count_calls(monkeypatch)
    urls = iter([DISCORD_A, DISCORD_B])
    monkeypatch.setattr(vision, "_image_refs", lambda msg: [_Ref(next(urls))])

    posted = _E2EMessage(4242, DISCORD_A)
    current = _E2EMessage(1, "")
    current.channel = _E2EChannel([posted])
    ctx = ToolContext(message=current, bot=None, config={})

    await view_image(ctx, {})
    await view_image(ctx, {})
    assert calls["n"] == 1, "a re-signed URL is the same attachment"


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
