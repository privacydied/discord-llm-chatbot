from types import SimpleNamespace

import pytest

from bot.router_components.x_routing import (
    canonicalize_twitter_status_url,
    collect_x_candidate_urls,
    extract_fxtwitter_tweet_node,
    extract_x_api_primary_text,
    extract_x_api_primary_tweet,
    extract_sparse_media_resolution,
    extract_raw_urls_from_texts,
    extract_x_status_urls_from_text,
    filter_canonical_x_urls,
    is_tweet_media_url,
    is_twitter_media_cdn,
    is_twitter_thumbnail_url,
    is_twitter_url,
    normalize_x_url,
    parse_twitter_status_id,
    classify_stt_error_reason,
    resolve_twitter_status_id,
    is_twitter_status_url,
    stt_result_has_transcription,
    unwrap_x_media_url,
    x_syn_probe_budget_timeout_s,
    x_syn_quick_request_timeouts,
    build_syndication_photo_payload,
    format_twitter_syndication_images_log_line,
    resolve_and_probe_twitter_images,
)


def test_is_twitter_url_and_status_id_parsing() -> None:
    url = "https://x.com/user/status/2022790791047823773?s=20"
    assert is_twitter_url(url) is True
    assert parse_twitter_status_id(url) == "2022790791047823773"
    assert is_twitter_url("https://example.com/page") is False


def test_canonicalize_and_normalize_x_urls() -> None:
    src = "https://twitter.com/user/status/2022790791047823773?s=20"
    assert (
        canonicalize_twitter_status_url(src)
        == "https://x.com/i/status/2022790791047823773"
    )
    assert normalize_x_url("https://mobile.twitter.com/user/status/1?s=20#frag") == (
        "https://x.com/user/status/1"
    )


def test_collect_x_candidate_urls_for_source_types() -> None:
    url_item = SimpleNamespace(source_type="url", payload="https://x.com/u/status/1")
    assert collect_x_candidate_urls(url_item) == ["https://x.com/u/status/1"]

    embed_item = SimpleNamespace(
        source_type="embed",
        payload=SimpleNamespace(
            url="https://x.com/u/status/2",
            video=SimpleNamespace(url="https://video.twimg.com/ext_tw_video/abc"),
            image=SimpleNamespace(url="https://pbs.twimg.com/media/xyz.jpg"),
            thumbnail=None,
        ),
    )
    embed_urls = collect_x_candidate_urls(embed_item)
    assert "https://x.com/u/status/2" in embed_urls
    assert "https://video.twimg.com/ext_tw_video/abc" in embed_urls
    assert "https://pbs.twimg.com/media/xyz.jpg" in embed_urls

    att_item = SimpleNamespace(
        source_type="attachment",
        payload=SimpleNamespace(
            url="https://video.twimg.com/ext_tw_video/att.mp4",
            proxy_url="https://cdn.discordapp.com/proxy",
        ),
    )
    att_urls = collect_x_candidate_urls(att_item)
    assert "https://video.twimg.com/ext_tw_video/att.mp4" in att_urls
    assert "https://cdn.discordapp.com/proxy" in att_urls


def test_twitter_host_and_media_path_helpers() -> None:
    assert is_twitter_thumbnail_url("https://pbs.twimg.com/media/a.jpg") is True
    assert is_twitter_media_cdn("https://video.twimg.com/ext_tw_video/abc") is True
    assert is_tweet_media_url("https://pbs.twimg.com/media/abc123.jpg") is True
    assert (
        is_tweet_media_url("https://pbs.twimg.com/profile_images/123/avatar.jpg")
        is False
    )
    assert (
        is_tweet_media_url("https://pbs.twimg.com/ext_tw_video_thumb/123/pu/img.jpg")
        is False
    )


def test_extract_x_status_urls_from_text() -> None:
    text = (
        "a https://x.com/u/status/1 and b "
        "https://twitter.com/u/status/2?s=20 and duplicate https://x.com/u/status/1"
    )
    urls = extract_x_status_urls_from_text(
        text,
        is_status_url=lambda u: "/status/" in u,
        canonicalize_status_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    assert urls == ["https://x.com/u/status/1", "https://x.com/u/status/2"]


def test_extract_raw_urls_and_filter_canonical_x_urls() -> None:
    raw = extract_raw_urls_from_texts(
        [
            "one https://x.com/a/status/1",
            "two https://example.com/z and https://twitter.com/a/status/1?s=20",
        ]
    )
    assert "https://x.com/a/status/1" in raw
    assert "https://example.com/z" in raw

    filtered = filter_canonical_x_urls(
        raw,
        is_x_url=lambda u: ("x.com/" in u or "twitter.com/" in u),
        canonicalize_x_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    assert filtered == ["https://x.com/a/status/1"]


def test_unwrap_x_media_url() -> None:
    wrapped = (
        "https://api.fxtwitter.com/dl?url="
        "https%3A%2F%2Fvideo.twimg.com%2Fext_tw_video%2Fabc%2Fvid%2F720x1280%2Fv.mp4"
    )
    assert (
        unwrap_x_media_url(wrapped)
        == "https://video.twimg.com/ext_tw_video/abc/vid/720x1280/v.mp4"
    )
    assert unwrap_x_media_url("https://x.com/user/status/1") == "https://x.com/user/status/1"


def test_extract_x_api_primary_tweet_variants() -> None:
    assert extract_x_api_primary_tweet({"data": {"id": "1"}}) == {"id": "1"}
    assert extract_x_api_primary_tweet({"data": [{"id": "2"}]}) == {"id": "2"}
    assert extract_x_api_primary_tweet({"data": []}) == {}
    assert extract_x_api_primary_tweet({"data": ["bad"]}) == {}
    assert extract_x_api_primary_tweet(None) == {}


def test_extract_x_api_primary_text_variants() -> None:
    assert extract_x_api_primary_text({"data": {"text": "dict text"}}) == "dict text"
    assert extract_x_api_primary_text({"data": [{"text": "list text"}]}) == "list text"
    assert extract_x_api_primary_text({"data": []}) == ""
    assert extract_x_api_primary_text(None) == ""


def test_extract_sparse_media_resolution_defaults_and_sanitizes() -> None:
    assert extract_sparse_media_resolution(
        None,
        default_url="https://x.com/a",
    ) == ("unknown", [], "https://x.com/a")

    assert extract_sparse_media_resolution(
        {"kind": "video", "images": "bad", "url": ""},
        default_url="https://x.com/b",
    ) == ("video", [], "https://x.com/b")

    assert extract_sparse_media_resolution(
        {"kind": "", "images": ["i1"], "url": "https://x.com/c"},
        default_url="https://x.com/d",
    ) == ("unknown", ["i1"], "https://x.com/c")


def test_extract_fxtwitter_tweet_node_variants() -> None:
    assert extract_fxtwitter_tweet_node({"tweet": {"id": "1"}}) == {"id": "1"}
    assert extract_fxtwitter_tweet_node({"status": {"id": "2"}}) == {"id": "2"}
    assert extract_fxtwitter_tweet_node({"tweet": "bad"}) == {}
    assert extract_fxtwitter_tweet_node({"status": []}) == {}
    assert extract_fxtwitter_tweet_node(None) == {}


def test_stt_result_has_transcription_matches_router_semantics() -> None:
    assert stt_result_has_transcription({"transcription": "hello"}) is True
    # Preserve bool() semantics used in router call sites.
    assert stt_result_has_transcription({"transcription": "   "}) is True
    assert stt_result_has_transcription({"transcription": ""}) is False
    assert stt_result_has_transcription({"text": "fallback only"}) is False
    assert stt_result_has_transcription(None) is False


def test_resolve_twitter_status_id_prefers_hint_then_parser() -> None:
    assert (
        resolve_twitter_status_id(
            "https://x.com/user/status/111",
            tweet_id="123",
            parse_status_id=lambda _url: (_ for _ in ()).throw(
                AssertionError("parser should not run when hint is provided")
            ),
        )
        == "123"
    )

    assert (
        resolve_twitter_status_id(
            "https://x.com/user/status/111",
            parse_status_id=lambda _url: "456",
        )
        == "456"
    )

    assert (
        resolve_twitter_status_id(
            "https://x.com/user/status/111",
            parse_status_id=lambda _url: None,
        )
        == ""
    )


def test_is_twitter_status_url_uses_parser() -> None:
    assert (
        is_twitter_status_url(
            "https://x.com/user/status/111",
            parse_status_id=lambda _url: "111",
        )
        is True
    )
    assert (
        is_twitter_status_url(
            "https://x.com/user/status/111",
            parse_status_id=lambda _url: None,
        )
        is False
    )


def test_classify_stt_error_reason_matches_router_semantics() -> None:
    assert classify_stt_error_reason("error") == "error"
    assert classify_stt_error_reason(None) == "no_speech"
    assert classify_stt_error_reason("timeout") == "no_speech"
    # Preserve legacy exact-match behavior (case-sensitive).
    assert classify_stt_error_reason("ERROR") == "no_speech"


def test_x_syn_probe_budget_timeout_s_caps_and_offsets() -> None:
    assert x_syn_probe_budget_timeout_s(9.0) == 4.5
    assert x_syn_probe_budget_timeout_s(2.2) == 3.2


def test_x_syn_quick_request_timeouts_caps_and_offsets() -> None:
    assert x_syn_quick_request_timeouts(9.0) == (3.0, 3.0, 3.5)
    assert x_syn_quick_request_timeouts(1.2) == (1.2, 1.2, 1.7)


def test_build_syndication_photo_payload_shape() -> None:
    payload = build_syndication_photo_payload("caption text", ["u1", "u2"])
    assert payload == {
        "text": "caption text",
        "photos": [{"url": "u1"}, {"url": "u2"}],
    }

    payload_none = build_syndication_photo_payload(None, [])
    assert payload_none == {"text": None, "photos": []}


def test_format_twitter_syndication_images_log_line_with_and_without_msg_id() -> None:
    assert (
        format_twitter_syndication_images_log_line(
            ["https://pbs.twimg.com/media/abc.jpg"],
            msg_id=123,
        )
        == "route.twitter.syndication | images=1 | pbs.twimg.com | msg_id=123"
    )
    assert (
        format_twitter_syndication_images_log_line(["not a url"])
        == "route.twitter.syndication | images=1 | n/a"
    )


@pytest.mark.asyncio
async def test_resolve_and_probe_twitter_images_delegates_and_normalizes() -> None:
    calls = {}

    def _resolve_status(url, tweet_id=None):
        calls["resolve"] = (url, tweet_id)
        return "123"

    async def _probe_images(url, status_id):
        calls["probe"] = (url, status_id)
        return ["u1", "u2"]

    status_id, image_urls = await resolve_and_probe_twitter_images(
        url="https://x.com/u/status/1",
        tweet_id="hint",
        resolve_status_id=_resolve_status,
        probe_images=_probe_images,
    )

    assert status_id == "123"
    assert image_urls == ["u1", "u2"]
    assert calls["resolve"] == ("https://x.com/u/status/1", "hint")
    assert calls["probe"] == ("https://x.com/u/status/1", "123")


@pytest.mark.asyncio
async def test_resolve_and_probe_twitter_images_normalizes_empty_probe() -> None:
    status_id, image_urls = await resolve_and_probe_twitter_images(
        url="https://x.com/u/status/1",
        resolve_status_id=lambda _url, tweet_id=None: "123",
        probe_images=lambda _url, _status_id: _async_none(),
    )
    assert status_id == "123"
    assert image_urls == []


async def _async_none():
    return None
