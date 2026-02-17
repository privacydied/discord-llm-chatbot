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
    build_stt_fail_log_payload,
    build_caption_only_fallback_log_payload,
    build_x_video_stt_error_result_payload,
    resolve_caption_only_base_text,
    resolve_video_stt_error_base_text,
    syndication_article_has_blocks,
    extract_x_article_text,
    syndication_needs_article_hydration,
    extract_syndication_base_text,
    merge_syndication_base_with_article,
    extract_syndication_text,
    build_x_text_miss_log_payload,
    build_x_text_miss_payload,
    build_x_text_resolve_payload,
    build_syndication_non_200_log_payload,
    build_syndication_non_200_metric_payload,
    build_syndication_fetch_failed_payload,
    build_x_text_canon_payload,
    build_oembed_text_payload,
    extract_oembed_payload_from_response,
    build_syndication_oembed_url,
    build_syndication_base_url,
    build_syndication_fetch_user_agent,
    build_syndication_fetch_accept_language,
    build_syndication_fetch_referer,
    build_syndication_fetch_accept,
    build_syndication_lang,
    build_syndication_dnt_value,
    build_syndication_omit_script_value,
    build_syndication_hide_thread_value,
    build_syndication_fetch_headers,
    build_syndication_fetch_params,
    build_syndication_fetch_params_variants,
    build_syndication_widgets_endpoint,
    build_syndication_tweet_result_endpoint,
    build_syndication_widgets_tweet_path,
    build_syndication_tweet_result_path,
    build_syndication_oembed_params,
    build_syndication_oembed_status_url,
    build_syndication_oembed_options,
    build_syndication_oembed_hosts,
    build_syndication_oembed_metric_endpoint,
    build_syndication_oembed_fallback_params,
    build_syndication_oembed_fallback_plan,
    build_syndication_fetch_plan,
    build_syndication_fetch_metric_payload,
    syndication_cache_ttl_s,
    syndication_cache_is_fresh,
    classify_syndication_cache_hit,
    build_syndication_negative_cache_entry,
    build_syndication_cache_entry,
    build_syndication_endpoint_url,
    build_syndication_negative_cache_ttl_cap_s,
    extract_oembed_html_text,
    syndication_has_usable_payload,
    syndication_media_hint_keys,
    format_syndication_body_text,
    format_syndication_header_line,
    format_syndication_error_fallback,
    extract_syndication_photo_urls,
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


def test_build_stt_fail_log_payload_includes_optional_fields() -> None:
    assert build_stt_fail_log_payload("no_speech") == {
        "event": "stt.fail",
        "detail": {"reason": "no_speech"},
    }

    assert build_stt_fail_log_payload(
        "error",
        media_kind="video",
        msg_id=123,
    ) == {
        "event": "stt.fail",
        "detail": {"reason": "error", "media_kind": "video"},
        "msg_id": 123,
    }


def test_build_caption_only_fallback_log_payload_shape() -> None:
    assert build_caption_only_fallback_log_payload() == {
        "event": "fallback",
        "detail": {"kind": "caption_only"},
    }


def test_build_x_video_stt_error_result_payload_defaults_and_shape() -> None:
    assert build_x_video_stt_error_result_payload(
        url="https://x.com/u/status/1",
        stt_error=None,
    ) == {
        "transcription": None,
        "error": "transcription_failed",
        "media_kind": "video",
        "url": "https://x.com/u/status/1",
    }

    assert build_x_video_stt_error_result_payload(
        url="https://x.com/u/status/2",
        stt_error="network_error",
    ) == {
        "transcription": None,
        "error": "network_error",
        "media_kind": "video",
        "url": "https://x.com/u/status/2",
    }


def test_resolve_caption_only_base_text_preserves_router_precedence() -> None:
    assert (
        resolve_caption_only_base_text(
            api_text="api text",
            tweet_text="tweet text",
            base_text="base text",
        )
        == "api text"
    )
    assert (
        resolve_caption_only_base_text(
            api_text="",
            tweet_text="tweet text",
            base_text="base text",
        )
        == "tweet text"
    )
    # Preserve legacy behavior: truthy whitespace API text still wins before strip().
    assert (
        resolve_caption_only_base_text(
            api_text="   ",
            tweet_text="tweet text",
            base_text="base text",
        )
        == ""
    )


def test_resolve_video_stt_error_base_text_preserves_router_precedence() -> None:
    assert (
        resolve_video_stt_error_base_text(
            tweet_text="tweet text",
            base_text="base text",
        )
        == "tweet text"
    )
    assert (
        resolve_video_stt_error_base_text(
            tweet_text="",
            base_text="base text",
        )
        == "base text"
    )
    # Preserve legacy behavior: truthy whitespace tweet text still wins before strip().
    assert (
        resolve_video_stt_error_base_text(
            tweet_text="   ",
            base_text="base text",
        )
        == ""
    )


def test_syndication_article_has_blocks_variants() -> None:
    assert syndication_article_has_blocks(None) is False
    assert syndication_article_has_blocks({}) is False
    assert syndication_article_has_blocks({"content": {"blocks": "bad"}}) is False
    assert (
        syndication_article_has_blocks({"content": {"blocks": [{"text": "   "}]}}) is False
    )
    assert (
        syndication_article_has_blocks({"content": {"blocks": [{"text": "hello"}]}})
        is True
    )
    assert (
        syndication_article_has_blocks({"content": {"blocks": [{"x": 1}, "bad"]}})
        is False
    )


def test_extract_x_article_text_dedupes_unescapes_and_caps() -> None:
    article = {
        "title": "Title &amp; Co",
        "preview_text": "Preview",
        "content": {
            "blocks": [
                {"text": "Body A"},
                {"text": "Body A"},
                {"text": "Body &amp; B"},
                {"x": 1},
                "bad",
            ]
        },
    }
    out = extract_x_article_text(article)
    assert out == "Title & Co\n\nPreview\n\nBody A\n\nBody & B"


def test_extract_x_article_text_truncates_at_12000_chars() -> None:
    huge = "a" * 12050
    out = extract_x_article_text({"title": huge})
    assert len(out) == 12000
    assert out.endswith("…")


def test_syndication_needs_article_hydration_variants() -> None:
    assert syndication_needs_article_hydration({}) is False
    assert syndication_needs_article_hydration({"news_action_type": "article"}) is True
    assert (
        syndication_needs_article_hydration(
            {"text": "https://t.co/abc123"},
            allow_tco_pointer=True,
        )
        is True
    )
    assert (
        syndication_needs_article_hydration(
            {
                "article": {"id": "1"},
            },
            article_has_blocks=lambda _article: False,
        )
        is True
    )
    assert (
        syndication_needs_article_hydration(
            {
                "article": {"id": "1"},
            },
            article_has_blocks=lambda _article: True,
        )
        is False
    )


def test_extract_syndication_base_text_precedence() -> None:
    assert (
        extract_syndication_base_text(
            {
                "note_tweet": {"text": "note"},
                "legacy": {"full_text": "legacy"},
                "full_text": "full",
                "text": "text",
            }
        )
        == "note"
    )
    assert (
        extract_syndication_base_text(
            {
                "legacy": {"full_text": "legacy"},
                "full_text": "full",
                "text": "text",
            }
        )
        == "legacy"
    )
    assert extract_syndication_base_text({"text": " text "}) == "text"
    assert extract_syndication_base_text(None) == ""


def test_merge_syndication_base_with_article_variants() -> None:
    assert (
        merge_syndication_base_with_article(
            base_text="base",
            article_text="",
        )
        == "base"
    )
    assert (
        merge_syndication_base_with_article(
            base_text="base",
            article_text="article",
        )
        == "base\n\n[Linked X Article]\narticle"
    )
    assert (
        merge_syndication_base_with_article(
            base_text="base article body",
            article_text="article body",
        )
        == "base article body"
    )
    assert (
        merge_syndication_base_with_article(
            base_text="https://t.co/abc123",
            article_text="article",
        )
        == "article"
    )
    assert (
        merge_syndication_base_with_article(
            base_text="",
            article_text="article",
        )
        == "article"
    )


def test_extract_syndication_text_variants() -> None:
    assert extract_syndication_text(None) == ""
    assert extract_syndication_text({"text": " hello "}) == "hello"

    merged = extract_syndication_text(
        {"text": "base", "article": {"id": "1"}},
        extract_article_text=lambda _article: "article",
    )
    assert merged == "base\n\n[Linked X Article]\narticle"

    # Preserve fail-open behavior when article extractor errors.
    assert (
        extract_syndication_text(
            {"text": "base", "article": {"id": "1"}},
            extract_article_text=lambda _article: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
        )
        == "base"
    )


def test_build_x_text_miss_log_payload_shape_and_primary_id() -> None:
    assert build_x_text_miss_log_payload(
        "https://x.com/u/status/2022790791047823773"
    ) == {
        "event": "x.text.miss",
        "detail": {
            "primary": "2022790791047823773",
            "layer": "format",
            "reason": "empty_text",
        },
    }
    assert build_x_text_miss_log_payload("https://example.com/nope") == {
        "event": "x.text.miss",
        "detail": {
            "primary": "",
            "layer": "format",
            "reason": "empty_text",
        },
    }


def test_build_x_text_miss_payload_accepts_explicit_fields() -> None:
    assert build_x_text_miss_payload(
        primary="2022790791047823773",
        layer="syndication",
        reason="no_text",
    ) == {
        "event": "x.text.miss",
        "detail": {
            "primary": "2022790791047823773",
            "layer": "syndication",
            "reason": "no_text",
        },
    }


def test_build_x_text_resolve_payload_shape() -> None:
    assert build_x_text_resolve_payload(
        primary="2022790791047823773",
        source="syndication",
        chars=42,
    ) == {
        "event": "x.text.resolve",
        "detail": {
            "primary": "2022790791047823773",
            "source": "syndication",
            "chars": 42,
        },
    }


def test_build_syndication_non_200_log_payload_shape() -> None:
    assert build_syndication_non_200_log_payload(
        tweet_id="2022790791047823773",
        status=403,
        endpoint="widgets",
    ) == {
        "detail": {
            "tweet_id": "2022790791047823773",
            "status": 403,
            "endpoint": "widgets",
        }
    }


def test_build_syndication_non_200_metric_payload_shape() -> None:
    assert build_syndication_non_200_metric_payload(
        status=403,
        endpoint="widgets",
    ) == {
        "status": "403",
        "endpoint": "widgets",
    }


def test_build_syndication_fetch_failed_payload_shape() -> None:
    assert build_syndication_fetch_failed_payload(
        tweet_id="2022790791047823773",
        error="timeout",
    ) == {
        "detail": {
            "tweet_id": "2022790791047823773",
            "error": "timeout",
        }
    }


def test_build_x_text_canon_payload_shape() -> None:
    assert build_x_text_canon_payload(
        url="https://x.com/u/status/2022790791047823773?ptid=2022790791047823773",
        primary="2022790791047823773",
    ) == {
        "event": "x.text.canon",
        "detail": {
            "url": "https://x.com/u/status/2022790791047823773?ptid=2022790791047823773",
            "primary": "2022790791047823773",
        },
    }


def test_extract_oembed_html_text_strips_tags_and_unescapes() -> None:
    assert (
        extract_oembed_html_text("<p>Hello <strong>world</strong> &amp; co</p>")
        == "Hello world & co"
    )


def test_extract_oembed_html_text_empty_input_returns_empty_string() -> None:
    assert extract_oembed_html_text("") == ""
    assert extract_oembed_html_text(None) == ""


def test_extract_oembed_html_text_non_string_truthy_input_raises() -> None:
    with pytest.raises(TypeError):
        extract_oembed_html_text({"html": "<p>bad</p>"})


def test_build_oembed_text_payload_happy_path_shape() -> None:
    assert build_oembed_text_payload(
        {"html": "<p>Hello &amp; co</p>", "author_name": "alice"}
    ) == {
        "text": "Hello & co",
        "user": {"name": "alice"},
    }


def test_build_oembed_text_payload_returns_none_for_non_usable_obj() -> None:
    assert build_oembed_text_payload(None) is None
    assert build_oembed_text_payload({"html": ""}) is None


def test_build_oembed_text_payload_non_string_html_raises() -> None:
    with pytest.raises(TypeError):
        build_oembed_text_payload({"html": {"bad": "value"}})


def test_build_syndication_oembed_params_default_and_x_host() -> None:
    assert build_syndication_oembed_params("2022790791047823773") == {
        "url": "https://twitter.com/i/status/2022790791047823773",
        "dnt": "false",
        "omit_script": "true",
        "hide_thread": "true",
        "lang": "en",
    }
    assert build_syndication_oembed_params(
        "2022790791047823773", use_x_host=True
    ) == {
        "url": "https://x.com/i/status/2022790791047823773",
        "dnt": "false",
        "omit_script": "true",
        "hide_thread": "true",
        "lang": "en",
    }


def test_build_syndication_oembed_hosts_order() -> None:
    assert build_syndication_oembed_hosts() == ("twitter.com", "x.com")


def test_build_syndication_oembed_status_url_shape() -> None:
    assert (
        build_syndication_oembed_status_url("twitter.com", "2022790791047823773")
        == "https://twitter.com/i/status/2022790791047823773"
    )


def test_build_syndication_oembed_options_constant() -> None:
    assert build_syndication_oembed_options() == {
        "dnt": "false",
        "omit_script": "true",
        "hide_thread": "true",
    }


def test_build_syndication_oembed_metric_endpoint_mapping() -> None:
    assert build_syndication_oembed_metric_endpoint("twitter.com") == "oembed"
    assert build_syndication_oembed_metric_endpoint("x.com") == "oembed_x"
    assert build_syndication_oembed_metric_endpoint("unknown") == "oembed"


def test_build_syndication_oembed_fallback_params_ordered_variants() -> None:
    assert build_syndication_oembed_fallback_params("2022790791047823773") == [
        (
            "oembed",
            {
                "url": "https://twitter.com/i/status/2022790791047823773",
                "dnt": "false",
                "omit_script": "true",
                "hide_thread": "true",
                "lang": "en",
            },
        ),
        (
            "oembed_x",
            {
                "url": "https://x.com/i/status/2022790791047823773",
                "dnt": "false",
                "omit_script": "true",
                "hide_thread": "true",
                "lang": "en",
            },
        ),
    ]


def test_build_syndication_oembed_fallback_plan_shape() -> None:
    url, variants = build_syndication_oembed_fallback_plan("2022790791047823773")
    assert url == "https://publish.twitter.com/oembed"
    assert variants == build_syndication_oembed_fallback_params("2022790791047823773")


def test_build_syndication_oembed_url_constant() -> None:
    assert build_syndication_oembed_url() == "https://publish.twitter.com/oembed"


def test_build_syndication_base_url_constant() -> None:
    assert build_syndication_base_url() == "https://cdn.syndication.twimg.com/"


def test_build_syndication_fetch_headers_shape() -> None:
    headers = build_syndication_fetch_headers()
    assert headers["Referer"] == "https://platform.twitter.com/"
    assert headers["Accept-Language"] == "en-US,en;q=0.9"
    assert "Mozilla/5.0" in headers["User-Agent"]


def test_build_syndication_fetch_user_agent_shape() -> None:
    user_agent = build_syndication_fetch_user_agent()
    assert "Mozilla/5.0" in user_agent
    assert "Chrome/126.0.0.0" in user_agent


def test_build_syndication_fetch_accept_language_constant() -> None:
    assert build_syndication_fetch_accept_language() == "en-US,en;q=0.9"


def test_build_syndication_fetch_referer_constant() -> None:
    assert build_syndication_fetch_referer() == "https://platform.twitter.com/"


def test_build_syndication_fetch_accept_constant() -> None:
    assert build_syndication_fetch_accept() == (
        "application/json, text/javascript;q=0.9, */*;q=0.8"
    )


def test_build_syndication_lang_constant() -> None:
    assert build_syndication_lang() == "en"


def test_build_syndication_dnt_value_constant() -> None:
    assert build_syndication_dnt_value() == "false"


def test_build_syndication_omit_script_value_constant() -> None:
    assert build_syndication_omit_script_value() == "true"


def test_build_syndication_hide_thread_value_constant() -> None:
    assert build_syndication_hide_thread_value() == "true"


def test_build_syndication_fetch_params_variants_with_and_without_dnt() -> None:
    assert build_syndication_fetch_params("2022790791047823773") == {
        "id": "2022790791047823773",
        "lang": "en",
    }
    assert build_syndication_fetch_params(
        "2022790791047823773",
        include_dnt=True,
    ) == {
        "id": "2022790791047823773",
        "lang": "en",
        "dnt": "false",
    }


def test_build_syndication_fetch_params_variants_shape() -> None:
    variants = build_syndication_fetch_params_variants("2022790791047823773")
    assert variants == [
        ("widgets", {"id": "2022790791047823773", "lang": "en"}),
        ("tweet-result", {"id": "2022790791047823773", "lang": "en"}),
        ("widgets", {"id": "2022790791047823773", "lang": "en", "dnt": "false"}),
    ]


def test_build_syndication_endpoint_name_constants() -> None:
    assert build_syndication_widgets_endpoint() == "widgets"
    assert build_syndication_tweet_result_endpoint() == "tweet-result"


def test_build_syndication_endpoint_path_constants() -> None:
    assert build_syndication_widgets_tweet_path() == "widgets/tweet"
    assert build_syndication_tweet_result_path() == "tweet-result"


def test_extract_oembed_payload_from_response_variants() -> None:
    good = SimpleNamespace(
        status_code=200,
        json=lambda: {"html": "<p>Hello &amp; world</p>", "author_name": "alice"},
    )
    assert extract_oembed_payload_from_response(good) == {
        "text": "Hello & world",
        "user": {"name": "alice"},
    }

    non_200 = SimpleNamespace(status_code=404, json=lambda: {"html": "<p>x</p>"})
    assert extract_oembed_payload_from_response(non_200) is None

    bad_json = SimpleNamespace(
        status_code=200,
        json=lambda: (_ for _ in ()).throw(ValueError("boom")),
    )
    assert extract_oembed_payload_from_response(bad_json) is None


def test_extract_oembed_payload_from_response_missing_status_raises() -> None:
    with pytest.raises(AttributeError):
        extract_oembed_payload_from_response(object())


def test_build_syndication_fetch_plan_shape_and_values() -> None:
    base, headers, variants = build_syndication_fetch_plan("2022790791047823773")
    assert base == "https://cdn.syndication.twimg.com/"
    assert headers["Referer"] == "https://platform.twitter.com/"
    assert headers["Accept-Language"] == "en-US,en;q=0.9"
    assert variants == [
        ("widgets", {"id": "2022790791047823773", "lang": "en"}),
        ("tweet-result", {"id": "2022790791047823773", "lang": "en"}),
        ("widgets", {"id": "2022790791047823773", "lang": "en", "dnt": "false"}),
    ]


def test_build_syndication_fetch_metric_payload_shape() -> None:
    assert build_syndication_fetch_metric_payload("widgets") == {"endpoint": "widgets"}
    assert build_syndication_fetch_metric_payload("oembed_x") == {
        "endpoint": "oembed_x"
    }


def test_syndication_cache_ttl_s_caps_negative_entries() -> None:
    assert syndication_cache_ttl_s(600.0, {"neg": True}) == 300.0
    assert syndication_cache_ttl_s(120.0, {"neg": True}) == 120.0
    assert syndication_cache_ttl_s(600.0, {"neg": False}) == 600.0


def test_build_syndication_negative_cache_ttl_cap_s_constant() -> None:
    assert build_syndication_negative_cache_ttl_cap_s() == 300.0


def test_syndication_cache_ttl_s_preserves_attribute_error_for_bad_cache() -> None:
    with pytest.raises(AttributeError):
        syndication_cache_ttl_s(600.0, object())


def test_syndication_cache_is_fresh_respects_ttl_policy() -> None:
    assert (
        syndication_cache_is_fresh(
            1_000.0,
            600.0,
            {"ts": 500.0, "neg": False},
        )
        is True
    )
    assert (
        syndication_cache_is_fresh(
            1_000.0,
            600.0,
            {"ts": 500.0, "neg": True},
        )
        is False
    )


def test_syndication_cache_is_fresh_preserves_ts_parse_error() -> None:
    with pytest.raises(ValueError):
        syndication_cache_is_fresh(1_000.0, 600.0, {"ts": "bad", "neg": False})


def test_classify_syndication_cache_hit_variants() -> None:
    assert (
        classify_syndication_cache_hit(
            1_000.0,
            600.0,
            {"ts": 500.0, "neg": False},
        )
        == "data"
    )
    assert (
        classify_syndication_cache_hit(
            1_000.0,
            600.0,
            {"ts": 850.0, "neg": True},
        )
        == "neg"
    )
    assert (
        classify_syndication_cache_hit(
            1_000.0,
            120.0,
            {"ts": 850.0, "neg": False},
        )
        is None
    )


def test_classify_syndication_cache_hit_preserves_ts_parse_error() -> None:
    with pytest.raises(ValueError):
        classify_syndication_cache_hit(1_000.0, 600.0, {"ts": "bad", "neg": False})


def test_build_syndication_negative_cache_entry_shape() -> None:
    assert build_syndication_negative_cache_entry(123.45) == {
        "neg": True,
        "ts": 123.45,
    }


def test_build_syndication_cache_entry_shape() -> None:
    data = {"text": "hello"}
    assert build_syndication_cache_entry(data, 321.0) == {
        "data": data,
        "ts": 321.0,
    }


def test_build_syndication_endpoint_url_mapping() -> None:
    base = "https://cdn.syndication.twimg.com/"
    assert build_syndication_endpoint_url(base, "widgets") == (
        "https://cdn.syndication.twimg.com/widgets/tweet"
    )
    assert build_syndication_endpoint_url(base, "tweet-result") == (
        "https://cdn.syndication.twimg.com/tweet-result"
    )
    # Preserve legacy fallback behavior for unknown endpoint values.
    assert build_syndication_endpoint_url(base, "unknown") == (
        "https://cdn.syndication.twimg.com/tweet-result"
    )


def test_syndication_has_usable_payload_with_text_or_media_hints() -> None:
    assert (
        syndication_has_usable_payload(
            {"text": "hello"},
            extract_text=lambda node: str(node.get("text") or "").strip(),
            media_hint_keys=("entities", "media"),
        )
        is True
    )
    assert (
        syndication_has_usable_payload(
            {"entities": {}},
            extract_text=lambda _node: "",
            media_hint_keys=("entities", "media"),
        )
        is True
    )
    assert (
        syndication_has_usable_payload(
            {"x": 1},
            extract_text=lambda _node: "",
            media_hint_keys=("entities", "media"),
        )
        is False
    )


def test_syndication_has_usable_payload_non_dict_returns_false() -> None:
    assert (
        syndication_has_usable_payload(
            None,
            extract_text=lambda _node: "text",
            media_hint_keys=("entities",),
        )
        is False
    )


def test_syndication_media_hint_keys_matches_router_contract() -> None:
    assert syndication_media_hint_keys() == (
        "media",
        "photos",
        "video",
        "video_info",
        "video_variants",
        "video_urls",
        "media_duration",
        "duration_ms",
        "extended_entities",
        "entities",
        "quoted_tweet",
        "quoted_status",
        "retweeted_status",
        "legacy",
        "card",
        "image",
        "article",
    )


def test_format_syndication_body_text_variants() -> None:
    assert format_syndication_body_text("short") == "short"
    assert format_syndication_body_text("") == (
        "(Tweet text not available. If you want analysis, paste the text or add a screenshot.)"
    )
    long_text = "a" * 5000
    out = format_syndication_body_text(long_text)
    assert len(out) == 3991
    assert out.endswith("…")


def test_format_syndication_header_line_variants() -> None:
    assert (
        format_syndication_header_line(
            user={"screen_name": "alice", "name": "Alice"},
            created_at="2026-02-17",
            photos=["p1", "p2"],
            url="https://x.com/u/status/1",
        )
        == "@alice • 2026-02-17 • media:2 → https://x.com/u/status/1"
    )
    assert (
        format_syndication_header_line(
            user={},
            created_at=None,
            photos=[],
            url="https://x.com/u/status/1",
        )
        == "Tweet → https://x.com/u/status/1"
    )


def test_format_syndication_header_line_preserves_len_error_behavior() -> None:
    with pytest.raises(TypeError):
        format_syndication_header_line(
            user={"name": "Alice"},
            created_at="2026-02-17",
            photos=3,
            url="https://x.com/u/status/1",
        )


def test_format_syndication_error_fallback_truncates_payload_repr() -> None:
    syn_data = {"blob": "a" * 5000}
    out = format_syndication_error_fallback("https://x.com/u/status/1", syn_data)
    assert out.startswith("Tweet → https://x.com/u/status/1\n")
    payload_part = out.split("\n", 1)[1]
    assert len(payload_part) == 4000


def test_extract_syndication_photo_urls_variants() -> None:
    photos = [
        {"url": "u1"},
        {"media_url_https": "u2"},
        {"media_url": "u3"},
        {"url": None},
        "u4",
    ]
    assert extract_syndication_photo_urls(photos) == ["u1", "u2", "u3", "u4"]

    # Preserve current semantics for string payloads (iterates chars).
    assert extract_syndication_photo_urls("ab") == ["a", "b"]

    with pytest.raises(TypeError):
        extract_syndication_photo_urls(1)


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
