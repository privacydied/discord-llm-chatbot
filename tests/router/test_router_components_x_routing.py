from types import SimpleNamespace

from bot.router_components.x_routing import (
    canonicalize_twitter_status_url,
    collect_x_candidate_urls,
    extract_raw_urls_from_texts,
    extract_x_status_urls_from_text,
    filter_canonical_x_urls,
    is_tweet_media_url,
    is_twitter_media_cdn,
    is_twitter_thumbnail_url,
    is_twitter_url,
    normalize_x_url,
    parse_twitter_status_id,
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
