from types import SimpleNamespace

from bot.router_components.input_harvest import (
    all_attachments_are_text,
    append_embed_related_urls,
    append_unique_url_items,
    existing_url_payloads,
    extract_urls_loose,
    extract_urls_strict,
    has_explicit_media_intent,
    has_meaningful_text,
    is_direct_image_url,
    is_text_attachment,
    strip_discord_mentions_and_urls,
    strip_urls,
)


def test_is_text_attachment_by_filename_or_content_type() -> None:
    assert is_text_attachment(SimpleNamespace(filename="note.txt", content_type=None))
    assert is_text_attachment(SimpleNamespace(filename="note.bin", content_type="text/plain"))
    assert not is_text_attachment(SimpleNamespace(filename="image.png", content_type="image/png"))


def test_all_attachments_are_text() -> None:
    text_atts = [
        SimpleNamespace(filename="a.txt", content_type=None),
        SimpleNamespace(filename="b.md", content_type="text/markdown"),
    ]
    mixed_atts = [
        SimpleNamespace(filename="a.txt", content_type=None),
        SimpleNamespace(filename="image.png", content_type="image/png"),
    ]
    assert all_attachments_are_text(text_atts) is True
    assert all_attachments_are_text(mixed_atts) is False
    assert all_attachments_are_text([]) is False


def test_has_meaningful_text_variants() -> None:
    assert has_meaningful_text("hello")
    assert has_meaningful_text("?")
    assert has_meaningful_text("yo")
    assert not has_meaningful_text("   ")


def test_has_explicit_media_intent() -> None:
    assert has_explicit_media_intent("please summarize this video")
    assert has_explicit_media_intent("analyze this image for me")
    assert not has_explicit_media_intent("just chatting")


def test_extract_urls_loose_and_strict_and_strip() -> None:
    text = "one https://x.com/a/status/1 and https://youtube.com/watch?v=abc."
    loose = extract_urls_loose(text)
    strict = extract_urls_strict(text)

    assert any("x.com/a/status/1" in u for u in loose)
    assert any("youtube.com/watch?v=abc" in u for u in strict)

    cleaned = strip_urls(text)
    assert "https://x.com" not in cleaned
    assert "https://youtube.com" not in cleaned


def test_existing_url_payloads_and_append_unique_url_items() -> None:
    items = [
        SimpleNamespace(source_type="url", payload="https://a.com/1", order_index=0),
        SimpleNamespace(source_type="attachment", payload="file", order_index=1),
    ]
    assert existing_url_payloads(items) == {"https://a.com/1"}

    def ctor(**kwargs):
        return SimpleNamespace(**kwargs)

    added = append_unique_url_items(
        items,
        ["https://a.com/1", "https://b.com/2"],
        item_ctor=ctor,
    )
    assert added == 1
    assert any(getattr(it, "source_type", None) == "url" and getattr(it, "payload", None) == "https://b.com/2" for it in items)

    # strip_key behavior mirrors call sites that dedupe on stripped URL.
    added_strip = append_unique_url_items(
        items,
        [" https://b.com/2 "],
        item_ctor=ctor,
        strip_key=True,
        existing_urls={"https://b.com/2"},
    )
    assert added_strip == 0


def test_append_embed_related_urls() -> None:
    found_urls = ["https://x.com/u/status/1"]
    embeds = [
        SimpleNamespace(
            url="https://x.com/u/status/1",
            video=SimpleNamespace(url="https://video.twimg.com/ext_tw_video/1"),
            author=SimpleNamespace(url="https://x.com/u"),
        ),
        SimpleNamespace(
            url="https://x.com/u/status/2",
            video=None,
            author=None,
        ),
    ]

    append_embed_related_urls(found_urls, embeds)
    assert "https://video.twimg.com/ext_tw_video/1" in found_urls
    assert "https://x.com/u" in found_urls
    assert "https://x.com/u/status/2" in found_urls


def test_strip_discord_mentions_and_urls() -> None:
    text = "<@123> check this <#999> https://x.com/u/status/1 hello"
    cleaned = strip_discord_mentions_and_urls(text)
    assert "<@123>" not in cleaned
    assert "<#999>" not in cleaned
    assert "https://x.com" not in cleaned
    assert "hello" in cleaned


def test_is_direct_image_url() -> None:
    assert is_direct_image_url("https://cdn.example.com/image.jpg")
    assert is_direct_image_url("https://cdn.example.com/image.PNG?x=1")
    assert is_direct_image_url("https://cdn.example.com/image.webp#frag")
    assert not is_direct_image_url("https://cdn.example.com/image.mp4")
    assert not is_direct_image_url("not-a-url")
