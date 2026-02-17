from types import SimpleNamespace

from bot.router_components.input_harvest import (
    extract_urls_loose,
    extract_urls_strict,
    has_explicit_media_intent,
    has_meaningful_text,
    is_text_attachment,
    strip_urls,
)


def test_is_text_attachment_by_filename_or_content_type() -> None:
    assert is_text_attachment(SimpleNamespace(filename="note.txt", content_type=None))
    assert is_text_attachment(
        SimpleNamespace(filename="note.bin", content_type="text/plain")
    )
    assert not is_text_attachment(
        SimpleNamespace(filename="image.png", content_type="image/png")
    )


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
