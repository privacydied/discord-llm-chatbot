"""Spotify show URLs as RSS feeds (bot/rss/spotify_show.py)."""

from __future__ import annotations

import json

import pytest

from bot.rss.feeds import FeedParseError
from bot.rss.spotify_show import embed_url, is_spotify_show, parse_embed

SHOW = "https://open.spotify.com/show/1zTsFdPrKKOubsqnvDMQh0"


def _page(entity: dict) -> bytes:
    data = {"props": {"pageProps": {"state": {"data": {"entity": entity}}}}}
    return f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(data)}</script></html>'.encode()


def test_show_url_detection_and_embed_url() -> None:
    assert is_spotify_show(SHOW + "?si=abc")
    assert is_spotify_show("https://open.spotify.com/intl-fr/show/1zTsFdPrKKOubsqnvDMQh0")
    assert not is_spotify_show("https://open.spotify.com/episode/3X1EZvm9NVdlFeAaY3emy7")
    assert not is_spotify_show("https://feeds.soundcloud.com/x.rss")
    assert embed_url(SHOW + "?si=abc") == "https://open.spotify.com/embed/show/1zTsFdPrKKOubsqnvDMQh0"


def test_parse_embed_latest_episode() -> None:
    title, items = parse_embed(
        _page(
            {
                "type": "episode",
                "title": "[#379] MASTERY",
                "subtitle": "Subliminal Jihad [Al-Wara' Frequency]",
                "uri": "spotify:episode:3X1EZvm9NVdlFeAaY3emy7",
                "releaseDate": {"isoString": "2026-11-10T07:32:00Z"},
            }
        )
    )
    assert title == "Subliminal Jihad [Al-Wara' Frequency]"
    (item,) = items
    assert (item.title, item.pub_date, item.guid) == ("[#379] MASTERY", "2026-11-10T07:32:00Z", "spotify:episode:3X1EZvm9NVdlFeAaY3emy7")
    assert item.link == "https://open.spotify.com/episode/3X1EZvm9NVdlFeAaY3emy7"


@pytest.mark.parametrize("body", [b"<html></html>", _page({"type": "show", "uri": "spotify:show:x"}), b'<script id="__NEXT_DATA__">{oops</script>'])
def test_parse_embed_rejects_pages_without_an_episode(body: bytes) -> None:
    with pytest.raises(FeedParseError):
        parse_embed(body)
