"""Spotify non-track links: bot-built cards from open.spotify.com Open Graph tags."""

from __future__ import annotations

import asyncio
from typing import cast

import aiohttp
import discord

from bot.embed_fix import spotify
from bot.embed_fix.instagram import parse_og, read_capped

PLAYLIST = "https://open.spotify.com/playlist/6wuyJaMQZPViOk27wS9GIM"
OG_HTML = (
    '<html><head><meta property="og:title" content="weekly"/>'
    '<meta property="og:description" content="Playlist · mir · 30 items"/>'
    '<meta property="og:image" content="https://image-cdn-ak.spotifycdn.com/image/abc"/>'
    "</head></html>"
)


def test_build_embed_uses_og_title_description_and_image() -> None:
    embed = spotify.build_embed(parse_og(OG_HTML), PLAYLIST)
    assert isinstance(embed, discord.Embed)
    assert embed.title == "weekly"
    assert embed.description == "Playlist · mir · 30 items"
    assert embed.url == PLAYLIST
    assert embed.image.url == "https://image-cdn-ak.spotifycdn.com/image/abc"
    assert embed.color is not None and embed.color.value == spotify.SPOTIFY_COLOR


def test_build_embed_without_title_is_dropped() -> None:
    assert spotify.build_embed({"og:description": "x"}, PLAYLIST) is None


def test_build_embed_ignores_non_https_image() -> None:
    embed = spotify.build_embed({"og:title": "t", "og:image": "http://insecure/x.jpg"}, PLAYLIST)
    assert embed is not None
    assert embed.image.url is None


def test_build_cards_empty_input_makes_no_request() -> None:
    assert asyncio.run(spotify.build_cards([])) == []


class _FakeStream:
    """Mimics aiohttp StreamReader.read(n): returns at most n bytes, possibly fewer."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, n: int) -> bytes:
        if not self._chunks:
            return b""
        head, rest = self._chunks[0][:n], self._chunks[0][n:]
        if rest:
            self._chunks[0] = rest
        else:
            self._chunks.pop(0)
        return head


class _FakeResp:
    def __init__(self, chunks: list[bytes]) -> None:
        self.content = _FakeStream(chunks)


def test_read_capped_loops_past_short_chunks() -> None:
    # StreamReader.read(n) can return a short first chunk; the og:image tag lives in the tail.
    resp = cast(aiohttp.ClientResponse, _FakeResp([b"<meta a>", b"<meta og:image>"]))
    assert asyncio.run(read_capped(resp, 1024)) == b"<meta a><meta og:image>"


def test_read_capped_stops_at_limit() -> None:
    resp = cast(aiohttp.ClientResponse, _FakeResp([b"abcdef", b"ghijkl"]))
    assert asyncio.run(read_capped(resp, 8)) == b"abcdefgh"
