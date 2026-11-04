"""Rumble links: bot-built cards from the oEmbed API."""

from __future__ import annotations

import asyncio

from bot.embed_fix import rumble
from bot.embed_fix.sites import find_fixes

URL = "https://rumble.com/v7htuj8-2026-midterm-election.html?e9s=src_v1_sa%2Csrc_v1_ucp_a"
CANON = "https://rumble.com/v7htuj8-2026-midterm-election.html"


def test_rumble_link_is_a_canonical_card_fix() -> None:
    fix = find_fixes(URL)[0]
    assert (fix.site, fix.rewritten, fix.card) == ("rumble", CANON, True)
    assert find_fixes("https://www.rumble.com/v7htuj8-x.html")[0].rewritten == "https://rumble.com/v7htuj8-x.html"


def test_rumble_non_video_pages_are_ignored() -> None:
    assert find_fixes("https://rumble.com/c/nickjfuentes") == []
    assert find_fixes("https://notrumble.com/v7htuj8-x.html") == []


def test_build_embed_from_oembed_record() -> None:
    embed = rumble.build_embed(
        {"title": "T", "author_name": "A", "author_url": "https://rumble.com/c/a", "thumbnail_url": "https://cdn/x.jpg"},
        CANON,
    )
    assert embed is not None
    assert (embed.title, embed.url, embed.author.name, embed.image.url) == ("T", CANON, "A", "https://cdn/x.jpg")


def test_build_embed_without_title_or_with_bad_thumb() -> None:
    assert rumble.build_embed({"author_name": "A"}, CANON) is None
    embed = rumble.build_embed({"title": "T", "thumbnail_url": "http://insecure/x.jpg"}, CANON)
    assert embed is not None and embed.image.url is None


def test_build_cards_empty_makes_no_request() -> None:
    assert asyncio.run(rumble.build_cards([])) == []
