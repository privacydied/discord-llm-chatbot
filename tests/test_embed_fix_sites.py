"""Site matching / rewriting for the SaucyBot port (bot/embed_fix/sites.py)."""

from __future__ import annotations

import pytest

from bot.embed_fix.sites import SITE_IDS, find_fixes, page_slug, is_ignored, parse_domain_overrides, parse_site_list, rewrite_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # Twitter / X / nitter (SaucyBot VxTwitter pattern)
        ("https://x.com/jack/status/20", "https://vxtwitter.com/jack/status/20"),
        ("https://twitter.com/jack/status/20?s=20&t=abc", "https://vxtwitter.com/jack/status/20"),
        ("http://www.twitter.com/jack/status/20", "https://vxtwitter.com/jack/status/20"),
        ("https://mobile.twitter.com/jack/status/20", "https://vxtwitter.com/jack/status/20"),
        ("https://nitter.net/jack/status/20", "https://vxtwitter.com/jack/status/20"),
        ("https://X.COM/Jack/status/20", "https://vxtwitter.com/Jack/status/20"),
        ("https://x.com/jack/status/20/photo/1", "https://vxtwitter.com/jack/status/20"),
        ("https://x.com/jack/status/20/video/1", "https://vxtwitter.com/jack/status/20"),
        ("https://x.com/jack/status/20/en", "https://vxtwitter.com/jack/status/20/en"),
        ("https://x.com/i/web/status/20", "https://vxtwitter.com/i/web/status/20"),
        ("https://twitter.com/jack/statuses/20", "https://vxtwitter.com/jack/status/20"),
        # Instagram (SaucyBot VxInstagram: p / reel / reels)
        ("https://www.instagram.com/p/ABC123/", "https://www.oginstagram.com/p/ABC123/"),
        ("https://instagram.com/reel/C0de_-x/?igsh=xyz", "https://www.oginstagram.com/reel/C0de_-x/"),
        ("https://m.instagram.com/reels/XYZ", "https://www.oginstagram.com/reels/XYZ/"),
        ("https://www.instagram.com/reel/DFeXBh7xtkI/?igsh=abc", "https://www.oginstagram.com/reel/DFeXBh7xtkI/"),
        # Username-prefixed share links (seen live; previously ignored)
        ("https://www.instagram.com/twinklemommyyyy/p/DRbeQJ_DVnR/", "https://www.oginstagram.com/p/DRbeQJ_DVnR/"),
        ("https://www.instagram.com/waffles.weekly/reel/CzW4fBzvTMO/", "https://www.oginstagram.com/reel/CzW4fBzvTMO/"),
        # Bluesky (VixBluesky host)
        ("https://bsky.app/profile/bsky.app/post/3l6oveex3ii2l", "https://bskx.app/profile/bsky.app/post/3l6oveex3ii2l"),
        # FurAffinity (XFurAffinity)
        ("https://www.furaffinity.net/view/123456/", "https://xfuraffinity.net/view/123456/"),
        ("https://furaffinity.net/full/1", "https://xfuraffinity.net/full/1/"),
        # Reddit media wrapper -> decoded direct URL (SaucyBot Reddit.cs)
        ("https://www.reddit.com/media?url=https%3A%2F%2Fi.redd.it%2Fabc.jpg", "https://i.redd.it/abc.jpg"),
        # Pixiv / DeviantArt via rewrite proxies
        ("https://www.pixiv.net/en/artworks/12345", "https://phixiv.net/artworks/12345"),
        ("https://pixiv.net/artworks/12345", "https://phixiv.net/artworks/12345"),
        ("https://www.deviantart.com/someone/art/Cool-Thing-123", "https://fixdeviantart.com/someone/art/Cool-Thing-123"),
        # Additions beyond SaucyBot
        ("https://www.tiktok.com/@user.name/video/7300000000000000000", "https://tnktok.com/@user.name/video/7300000000000000000"),
        ("https://vm.tiktok.com/ZMabc123/", "https://tnktok.com/t/ZMabc123/"),
        ("https://vt.tiktok.com/ZSxyz/", "https://tnktok.com/t/ZSxyz/"),
        ("https://www.tiktok.com/t/ZTfM6GsWt/", "https://tnktok.com/t/ZTfM6GsWt/"),  # share links (most common)
        ("https://old.reddit.com/r/videos/comments/abc12/title_here/", "https://vxreddit.com/r/videos/comments/abc12/title_here"),
    ],
)
def test_rewrites(url, expected):
    assert rewrite_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://google.com",
        "https://fox.com/jack/status/20",  # Copilot plan bug: substring 'x.com' in 'fox.com'
        "https://dropbox.com/jack/status/20",
        "https://vxtwitter.com/jack/status/20",  # already fixed
        "https://fxtwitter.com/jack/status/20",
        "https://fixupx.com/jack/status/20",
        "https://x.com/jack",  # profile, not a post
        "https://x.com/jack/likes",
        "https://www.instagram.com/someuser/",
        "https://www.instagram.com/someuser/tagged/",
        "https://www.kkinstagram.com/someuser/p/ABC/",
        "https://www.kkinstagram.com/p/ABC/",  # already fixed
        "https://kkinstagram.com/reel/ABC/",
        "https://www.oginstagram.com/reel/ABC/",
        "https://vxinstagram.com/p/ABC/",
        "https://xfuraffinity.net/view/1/",
        "https://www.reddit.com/media?url=javascript%3Aalert(1)",
        "https://www.reddit.com/r/videos/",
        "https://www.tiktok.com/@user",
        "https://tnktok.com/t/ZTfM6GsWt/",  # already fixed
        "https://www.kktiktok.com/t/ZTyc2cv8q/",
        "https://vxtiktok.com/@a/video/1",
        "https://notx.com/jack/status/20",
    ],
)
def test_non_matching_urls(url):
    assert rewrite_url(url) is None


def test_multiple_links_in_message_order_and_deduped():
    content = "look https://www.instagram.com/p/A1/ and https://x.com/a/status/1 and again https://twitter.com/a/status/1"
    fixes = find_fixes(content)
    assert [f.rewritten for f in fixes] == ["https://www.oginstagram.com/p/A1/", "https://vxtwitter.com/a/status/1"]
    assert [f.site for f in fixes] == ["instagram", "twitter"]


def test_limit_caps_results():
    content = " ".join(f"https://x.com/a/status/{i}" for i in range(20))
    assert len(find_fixes(content)) == 8
    assert len(find_fixes(content, limit=3)) == 3
    assert find_fixes(content, limit=0) == []


def test_disabled_sites_are_skipped():
    content = "https://x.com/a/status/1 https://www.instagram.com/p/A1/"
    fixes = find_fixes(content, disabled={"twitter"})
    assert [f.site for f in fixes] == ["instagram"]


def test_domain_overrides_apply_and_validate():
    overrides = parse_domain_overrides("twitter=fixupx.com, instagram=KKInstagram.com, bogus=x.y, pixiv=not a host")
    assert overrides == {"twitter": "fixupx.com", "instagram": "kkinstagram.com"}
    assert rewrite_url("https://x.com/a/status/1", overrides=overrides) == "https://fixupx.com/a/status/1"
    assert rewrite_url("https://instagram.com/p/A1", overrides=overrides) == "https://kkinstagram.com/p/A1/"


def test_nsfw_flags_follow_saucybot():
    nsfw = {f.site: f.nsfw for f in find_fixes("https://furaffinity.net/view/1 https://x.com/a/status/1 https://reddit.com/media?url=https%3A%2F%2Fi.redd.it%2Fa.png")}
    assert nsfw == {"furaffinity": True, "twitter": False, "reddit_media": True}


@pytest.mark.parametrize(
    ("content", "ignored"),
    [
        ("<https://x.com/a/status/1>", True),
        ("||https://x.com/a/status/1||", True),
        ("check this <https://x.com/a/status/1> out", True),
        ("https://x.com/a/status/1", False),
        ("<@123> https://x.com/a/status/1", False),
        ("<#456> https://x.com/a/status/1", False),
        ("<:emoji:789> https://x.com/a/status/1", False),
        ("<a:anim:789> https://x.com/a/status/1", False),
        ("", False),
    ],
)
def test_ignore_tags(content, ignored):
    assert is_ignored(content) is ignored


def test_parse_site_list():
    assert parse_site_list(" Twitter, reddit ,,") == frozenset({"twitter", "reddit"})
    assert parse_site_list(None) == frozenset()
    assert {"twitter", "instagram", "bluesky", "furaffinity", "reddit_media", "pixiv", "deviantart", "tiktok", "reddit", "ppv"} == SITE_IDS


PPV_SLUG = "misfits-boxing-joey-essex-vs-dapper-laughs-30174"


@pytest.mark.parametrize("host", ["ppv.st", "ppv.tj", "ppvs.pk", "ppv.rw", "ppv.ms", "ppv.bi", "ppv.ug", "www.ppvs.pk", "PPVS.PK"])
def test_ppv_links_become_cards_on_posted_host(host):
    fixes = find_fixes(f"watch https://{host}/live/{PPV_SLUG}?ref=x tonight")
    assert len(fixes) == 1
    fix = fixes[0]
    assert fix.site == "ppv" and fix.card is True and fix.nsfw is False
    assert fix.rewritten == f"https://{host.lower().removeprefix('www.')}/live/{PPV_SLUG}"
    assert page_slug(fix.rewritten) == PPV_SLUG


@pytest.mark.parametrize(
    "url",
    ["https://ppv.st/", "https://ppv.st/category/boxing", "https://notppv.st/live/x-1", "https://ppv.stream/live/x-1", "https://api.ppv.st/live/x-1"],
)
def test_ppv_non_matching(url):
    assert find_fixes(url) == []


@pytest.mark.parametrize(
    ("posted", "slug"),
    [
        ("https://ppvs.pk/live/cfb/2026-10-10/tex-ou", "cfb/2026-10-10/tex-ou"),
        ("https://ppvs.pk/live/cfb/2026-10-10/tex-ou/skycast", "cfb/2026-10-10/tex-ou/skycast"),
        ("https://ppvs.pk/live/cfb/2026-10-10/tex-ou/?x=1", "cfb/2026-10-10/tex-ou"),
    ],
)
def test_ppv_multi_segment_slugs(posted, slug):
    fixes = find_fixes(posted)
    assert [fix.rewritten for fix in fixes] == [f"https://ppvs.pk/live/{slug}"]
    assert page_slug(fixes[0].rewritten) == slug


def test_page_slug_rejects_untrusted_urls():
    assert page_slug("https://ppv.st/live/abc-1") == "abc-1"
    assert page_slug("https://ppv.st/live/a/b/c/d") == "a/b/c/d"
    assert page_slug("https://ppv.st/live/a/b/c/d/e") is None
    assert page_slug("https://ppv.st/live/a/../api") is None
    assert page_slug("https://ppv.st/live/a//b") is None
    assert page_slug("https://evil.com/live/abc-1") is None
    assert page_slug("https://ppv.st/live/../../api") is None
    assert page_slug("") is None


def test_rewrite_fixes_are_not_cards():
    assert all(not f.card for f in find_fixes("https://x.com/a/status/1 https://instagram.com/p/A"))
