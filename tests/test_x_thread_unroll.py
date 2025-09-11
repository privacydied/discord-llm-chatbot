from pathlib import Path

import pytest

from bot.threads.x_thread_unroll import unroll_author_thread


def _read_fixture(name: str) -> str:
    p = Path(__file__).parent / "fixtures" / "x" / name
    return p.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_single_tweet_minimal(monkeypatch):
    html = _read_fixture("single_tweet.html")

    async def fake_fetch(url: str, timeout_s: float):
        return html

    async def no_expand(url: str, timeout_s: float):
        return url

    from bot.threads import x_thread_unroll as xu

    monkeypatch.setattr(xu, "_fetch_html_with_playwright", fake_fetch)
    monkeypatch.setattr(xu, "_expand_tco_if_needed", no_expand)

    ctx, reason = await unroll_author_thread(
        "https://x.com/author/status/9999", timeout_s=5, max_tweets=30, max_chars=6000
    )
    assert ctx is not None, reason
    assert ctx.tweet_count == 1
    assert "Single tweet only" in ctx.joined_text


@pytest.mark.asyncio
async def test_mid_thread_collects_all_in_order(monkeypatch):
    html = _read_fixture("thread_7_mid.html")

    async def fake_fetch(url: str, timeout_s: float):
        return html

    async def no_expand(url: str, timeout_s: float):
        return url

    from bot.threads import x_thread_unroll as xu
    monkeypatch.setattr(xu, "_fetch_html_with_playwright", fake_fetch)
    monkeypatch.setattr(xu, "_expand_tco_if_needed", no_expand)

    # Linked to the 4/7 tweet
    ctx, reason = await unroll_author_thread(
        "https://x.com/author/status/4444", timeout_s=5, max_tweets=30, max_chars=6000
    )
    assert ctx is not None, reason
    assert ctx.tweet_count == 7
    # Ensure chronological order and coverage
    assert "[1/7] @author" in ctx.joined_text
    assert "First tweet of thread" in ctx.joined_text.splitlines()[1]
    assert "Seventh tweet" in ctx.joined_text


@pytest.mark.asyncio
async def test_non_author_replies_are_skipped(monkeypatch):
    html = _read_fixture("with_other_replies.html")

    async def fake_fetch(url: str, timeout_s: float):
        return html

    async def no_expand(url: str, timeout_s: float):
        return url

    from bot.threads import x_thread_unroll as xu
    monkeypatch.setattr(xu, "_fetch_html_with_playwright", fake_fetch)
    monkeypatch.setattr(xu, "_expand_tco_if_needed", no_expand)

    ctx, reason = await unroll_author_thread(
        "https://x.com/author/status/1001", timeout_s=5, max_tweets=30, max_chars=6000
    )
    assert ctx is not None, reason
    # Only two author tweets should be present (other user's reply ignored)
    assert ctx.tweet_count == 2
    assert "Other user's reply" not in ctx.joined_text


@pytest.mark.asyncio
async def test_limits_cap_items_and_mark_truncated(monkeypatch):
    from datetime import datetime
    # Generate 45 tweet HTMLs for same author
    parts = ["<html><body>"]
    base_time = 1672531200  # 2023-01-01T00:00:00Z
    for i in range(45):
        tid = 10000 + i
        ts = base_time + i * 60
        parts.append(
            f"<article><a href=\"/author/status/{tid}\">link</a>"
            f"<a href=\"/author\">@author</a>"
            f"<time datetime=\"{datetime.utcfromtimestamp(ts).isoformat()}Z\"></time>"
            f"<div data-testid=\"tweetText\">Tweet number {i+1}.</div>"
            f"</article>"
        )
    parts.append("</body></html>")
    html = "".join(parts)

    async def fake_fetch(url: str, timeout_s: float):
        return html

    async def no_expand(url: str, timeout_s: float):
        return url

    from bot.threads import x_thread_unroll as xu
    monkeypatch.setattr(xu, "_fetch_html_with_playwright", fake_fetch)
    monkeypatch.setattr(xu, "_expand_tco_if_needed", no_expand)

    ctx, reason = await unroll_author_thread(
        "https://x.com/author/status/10000", timeout_s=5, max_tweets=30, max_chars=6000
    )
    assert ctx is not None, reason
    assert ctx.tweet_count == 30
    assert ctx.truncated is True


@pytest.mark.asyncio
async def test_dom_change_or_timeout_fallback(monkeypatch):
    async def fake_fetch(url: str, timeout_s: float):
        return "<html><body><div>No tweet structure here</div></body></html>"

    async def no_expand(url: str, timeout_s: float):
        return url

    from bot.threads import x_thread_unroll as xu
    monkeypatch.setattr(xu, "_fetch_html_with_playwright", fake_fetch)
    monkeypatch.setattr(xu, "_expand_tco_if_needed", no_expand)

    ctx, reason = await unroll_author_thread(
        "https://x.com/author/status/424242", timeout_s=5, max_tweets=30, max_chars=6000
    )
    assert ctx is None
    assert reason in {"dom_mismatch", "no_items"}


@pytest.mark.asyncio
async def test_mirrors_and_mobile_urls_normalize(monkeypatch):
    # Use a valid HTML so fetch path works
    html = _read_fixture("single_tweet.html")

    async def fake_fetch(url: str, timeout_s: float):
        return html

    async def no_expand(url: str, timeout_s: float):
        return url

    from bot.threads import x_thread_unroll as xu
    monkeypatch.setattr(xu, "_fetch_html_with_playwright", fake_fetch)
    monkeypatch.setattr(xu, "_expand_tco_if_needed", no_expand)

    for u in [
        "https://mobile.twitter.com/author/status/9999",
        "https://fxtwitter.com/author/status/9999",
        "https://vxtwitter.com/author/status/9999",
        "https://fixupx.com/author/status/9999",
    ]:
        ctx, reason = await unroll_author_thread(
            u, timeout_s=5, max_tweets=30, max_chars=6000
        )
        assert ctx is not None, reason
        assert ctx.tweet_count == 1
        assert "Single tweet only" in ctx.joined_text


@pytest.mark.asyncio
async def test_fx_meta_redirect_single_block(monkeypatch):
    """If fx/vx mirrors only expose meta + redirect, synthesize a single block from meta."""
    html = (
        "<!DOCTYPE html><html><head>"
        '<link rel="canonical" href="https://x.com/author/status/424242" />'
        '<meta property="twitter:creator" content="@author" />'
        '<meta property="og:description" content="Single tweet only via meta." />'
        "</head><body>Redirecting...</body></html>"
    )

    async def fake_fetch(url: str, timeout_s: float):
        return html

    async def no_expand(url: str, timeout_s: float):
        return url

    from bot.threads import x_thread_unroll as xu
    monkeypatch.setattr(xu, "_fetch_html_with_playwright", fake_fetch)
    monkeypatch.setattr(xu, "_expand_tco_if_needed", no_expand)

    ctx, reason = await unroll_author_thread(
        "https://fxtwitter.com/author/status/424242", timeout_s=5, max_tweets=30, max_chars=6000
    )
    assert ctx is not None, reason
    assert ctx.tweet_count == 1
    assert "Single tweet only via meta." in ctx.joined_text
