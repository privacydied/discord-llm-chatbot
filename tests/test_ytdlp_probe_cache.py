"""Tests for the yt-dlp probe payload cache and budget-aware stage timeouts."""

from __future__ import annotations

import pytest

from bot import video_ingest, ytdlp_probe_cache
from bot.time_budget import clear_deadline, set_deadline
from bot.video_ingest import VideoIngestionManager, _resolve_ytdlp_stage_timeouts

SINGLE_VIDEO_PAYLOAD = {"id": "abc123", "formats": [{"format_id": "140", "acodec": "mp4a"}]}


@pytest.fixture(autouse=True)
def _clean_cache():
    ytdlp_probe_cache.clear()
    yield
    ytdlp_probe_cache.clear()


def test_put_then_get_roundtrip():
    ytdlp_probe_cache.put("youtube:video/abc123", ytdlp_probe_cache.NO_COOKIES, SINGLE_VIDEO_PAYLOAD)
    assert ytdlp_probe_cache.get("youtube:video/abc123", ytdlp_probe_cache.NO_COOKIES) == SINGLE_VIDEO_PAYLOAD


def test_cookie_signature_isolates_entries():
    """A cookie-authenticated dump must not be served to a cookie-less caller."""
    ytdlp_probe_cache.put("youtube:video/abc123", "browser:firefox", SINGLE_VIDEO_PAYLOAD)
    assert ytdlp_probe_cache.get("youtube:video/abc123", ytdlp_probe_cache.NO_COOKIES) is None


def test_playlist_payload_is_not_cached():
    """Playlist dumps have no formats and would break audio selection downstream."""
    ytdlp_probe_cache.put("youtube:video/abc123", ytdlp_probe_cache.NO_COOKIES, {"_type": "playlist", "entries": []})
    assert ytdlp_probe_cache.get("youtube:video/abc123", ytdlp_probe_cache.NO_COOKIES) is None


@pytest.mark.parametrize("payload", [None, {}, {"id": "x"}, "not-a-dict"])
def test_uncacheable_payloads_rejected(payload):
    assert ytdlp_probe_cache.is_cacheable(payload) is False


def test_expired_entry_is_dropped(monkeypatch):
    monkeypatch.setenv("YTDLP_PROBE_CACHE_TTL_S", "0")
    ytdlp_probe_cache.put("youtube:video/abc123", ytdlp_probe_cache.NO_COOKIES, SINGLE_VIDEO_PAYLOAD)
    assert ytdlp_probe_cache.get("youtube:video/abc123", ytdlp_probe_cache.NO_COOKIES) is None


def test_entry_count_is_bounded(monkeypatch):
    monkeypatch.setenv("YTDLP_PROBE_CACHE_MAX_ENTRIES", "2")
    for i in range(4):
        ytdlp_probe_cache.put(f"youtube:video/v{i}", ytdlp_probe_cache.NO_COOKIES, dict(SINGLE_VIDEO_PAYLOAD))
    assert ytdlp_probe_cache.get("youtube:video/v0", ytdlp_probe_cache.NO_COOKIES) is None
    assert ytdlp_probe_cache.get("youtube:video/v3", ytdlp_probe_cache.NO_COOKIES) is not None


@pytest.mark.parametrize(
    "url",
    ["https://youtu.be/abc123", "https://www.youtube.com/watch?v=abc123", "https://www.youtube.com/shorts/abc123"],
)
def test_cache_key_matches_ingest_identity(url):
    """Pins the two key derivations together across modules.

    video_ingest keys on _canonicalize_video_identity; youtube_transcript keys on
    key_for_youtube_id. If they ever diverge the probe is silently paid twice.
    """
    assert VideoIngestionManager._canonicalize_video_identity(url) == ytdlp_probe_cache.key_for_youtube_id("abc123")


class TestStageTimeouts:
    """Env values are no-deadline defaults; a live deadline sizes the stages."""

    def test_defaults_apply_without_a_deadline(self, monkeypatch):
        clear_deadline_safely()
        monkeypatch.setattr(video_ingest, "load_config", lambda: {})
        assert _resolve_ytdlp_stage_timeouts() == (
            video_ingest.YTDLP_METADATA_TIMEOUT_DEFAULT_S,
            video_ingest.YTDLP_DOWNLOAD_TIMEOUT_DEFAULT_S,
        )

    def test_generous_budget_widens_the_probe(self, monkeypatch):
        """The regression: a 10s probe died while 200s of budget sat unused."""
        monkeypatch.setattr(video_ingest, "load_config", lambda: {"MEDIA_PER_ITEM_BUDGET": 120.0})
        token = set_deadline(216.0)
        try:
            metadata, download = _resolve_ytdlp_stage_timeouts()
        finally:
            clear_deadline(token)
        assert metadata > video_ingest.YTDLP_METADATA_TIMEOUT_DEFAULT_S
        assert metadata <= video_ingest.YTDLP_METADATA_CEILING_DEFAULT_S
        assert download > video_ingest.YTDLP_DOWNLOAD_TIMEOUT_DEFAULT_S

    def test_tight_deadline_narrows_below_defaults(self, monkeypatch):
        monkeypatch.setattr(video_ingest, "load_config", lambda: {})
        token = set_deadline(18.0)
        try:
            metadata, download = _resolve_ytdlp_stage_timeouts()
        finally:
            clear_deadline(token)
        assert metadata == video_ingest.YTDLP_METADATA_FLOOR_S
        assert download == video_ingest.YTDLP_DOWNLOAD_FLOOR_S

    def test_ambient_deadline_beats_a_larger_item_budget(self, monkeypatch):
        """The tighter of (deadline, per-item budget) must win."""
        monkeypatch.setattr(video_ingest, "load_config", lambda: {"MEDIA_PER_ITEM_BUDGET": 600.0})
        token = set_deadline(60.0)
        try:
            metadata, _ = _resolve_ytdlp_stage_timeouts()
        finally:
            clear_deadline(token)
        assert metadata == pytest.approx((60.0 - video_ingest.YTDLP_STAGE_RESERVE_S) * video_ingest.YTDLP_METADATA_BUDGET_SHARE)

    def test_ceilings_bound_an_unbounded_budget(self, monkeypatch):
        monkeypatch.setattr(video_ingest, "load_config", lambda: {})
        token = set_deadline(3600.0)
        try:
            metadata, download = _resolve_ytdlp_stage_timeouts()
        finally:
            clear_deadline(token)
        assert metadata == video_ingest.YTDLP_METADATA_CEILING_DEFAULT_S
        assert download == video_ingest.YTDLP_DOWNLOAD_CEILING_DEFAULT_S

    def test_unparseable_budget_is_ignored(self, monkeypatch):
        monkeypatch.setattr(video_ingest, "load_config", lambda: {"MEDIA_PER_ITEM_BUDGET": "not-a-number"})
        clear_deadline_safely()
        assert _resolve_ytdlp_stage_timeouts() == (
            video_ingest.YTDLP_METADATA_TIMEOUT_DEFAULT_S,
            video_ingest.YTDLP_DOWNLOAD_TIMEOUT_DEFAULT_S,
        )


def clear_deadline_safely() -> None:
    """Ensure no deadline leaks in from another test's context."""
    from bot import time_budget

    time_budget._deadline.set(None)
