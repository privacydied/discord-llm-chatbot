"""Short-TTL, in-process cache for yt-dlp metadata payloads. [PA][RM][CMV]

The YouTube STT path probes the same video **twice**: once from the caption-track
resolver (``youtube_transcript._resolve_via_ytdlp_captions``) and again seconds
later from the audio ingest stage (``video_ingest._probe_metadata``). Both are a
``yt-dlp --dump-json`` subprocess against the same URL, and the first payload was
thrown away when the video had no usable caption tracks -- so a YouTube job paid
two cold metadata probes and could exhaust its budget before downloading a byte.

This cache lets the second probe reuse the first payload. It is deliberately
small and dumb:

- keyed by ``(identity, cookie_signature)`` so a cookie-authenticated probe never
  serves a cookie-less caller (their visible formats differ),
- TTL'd well below the lifetime of yt-dlp's signed format URLs,
- bounded entry count, since one payload can be several hundred KB of formats,
- single-video payloads only: playlist dumps have no ``formats`` and would break
  format selection downstream.

Payloads are shared by reference and MUST be treated as read-only by callers.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from typing import Any

from .utils.logging import get_logger

logger = get_logger(__name__)

# Signed CDN URLs inside a yt-dlp payload outlive this comfortably; the window
# only needs to span one STT job's probe -> download sequence. [CMV]
DEFAULT_TTL_S = 90.0
# One YouTube payload is O(100s of KB); keep the ceiling low so a busy bot can't
# accumulate tens of MB of stale format lists. [RM]
DEFAULT_MAX_ENTRIES = 8

NO_COOKIES = "none"

_cache: OrderedDict[tuple[str, str], tuple[float, dict[str, Any]]] = OrderedDict()


def _ttl_seconds() -> float:
    try:
        return float(os.getenv("YTDLP_PROBE_CACHE_TTL_S", str(DEFAULT_TTL_S)))
    except (TypeError, ValueError):
        return DEFAULT_TTL_S


def _max_entries() -> int:
    try:
        return max(1, int(os.getenv("YTDLP_PROBE_CACHE_MAX_ENTRIES", str(DEFAULT_MAX_ENTRIES))))
    except (TypeError, ValueError):
        return DEFAULT_MAX_ENTRIES


def key_for_youtube_id(video_id: str) -> str:
    """Cache key for a YouTube video id.

    Must stay identical to ``VideoIngestionManager._canonicalize_video_identity``
    for YouTube URLs -- ``tests/test_ytdlp_probe_cache.py`` pins the two together.
    """
    return f"youtube:video/{video_id}"


def cookie_signature(*, browser: str | None = None, cookie_file: str | None = None) -> str:
    """Identify which cookie source a probe used, so payloads aren't cross-served."""
    if browser:
        return f"browser:{browser}"
    if cookie_file:
        return f"file:{cookie_file}"
    return NO_COOKIES


def is_cacheable(payload: Any) -> bool:
    """True for single-video payloads with a usable format list. [IV]

    Playlist dumps (``--dump-single-json`` without ``--no-playlist`` on a
    ``watch?v=..&list=..`` URL) carry ``entries`` and no ``formats``; reusing one
    for audio selection would fail with "No audio-capable formats".
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("_type") == "playlist" or "entries" in payload:
        return False
    return bool(payload.get("formats") or payload.get("requested_downloads"))


def get(identity: str, signature: str) -> dict[str, Any] | None:
    """Return a fresh cached payload, or None. Expired entries are dropped."""
    if not identity:
        return None
    entry = _cache.get((identity, signature))
    if entry is None:
        return None
    stored_at, payload = entry
    if (time.monotonic() - stored_at) > _ttl_seconds():
        _cache.pop((identity, signature), None)
        return None
    _cache.move_to_end((identity, signature))
    return payload


def put(identity: str, signature: str, payload: Any) -> None:
    """Cache a probe payload if it is a reusable single-video dump."""
    if not identity or not is_cacheable(payload):
        return
    key = (identity, signature)
    _cache[key] = (time.monotonic(), payload)
    _cache.move_to_end(key)
    while len(_cache) > _max_entries():
        evicted, _ = _cache.popitem(last=False)
        logger.debug("ytdlp.probe_cache.evict identity=%s", evicted[0][:60])


def clear() -> None:
    """Drop every entry (test hook / hot-reload safety)."""
    _cache.clear()
