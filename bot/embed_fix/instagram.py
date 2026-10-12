"""Instagram posts as real attachments instead of a proxy link.

Discord no longer unfurls the kkinstagram proxy (it answers with a bare 302 to
the CDN file, no og: page), so the bot fetches the media itself and uploads it:

1. yt-dlp (no login needed for public reels/videos) gives the direct URL of
   the best single-file mp4, plus every video in a carousel.
2. Photo posts have no video for yt-dlp; the kkinstagram proxy still answers a
   Discordbot request with a 302 straight to the CDN image/video, so its
   ``Location`` header is used as the fallback.
3. Each URL is streamed with a hard byte cap (the guild's upload limit); the
   whole thing gives up after a total timeout and the caller falls back to the
   plain proxy link.

Nothing here touches Discord; the handler turns the result into
``discord.File`` objects.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from bot.utils.logging import get_logger

logger = get_logger(__name__)

_CODE_RE = re.compile(r"instagram\.com/(?:p|reels?|tv)/(?P<code>[A-Za-z0-9_-]+)", re.IGNORECASE)
_PROXY_HOST = "www.kkinstagram.com"
_PROXY_UA = "Mozilla/5.0 (compatible; Discordbot/2.0; +https://discordapp.com)"
_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


class _SilentLogger:
    """yt-dlp prints ERROR lines to stderr even with quiet=True; failures are expected here."""

    def debug(self, msg: str) -> None:
        pass

    info = warning = error = debug


_YTDLP_OPTS: dict[str, Any] = {
    "logger": _SilentLogger(),
    "quiet": True,
    "no_warnings": True,
    # Single-file format only: DASH needs an ffmpeg merge, progressive mp4 has audio.
    "format": "b[ext=mp4]/b",
    "socket_timeout": 15,
    "noplaylist": False,
}
_MEDIA_EXTS = {"mp4", "mov", "webm", "jpg", "jpeg", "png", "webp", "heic"}
_CONTENT_TYPE_EXT = {"video/mp4": "mp4", "image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "video/quicktime": "mov"}
_CHUNK = 64 * 1024

MAX_ITEMS = 10  # Discord's per-message attachment cap
DEFAULT_TIMEOUT_S = 45.0
# Leave room for multipart overhead under the guild's upload limit.
_UPLOAD_HEADROOM = 256 * 1024


@dataclass(frozen=True, slots=True)
class MediaFile:
    filename: str
    data: bytes


def shortcode(url: str) -> str | None:
    m = _CODE_RE.search(url or "")
    return m["code"] if m else None


def upload_budget(filesize_limit: int | None) -> int:
    """Usable bytes per message for a guild (discord.Guild.filesize_limit)."""
    limit = int(filesize_limit or 10 * 1024 * 1024)
    return max(0, limit - _UPLOAD_HEADROOM)


def _ytdlp_urls(post_url: str) -> list[str]:
    """Blocking: direct media URLs for the post's videos (empty for photo posts)."""
    import yt_dlp  # heavy import; only paid when an Instagram link shows up

    with yt_dlp.YoutubeDL(_YTDLP_OPTS) as ydl:
        info = ydl.extract_info(post_url, download=False)
    entries = info.get("entries") if isinstance(info, dict) else None
    items = [e for e in (entries or [info]) if isinstance(e, dict)]
    return [str(e["url"]) for e in items if str(e.get("url") or "").startswith("https://")][:MAX_ITEMS]


async def _resolve_via_ytdlp(post_url: str) -> list[str]:
    try:
        return await asyncio.to_thread(_ytdlp_urls, post_url)
    except Exception as exc:  # DownloadError for photo posts, network, extractor drift
        logger.debug(f"embed_fix.instagram.ytdlp_failed | url={post_url} error={str(exc)[:200]}")
        return []


async def _resolve_via_proxy(session: aiohttp.ClientSession, code: str, kind: str) -> list[str]:
    url = f"https://{_PROXY_HOST}/{kind}/{code}/"
    try:
        async with session.get(url, headers={"User-Agent": _PROXY_UA}, allow_redirects=False) as resp:
            location = resp.headers.get("Location", "")
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.debug(f"embed_fix.instagram.proxy_failed | code={code} error={exc}")
        return []
    return [location] if location.startswith("https://") else []


def _extension(url: str, content_type: str) -> str:
    ext = _CONTENT_TYPE_EXT.get(content_type.split(";")[0].strip().lower())
    if ext:
        return ext
    tail = urlsplit(url).path.rsplit(".", 1)
    guess = tail[-1].lower() if len(tail) == 2 else ""
    return guess if guess in _MEDIA_EXTS else "mp4"


async def _download(session: aiohttp.ClientSession, url: str, max_bytes: int) -> tuple[bytes, str] | None:
    """Stream *url*; None when it fails or would exceed *max_bytes*."""
    try:
        async with session.get(url, headers={"User-Agent": _BROWSER_UA}) as resp:
            if resp.status != 200 or (resp.content_length or 0) > max_bytes:
                return None
            buf = bytearray()
            async for chunk in resp.content.iter_chunked(_CHUNK):
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    return None
            return bytes(buf), resp.headers.get("Content-Type", "")
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.debug(f"embed_fix.instagram.download_failed | error={exc}")
        return None


async def _collect(post_url: str, max_bytes: int) -> list[MediaFile]:
    code = shortcode(post_url)
    if not code or max_bytes <= 0:
        return []
    kind = "reel" if re.search(r"/reels?/", post_url, re.IGNORECASE) else "p"
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        urls = await _resolve_via_ytdlp(post_url) or await _resolve_via_proxy(session, code, kind)
        files: list[MediaFile] = []
        remaining = max_bytes
        for index, url in enumerate(urls, start=1):
            got = await _download(session, url, remaining)
            if got is None:
                continue
            data, content_type = got
            suffix = f"_{index}" if len(urls) > 1 else ""
            files.append(MediaFile(f"instagram_{code}{suffix}.{_extension(url, content_type)}", data))
            remaining -= len(data)
    return files


async def fetch_media(post_url: str, *, max_bytes: int, timeout_s: float = DEFAULT_TIMEOUT_S) -> list[MediaFile]:
    """Media for one Instagram post, within *max_bytes* total. Empty on any failure."""
    try:
        return await asyncio.wait_for(_collect(post_url, max_bytes), timeout=timeout_s)
    except TimeoutError:
        logger.info(f"embed_fix.instagram.timeout | url={post_url} after_s={timeout_s}")
        return []
