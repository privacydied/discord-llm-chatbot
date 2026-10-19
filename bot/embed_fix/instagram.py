"""Instagram posts as real attachments plus a styled card, instead of a proxy link.

Discord no longer unfurls the kkinstagram proxy (it answers with a bare 302 to
the CDN file, no og: page), so the bot fetches the post itself:

Media, first source that yields URLs wins:
1. yt-dlp (no login needed for public reels/videos): the best single-file
   mp4, plus every video in a carousel.
2. oginstagram: a real og: page whose ``og:video`` / ``og:image`` point at a
   signed full-resolution copy (covers photo posts, which yt-dlp cannot do).
3. kkinstagram: a Discordbot request gets a 302 straight to the CDN file; its
   ``Location`` header is the last resort.

Metadata for the card (username, display name, caption, post time, likes,
comments, avatar) is merged from yt-dlp and the oginstagram post + profile
pages. Each media URL is streamed under a hard byte cap (the guild's upload
limit); the whole thing gives up after a total timeout and the caller falls
back to the plain proxy link.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

import aiohttp
import discord

from bot.utils.logging import get_logger

logger = get_logger(__name__)

_CODE_RE = re.compile(r"instagram\.com/(?:[A-Za-z0-9._]{1,30}/)?(?:p|reels?|tv)/(?P<code>[A-Za-z0-9_-]+)", re.IGNORECASE)
_USERNAME_RE = re.compile(r"^[A-Za-z0-9._]{1,30}$")
# og:title is "Display Name (@username)".
_OG_TITLE_RE = re.compile(r"^(?P<name>.*?)\s*\(@(?P<user>[A-Za-z0-9._]{1,30})\)\s*$")
OG_HOST = "www.oginstagram.com"
KK_HOST = "www.kkinstagram.com"
_PROXY_UA = "Mozilla/5.0 (compatible; Discordbot/2.0; +https://discordapp.com)"
_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
_MAX_PAGE_BYTES = 256 * 1024


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
_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}
_CONTENT_TYPE_EXT = {"video/mp4": "mp4", "image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "video/quicktime": "mov"}
_CHUNK = 64 * 1024

MAX_ITEMS = 10  # Discord's per-message attachment cap
DEFAULT_TIMEOUT_S = 45.0
# Leave room for multipart overhead under the guild's upload limit.
_UPLOAD_HEADROOM = 256 * 1024

CARD_COLOUR = 0xE1306C  # Instagram pink
_CAPTION_LIMIT = 600
_INSTAGRAM_ICON = "https://www.instagram.com/static/images/ico/favicon-192.png/68d99ba29cc8.png"


@dataclass(frozen=True, slots=True)
class MediaFile:
    filename: str
    data: bytes

    @property
    def is_image(self) -> bool:
        return self.filename.rsplit(".", 1)[-1].lower() in _IMAGE_EXTS


@dataclass(frozen=True, slots=True)
class PostMeta:
    username: str | None = None
    display_name: str | None = None
    caption: str | None = None
    published: datetime | None = None
    likes: int | None = None
    comments: int | None = None
    avatar_url: str | None = None

    def merged(self, other: PostMeta) -> PostMeta:
        """Fill fields this one lacks from *other*."""
        return PostMeta(**{name: getattr(self, name) or getattr(other, name) for name in PostMeta.__slots__})


@dataclass(frozen=True, slots=True)
class InstagramPost:
    url: str
    files: list[MediaFile] = field(default_factory=list)
    meta: PostMeta = field(default_factory=PostMeta)


def shortcode(url: str) -> str | None:
    m = _CODE_RE.search(url or "")
    return m["code"] if m else None


def upload_budget(filesize_limit: int | None) -> int:
    """Usable bytes per message for a guild (discord.Guild.filesize_limit)."""
    limit = int(filesize_limit or 10 * 1024 * 1024)
    return max(0, limit - _UPLOAD_HEADROOM)


# ---- sources ----------------------------------------------------------------


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _ytdlp_post(post_url: str) -> tuple[list[str], PostMeta]:
    """Blocking: direct video URLs (empty for photo posts) and metadata."""
    import yt_dlp  # heavy import; only paid when an Instagram link shows up

    with yt_dlp.YoutubeDL(_YTDLP_OPTS) as ydl:
        info = ydl.extract_info(post_url, download=False)
    if not isinstance(info, dict):
        return [], PostMeta()
    items = [e for e in (info.get("entries") or [info]) if isinstance(e, dict)]
    urls = [str(e["url"]) for e in items if str(e.get("url") or "").startswith("https://")][:MAX_ITEMS]
    first = items[0] if items else info
    username = str(info.get("channel") or first.get("channel") or "")
    timestamp = _as_int(info.get("timestamp") or first.get("timestamp"))
    meta = PostMeta(
        username=username if _USERNAME_RE.match(username) else None,
        display_name=str(info.get("uploader") or first.get("uploader") or "") or None,
        caption=str(info.get("description") or first.get("description") or "") or None,
        published=datetime.fromtimestamp(timestamp, tz=UTC) if timestamp else None,
        likes=_as_int(info.get("like_count") if "like_count" in info else first.get("like_count")),
        comments=_as_int(info.get("comment_count") if "comment_count" in info else first.get("comment_count")),
    )
    return urls, meta


async def _from_ytdlp(post_url: str) -> tuple[list[str], PostMeta]:
    try:
        return await asyncio.to_thread(_ytdlp_post, post_url)
    except Exception as exc:  # DownloadError for photo posts, network, extractor drift
        logger.debug(f"embed_fix.instagram.ytdlp_failed | url={post_url} error={str(exc)[:200]}")
        return [], PostMeta()


class _MetaParser(HTMLParser):
    """Collect <meta property|name=... content=...> (first value per key)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        a = dict(attrs)
        key, content = a.get("property") or a.get("name"), a.get("content")
        if key and content is not None:
            self.meta.setdefault(key.lower(), content)


def parse_og(html: str) -> dict[str, str]:
    parser = _MetaParser()
    parser.feed(html)
    return parser.meta


async def _get_text(session: aiohttp.ClientSession, url: str) -> str | None:
    try:
        async with session.get(url, headers={"User-Agent": _PROXY_UA}) as resp:
            if resp.status != 200:
                return None
            raw = await resp.content.read(_MAX_PAGE_BYTES)
            return raw.decode("utf-8", errors="replace")
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.debug(f"embed_fix.instagram.page_failed | url={url} error={exc}")
        return None


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def og_post_meta(og: dict[str, str]) -> tuple[list[str], PostMeta]:
    """Media URL (video preferred) and metadata from an oginstagram post page."""
    media = og.get("og:video") or og.get("og:video:secure_url") or og.get("og:image")
    urls = [media] if media and media.startswith("https://") else []
    m = _OG_TITLE_RE.match(og.get("og:title", ""))
    return urls, PostMeta(
        username=m["user"] if m else None,
        display_name=(m["name"] or None) if m else None,
        caption=og.get("og:image:alt") or og.get("og:description") or None,
        published=_parse_time(og.get("article:published_time")),
    )


async def _from_og(session: aiohttp.ClientSession, code: str, kind: str) -> tuple[list[str], PostMeta]:
    html = await _get_text(session, f"https://{OG_HOST}/{kind}/{code}/")
    return og_post_meta(parse_og(html)) if html else ([], PostMeta())


async def has_link_video(post_url: str, *, timeout_s: float = 8.0) -> bool:
    """True when oginstagram's page for the post carries a playable og:video.

    Bot-built embeds can never contain video; only Discord's own unfurl of a
    page with og:video (+ type/size, which oginstagram sets) plays inside the
    card. For such posts the plain oginstagram link is the better reply.
    """
    code = shortcode(post_url)
    if not code:
        return False
    kind = "reel" if re.search(r"/reels?/", post_url, re.IGNORECASE) else "p"
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            html = await _get_text(session, f"https://{OG_HOST}/{kind}/{code}/")
    except TimeoutError:
        return False
    video = parse_og(html).get("og:video", "") if html else ""
    return video.startswith("https://")


async def _avatar(session: aiohttp.ClientSession, username: str | None) -> str | None:
    if not username or not _USERNAME_RE.match(username):
        return None
    html = await _get_text(session, f"https://{OG_HOST}/{username}/")
    image = parse_og(html).get("og:image") if html else None
    return image if image and image.startswith("https://") else None


async def _from_kk(session: aiohttp.ClientSession, code: str, kind: str) -> list[str]:
    url = f"https://{KK_HOST}/{kind}/{code}/"
    try:
        async with session.get(url, headers={"User-Agent": _PROXY_UA}, allow_redirects=False) as resp:
            location = resp.headers.get("Location", "")
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.debug(f"embed_fix.instagram.proxy_failed | code={code} error={exc}")
        return []
    return [location] if location.startswith("https://") else []


# ---- download ---------------------------------------------------------------


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


async def _download_all(session: aiohttp.ClientSession, code: str, urls: list[str], max_bytes: int) -> list[MediaFile]:
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


async def _collect(post_url: str, max_bytes: int) -> InstagramPost | None:
    code = shortcode(post_url)
    if not code or max_bytes <= 0:
        return None
    kind = "reel" if re.search(r"/reels?/", post_url, re.IGNORECASE) else "p"
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        (yt_urls, yt_meta), (og_urls, og_meta) = await asyncio.gather(_from_ytdlp(post_url), _from_og(session, code, kind))
        meta = yt_meta.merged(og_meta)
        files = await _download_all(session, code, yt_urls, max_bytes) if yt_urls else []
        if not files and og_urls:
            files = await _download_all(session, code, og_urls, max_bytes)
        if not files:
            files = await _download_all(session, code, await _from_kk(session, code, kind), max_bytes)
        if not files:
            return None
        meta = replace(meta, avatar_url=await _avatar(session, meta.username))
    return InstagramPost(url=f"https://www.instagram.com/{kind}/{code}/", files=files, meta=meta)


async def fetch_post(post_url: str, *, max_bytes: int, timeout_s: float = DEFAULT_TIMEOUT_S) -> InstagramPost | None:
    """Media + metadata for one Instagram post within *max_bytes*. None on any failure."""
    try:
        return await asyncio.wait_for(_collect(post_url, max_bytes), timeout=timeout_s)
    except TimeoutError:
        logger.info(f"embed_fix.instagram.timeout | url={post_url} after_s={timeout_s}")
        return None


# ---- card -------------------------------------------------------------------


def _count(value: int) -> str:
    for size, suffix in ((1_000_000, "M"), (1_000, "K")):
        if value >= size:
            return f"{value / size:.1f}".rstrip("0").rstrip(".") + suffix
    return str(value)


def build_card(post: InstagramPost) -> discord.Embed:
    """Instagram-styled card. A photo goes inside the card; videos play as the attachment."""
    meta = post.meta
    caption = (meta.caption or "").strip()
    if len(caption) > _CAPTION_LIMIT:
        caption = caption[: _CAPTION_LIMIT - 1].rstrip() + "…"
    embed = discord.Embed(description=caption or None, colour=CARD_COLOUR, url=post.url, timestamp=meta.published)
    if meta.username:
        name = f"{meta.display_name} (@{meta.username})" if meta.display_name and meta.display_name != meta.username else f"@{meta.username}"
        embed.set_author(name=name, url=f"https://www.instagram.com/{meta.username}/", icon_url=meta.avatar_url)
    stats = []
    if meta.likes is not None:
        stats.append(f"❤️ {_count(meta.likes)}")
    if meta.comments:
        stats.append(f"💬 {_count(meta.comments)}")
    footer = "  ".join(["Instagram", *stats])
    embed.set_footer(text=footer, icon_url=_INSTAGRAM_ICON)
    first = post.files[0] if post.files else None
    if first is not None and first.is_image:
        embed.set_image(url=f"attachment://{first.filename}")
    return embed
