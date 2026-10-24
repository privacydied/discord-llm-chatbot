"""Is this AI? -- pull a post's images and score them with Sightengine. [CA][REH][SFT]

Sources (all fixed hosts or the bot's existing fetchers, so nothing here fetches
an arbitrary user URL itself):

* X / Twitter (incl. vxtwitter, fxtwitter, fixupx, fixvx links): media via
  ``api.fxtwitter.com``. Photos are scored by URL; videos/GIFs by their
  thumbnail frame.
* Instagram (incl. oginstagram/ddinstagram links): ``embed_fix.instagram.fetch_post``,
  images uploaded as bytes; videos as a frame grabbed with ffmpeg.
* Discord attachments and image embeds: scored by URL (Discord CDN).

Sightengine's ``genai`` model looks at pixels only (metadata is ignored) and
returns ``type.ai_generated`` in [0, 1]. Each image costs 5 operations; the free
plan allows 500/day and 2,000/month, so a check is capped at ``MAX_IMAGES``.
A score is a probability, never proof -- callers must say so.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from bot.utils.logging import get_logger

logger = get_logger(__name__)

SIGHTENGINE_URL = "https://api.sightengine.com/1.0/check.json"
FXTWITTER_STATUS_URL = "https://api.fxtwitter.com/status/{tweet_id}"
MAX_IMAGES = 4
HTTP_TIMEOUT_S = 25.0
IG_MAX_BYTES = 25 * 1024 * 1024
FRAME_TIMEOUT_S = 20.0

_TWEET_RE = re.compile(
    r"https?://(?:www\.|mobile\.)?(?:twitter|x|vxtwitter|fxtwitter|fixupx|fixvx|twittpr|nitter\.[a-z.]+)\.com/[^/\s]+/status(?:es)?/(?P<id>\d{5,25})",
    re.IGNORECASE,
)
_IG_RE = re.compile(r"https?://(?:www\.)?[a-z]*instagram\.com/(?:[A-Za-z0-9._]{1,30}/)?(?:p|reels?|tv)/[A-Za-z0-9_-]+", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s<>()\"']+")
_IMAGE_EXT_RE = re.compile(r"\.(?:jpe?g|png|webp|gif|bmp|avif)(?:[?#]|$)", re.IGNORECASE)
_DISCORD_CDN = ("cdn.discordapp.com", "media.discordapp.net")


@dataclass(frozen=True)
class MediaSource:
    label: str  # "photo 1", "video frame", ...
    url: str | None = None
    data: bytes | None = None
    filename: str = "image.jpg"


@dataclass(frozen=True)
class Score:
    label: str
    ai_probability: float | None
    error: str | None = None


class DetectorNotConfigured(RuntimeError):
    """SIGHTENGINE_API_USER / SIGHTENGINE_API_SECRET missing."""


def credentials(config: dict[str, Any] | None = None) -> tuple[str, str] | None:
    cfg = config or {}
    user = str(cfg.get("SIGHTENGINE_API_USER") or os.getenv("SIGHTENGINE_API_USER") or "").strip().strip('"')
    secret = str(cfg.get("SIGHTENGINE_API_SECRET") or os.getenv("SIGHTENGINE_API_SECRET") or "").strip().strip('"')
    return (user, secret) if user and secret else None


def find_post_urls(text: str | None) -> list[str]:
    """Tweet and Instagram links in ``text``, in order, deduplicated."""
    out: list[str] = []
    for url in _URL_RE.findall(text or ""):
        url = url.rstrip(".,!?)>]")
        if (_TWEET_RE.match(url) or _IG_RE.match(url)) and url not in out:
            out.append(url)
    return out


# -- collecting media ---------------------------------------------------------------


async def _tweet_media(session: aiohttp.ClientSession, url: str) -> tuple[list[MediaSource], str | None]:
    match = _TWEET_RE.match(url)
    if not match:
        return [], None
    api = FXTWITTER_STATUS_URL.format(tweet_id=match["id"])
    async with session.get(api, timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)) as resp:
        if resp.status != 200:
            return [], f"couldn't load the tweet (HTTP {resp.status})"
        payload = await resp.json(content_type=None)
    tweet = (payload or {}).get("tweet") or {}
    items = ((tweet.get("media") or {}).get("all")) or []
    quote_items = (((tweet.get("quote") or {}).get("media") or {}).get("all")) or []
    sources: list[MediaSource] = []
    photos = videos = 0
    for item in [*items, *quote_items]:
        kind = str(item.get("type") or "")
        if kind == "photo" and item.get("url"):
            photos += 1
            sources.append(MediaSource(label=f"photo {photos}", url=str(item["url"])))
        elif kind in ("video", "gif") and item.get("thumbnail_url"):
            videos += 1
            sources.append(MediaSource(label=f"{kind} {videos} (thumbnail frame)", url=str(item["thumbnail_url"])))
    if not sources:
        return [], "that tweet has no images or videos to check"
    return sources, None


async def _video_frame(data: bytes) -> bytes | None:
    """First good frame of a video as JPEG, via the same ffmpeg the bot uses for audio."""
    from bot.stt_pipeline.ffmpeg_runtime import resolve_ffmpeg_bin

    try:
        ffmpeg = resolve_ffmpeg_bin()
    except RuntimeError:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.mp4", Path(tmp) / "frame.jpg"
        await asyncio.to_thread(src.write_bytes, data)
        # Fixed argv, no shell; input is a temp file we wrote.
        proc = await asyncio.create_subprocess_exec(
            ffmpeg, "-y", "-loglevel", "error", "-ss", "1", "-i", str(src), "-frames:v", "1", "-q:v", "2", str(dst),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )  # fmt: skip
        try:
            await asyncio.wait_for(proc.wait(), timeout=FRAME_TIMEOUT_S)
        except TimeoutError:
            proc.kill()
            return None
        return dst.read_bytes() if dst.exists() and dst.stat().st_size else None


async def _instagram_media(url: str) -> tuple[list[MediaSource], str | None]:
    from bot.embed_fix.instagram import fetch_post

    post = await fetch_post(url, max_bytes=IG_MAX_BYTES)
    if post is None or not post.files:
        return [], "couldn't load that instagram post"
    sources: list[MediaSource] = []
    for i, media in enumerate(post.files, start=1):
        if media.is_image:
            sources.append(MediaSource(label=f"image {i}", data=media.data, filename=media.filename))
            continue
        frame = await _video_frame(media.data)
        if frame:
            sources.append(MediaSource(label=f"video {i} (frame)", data=frame, filename="frame.jpg"))
    return (sources, None) if sources else ([], "couldn't get an image out of that instagram post")


def message_media(message: Any) -> list[MediaSource]:
    """Images attached to or embedded in a Discord message (scored by CDN URL)."""
    sources: list[MediaSource] = []
    for att in getattr(message, "attachments", None) or []:
        ctype = str(getattr(att, "content_type", "") or "").lower()
        name = str(getattr(att, "filename", "") or "")
        if ctype.startswith("image/") or _IMAGE_EXT_RE.search(name):
            sources.append(MediaSource(label=f"attachment {name or len(sources) + 1}", url=str(att.url)))
    for embed in getattr(message, "embeds", None) or []:
        for part in (getattr(embed, "image", None), getattr(embed, "thumbnail", None)):
            url = str(getattr(part, "url", "") or "")
            if url and any(host in url for host in (*_DISCORD_CDN, "pbs.twimg.com")):
                sources.append(MediaSource(label=f"embedded image {len(sources) + 1}", url=url))
                break
    return sources


async def collect_media(urls: list[str], messages: list[Any]) -> tuple[list[MediaSource], list[str]]:
    """Media from post links first, then attachments/embeds. Returns (sources, problems)."""
    sources: list[MediaSource] = []
    problems: list[str] = []
    async with aiohttp.ClientSession() as session:
        for url in urls:
            try:
                found, problem = await (_tweet_media(session, url) if _TWEET_RE.match(url) else _instagram_media(url))
            except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
                found, problem = [], f"couldn't load {url} ({type(exc).__name__})"
            sources.extend(found)
            if problem:
                problems.append(problem)
    if not sources:
        for message in messages:
            sources.extend(message_media(message))
    seen: set[str] = set()
    unique = []
    for src in sources:
        key = src.url or str(hash(src.data))
        if key not in seen:
            seen.add(key)
            unique.append(src)
    return unique[:MAX_IMAGES], problems


# -- scoring ------------------------------------------------------------------------


def _parse(payload: dict[str, Any]) -> tuple[float | None, str | None]:
    if payload.get("status") != "success":
        err = payload.get("error") or {}
        return None, str(err.get("message") or err.get("type") or "detector error")[:160]
    try:
        return float((payload.get("type") or {})["ai_generated"]), None
    except (KeyError, TypeError, ValueError):
        return None, "detector returned no score"


async def score_source(session: aiohttp.ClientSession, source: MediaSource, creds: tuple[str, str]) -> Score:
    user, secret = creds
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
    try:
        if source.url:
            params = {"url": source.url, "models": "genai", "api_user": user, "api_secret": secret}
            async with session.get(SIGHTENGINE_URL, params=params, timeout=timeout) as resp:
                payload = await resp.json(content_type=None)
        else:
            form = aiohttp.FormData()
            form.add_field("media", source.data or b"", filename=source.filename)
            for key, value in (("models", "genai"), ("api_user", user), ("api_secret", secret)):
                form.add_field(key, value)
            async with session.post(SIGHTENGINE_URL, data=form, timeout=timeout) as resp:
                payload = await resp.json(content_type=None)
    except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
        return Score(source.label, None, f"detector unreachable ({type(exc).__name__})")
    probability, error = _parse(payload if isinstance(payload, dict) else {})
    return Score(source.label, probability, error)


async def score_all(sources: list[MediaSource], creds: tuple[str, str]) -> list[Score]:
    async with aiohttp.ClientSession() as session:
        return list(await asyncio.gather(*(score_source(session, s, creds) for s in sources)))


def verdict(probability: float) -> str:
    if probability >= 0.85:
        return "very likely AI-generated"
    if probability >= 0.5:
        return "possibly AI-generated"
    if probability >= 0.15:
        return "probably not AI-generated"
    return "very likely real (not AI)"
