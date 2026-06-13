"""Lightweight YouTube transcript resolver (caption-track first).

This module is intentionally small and fail-open:
- It only runs for YouTube URLs.
- It tries caption tracks before any audio download/transcription pipeline.
- On any parsing/network issue, callers can fall back to existing STT flow.
"""

from __future__ import annotations

import asyncio
import contextlib
import html as html_lib
import json
import os
import re
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from .http_client import RequestConfig, get_http_client
from .utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)

_YT_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
}

_YT_ID_RE = re.compile(r"^[0-9A-Za-z_-]{6,}$")
_CACHE_DIR = Path("cache/youtube_transcripts")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class YouTubeTranscriptResult:
    video_id: str
    url: str
    text: str
    title: str
    uploader: str
    duration_s: float
    source: str
    language: str
    cache_hit: bool
    cached_at: float


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    norm = str(raw).strip().lower()
    if norm in {"1", "true", "yes", "on", "enabled"}:
        return True
    if norm in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = float(str(raw).strip())
        return val if val > 0 else default
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = int(str(raw).strip())
        return val if val >= 0 else default
    except Exception:
        return default


def _preferred_langs() -> list[str]:
    raw = (os.getenv("YOUTUBE_TRANSCRIPT_PREFERRED_LANGS") or "en,en-US").strip()
    vals = [part.strip() for part in raw.split(",") if part.strip()]
    return vals or ["en", "en-US"]


def _cache_path(video_id: str) -> Path:
    return _CACHE_DIR / f"{video_id}.json"


def is_youtube_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        return host in _YT_HOSTS
    except Exception:
        return False


def is_youtube_shorts(url: str) -> bool:
    """Check if a YouTube URL is a Shorts video."""
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if host not in _YT_HOSTS:
            return False
        path = (parsed.path or "").strip()
        return path.startswith("/shorts/")
    except Exception:
        return False


def extract_youtube_video_id(url: str) -> str | None:
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").strip()

        if host in {"youtu.be", "www.youtu.be"}:
            vid = path.lstrip("/").split("/", 1)[0].strip()
            return vid if _YT_ID_RE.fullmatch(vid) else None

        if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            if path == "/watch":
                query = parse_qs(parsed.query)
                vid = (query.get("v") or [""])[0].strip()
                return vid if _YT_ID_RE.fullmatch(vid) else None

            for prefix in ("/shorts/", "/embed/", "/live/", "/v/"):
                if path.startswith(prefix):
                    vid = path[len(prefix) :].split("/", 1)[0].strip()
                    return vid if _YT_ID_RE.fullmatch(vid) else None
    except Exception:
        return None
    return None


def _watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _upsert_query(url: str, key: str, value: str) -> str:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query, keep_blank_values=True)
    qs[key] = [value]
    query = urlencode(qs, doseq=True)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            query,
            parsed.fragment,
        ),
    )


def _iter_caption_urls(base_url: str) -> Iterable[str]:
    candidate_urls: list[str] = []
    json3 = _upsert_query(base_url, "fmt", "json3")
    candidate_urls.append(json3)
    if base_url not in candidate_urls:
        candidate_urls.append(base_url)
    srv3 = _upsert_query(base_url, "fmt", "srv3")
    if srv3 not in candidate_urls:
        candidate_urls.append(srv3)
    return candidate_urls


def _extract_json_after_marker(text: str, marker: str) -> dict[str, Any] | None:
    idx = text.find(marker)
    if idx < 0:
        return None
    start = text.find("{", idx)
    if start < 0:
        return None

    depth = 0
    in_str = False
    escaped = False
    end = -1

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end <= start:
        return None

    raw = text[start:end]
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _extract_player_response(html_text: str) -> dict[str, Any] | None:
    markers = [
        "ytInitialPlayerResponse =",
        "window['ytInitialPlayerResponse'] =",
        'window["ytInitialPlayerResponse"] =',
    ]
    for marker in markers:
        data = _extract_json_after_marker(html_text, marker)
        if data:
            return data
    return None


def _normalize_transcript_text(text: str) -> str:
    unescaped = html_lib.unescape(text or "")
    return re.sub(r"\s+", " ", unescaped).strip()


def _parse_json3_transcript(data: dict[str, Any]) -> str:
    events = data.get("events")
    if not isinstance(events, list):
        return ""

    lines: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        segs = event.get("segs")
        if not isinstance(segs, list):
            continue
        parts: list[str] = []
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            token = seg.get("utf8")
            if isinstance(token, str):
                parts.append(token)
        line = "".join(parts).replace("\n", " ").strip()
        if line:
            lines.append(line)
    return _normalize_transcript_text(" ".join(lines))


def _parse_vtt_transcript(raw_text: str) -> str:
    lines: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.upper().startswith("WEBVTT"):
            continue
        if line.upper().startswith("NOTE"):
            continue
        if "-->" in line:
            continue
        if line.isdigit():
            continue
        lines.append(line)
    return _normalize_transcript_text(" ".join(lines))


def _parse_xml_transcript(raw_text: str) -> str:
    try:
        root = ET.fromstring(raw_text)  # nosec B314
    except Exception:
        return ""

    entries: list[str] = []
    for node in root.iter():
        tag = str(node.tag).split("}", 1)[-1].lower()
        if tag not in {"text", "p", "span"}:
            continue
        val = "".join(node.itertext()).replace("\n", " ").strip()
        if val:
            entries.append(val)

    if not entries:
        fallback = " ".join(part.strip() for part in root.itertext() if part.strip())
        return _normalize_transcript_text(fallback)

    return _normalize_transcript_text(" ".join(entries))


def _parse_caption_payload(raw_text: str) -> str:
    body = (raw_text or "").strip()
    if not body:
        return ""

    # Try JSON first (fmt=json3).
    if body.startswith("{") and '"events"' in body:
        try:
            data = json.loads(body)
        except Exception:
            data = None
        if isinstance(data, dict):
            text = _parse_json3_transcript(data)
            if text:
                return text

    if body.lstrip().upper().startswith("WEBVTT"):
        return _parse_vtt_transcript(body)

    return _parse_xml_transcript(body)


def _track_is_asr(track: dict[str, Any]) -> bool:
    kind = str(track.get("kind") or "").strip().lower()
    if kind == "asr":
        return True
    name_obj = track.get("name") or {}
    if isinstance(name_obj, dict):
        simple = str(name_obj.get("simpleText") or "").lower()
        if "auto" in simple:
            return True
    return False


def _lang_rank(lang: str, preferred: list[str]) -> int:
    code = (lang or "").strip().lower()
    if not code:
        return len(preferred) + 1
    for idx, pref in enumerate(preferred):
        pref_norm = pref.lower()
        if code == pref_norm or code.startswith(pref_norm):
            return idx
    return len(preferred)


def _sort_tracks(tracks: list[dict[str, Any]], preferred: list[str]) -> list[dict[str, Any]]:
    def _score(track: dict[str, Any]) -> tuple:
        lang = str(track.get("languageCode") or "")
        asr = 1 if _track_is_asr(track) else 0
        return (asr, _lang_rank(lang, preferred))

    return sorted(tracks, key=_score)


async def _fetch_text(url: str, timeout_s: float) -> str:
    client = await get_http_client()
    cfg = RequestConfig(
        connect_timeout=min(timeout_s, 3.0),
        read_timeout=timeout_s,
        total_timeout=timeout_s + 0.5,
        max_retries=0,
    )
    try:
        resp = await client.get(url, config=cfg)
    except Exception:
        return ""
    if resp.status_code != 200:
        return ""
    return resp.text or ""


def _find_ytdlp_bin() -> str | None:
    for env_key in ("YOUTUBE_TRANSCRIPT_YTDLP_BIN", "YTDLP_BIN", "YT_DLP_BIN"):
        value = (os.getenv(env_key) or "").strip()
        if value and os.path.isfile(value) and os.access(value, os.X_OK):
            return value

    which_val = shutil.which("yt-dlp")
    if which_val:
        return which_val

    for local_path in ("./.venv/bin/yt-dlp", ".venv/bin/yt-dlp"):
        if os.path.isfile(local_path) and os.access(local_path, os.X_OK):
            return local_path
    return None


def _apply_cookie_args(cmd: list[str]) -> None:
    browser = (os.getenv("VIDEO_COOKIES_FROM_BROWSER") or "").strip()
    cookie_file = (os.getenv("VIDEO_COOKIES_FILE") or "").strip()
    if browser:
        cmd.extend(["--cookies-from-browser", browser])
        return
    if cookie_file:
        cmd.extend(["--cookies", cookie_file])


def _parse_json_object(stdout: str) -> dict[str, Any] | None:
    body = (stdout or "").strip()
    if not body:
        return None
    try:
        data = json.loads(body)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.debug(f"json parse failed: {exc}")

    for line in reversed(body.splitlines()):
        row = line.strip()
        if not row.startswith("{"):
            continue
        try:
            data = json.loads(row)
        except Exception as exc:
            logger.debug(f"json line parse failed: {exc}")
            continue
        if isinstance(data, dict):
            return data
    return None


async def _run_ytdlp_probe(url: str, timeout_s: float) -> dict[str, Any] | None:
    ytdlp_bin = _find_ytdlp_bin()
    if not ytdlp_bin:
        return None

    cmd: list[str] = [
        ytdlp_bin,
        "--dump-single-json",
        "--skip-download",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]
    _apply_cookie_args(cmd)
    cmd.extend(["--", url])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception:
        return None

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        with contextlib.suppress(Exception):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.communicate()
        return None

    if proc.returncode != 0:
        stderr_text = (stderr_bytes or b"").decode("utf-8", errors="ignore")
        logger.debug(
            "yt.transcript.ytdlp_probe_failed code=%s err=%s",
            str(proc.returncode),
            stderr_text[:220],
        )
        return None

    stdout_text = (stdout_bytes or b"").decode("utf-8", errors="ignore")
    return _parse_json_object(stdout_text)


def _entry_ext_rank(ext: str) -> int:
    norm = (ext or "").strip().lower()
    if norm == "json3":
        return 0
    if norm == "vtt":
        return 1
    if norm in {"ttml", "xml"}:
        return 2
    if norm in {"srv3", "srv1"}:
        return 3
    return 4


def _collect_ytdlp_caption_entries(payload: dict[str, Any], source_label: str) -> list[dict[str, str]]:
    source = payload.get(source_label)
    if not isinstance(source, dict):
        return []
    out: list[dict[str, str]] = []
    for lang, items in source.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            out.append(
                {
                    "url": url,
                    "lang": str(lang or ""),
                    "ext": str(item.get("ext") or ""),
                    "protocol": str(item.get("protocol") or ""),
                    "source": source_label,
                },
            )
    return out


def _sort_ytdlp_caption_entries(entries: list[dict[str, str]], preferred: list[str]) -> list[dict[str, str]]:
    def _score(entry: dict[str, str]) -> tuple:
        source_rank = 0 if entry.get("source") == "subtitles" else 1
        lang_rank = _lang_rank(entry.get("lang", ""), preferred)
        ext_rank = _entry_ext_rank(entry.get("ext", ""))
        return (source_rank, lang_rank, ext_rank)

    return sorted(entries, key=_score)


def _parse_m3u8_lines(m3u8_text: str, base_url: str, max_segments: int) -> list[str]:
    urls: list[str] = []
    for line in m3u8_text.splitlines():
        row = line.strip()
        if not row or row.startswith("#"):
            continue
        urls.append(urljoin(base_url, row))
        if max_segments > 0 and len(urls) >= max_segments:
            break
    return urls


async def _resolve_from_caption_url(
    caption_url: str,
    timeout_s: float,
    max_chars: int,
    max_segments: int,
    depth: int = 0,
) -> str:
    payload = await _fetch_text(caption_url, timeout_s=timeout_s)
    if not payload:
        return ""

    body = payload.strip()
    if body.startswith("#EXTM3U"):
        # Handle caption playlists from yt-dlp automatic_captions URLs.
        if depth >= 2:
            return ""
        segment_urls = _parse_m3u8_lines(body, base_url=caption_url, max_segments=max_segments)
        if not segment_urls:
            return ""
        parts: list[str] = []
        current_len = 0
        for seg_url in segment_urls:
            chunk = await _resolve_from_caption_url(
                seg_url,
                timeout_s=timeout_s,
                max_chars=max_chars,
                max_segments=max_segments,
                depth=depth + 1,
            )
            if not chunk:
                continue
            parts.append(chunk)
            current_len += len(chunk)
            if max_chars > 0 and current_len >= max_chars:
                break
        merged = _normalize_transcript_text(" ".join(parts))
        if max_chars > 0 and len(merged) > max_chars:
            return merged[:max_chars].rstrip() + "..."
        return merged

    text = _parse_caption_payload(payload)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


async def _resolve_via_ytdlp_captions(
    url: str,
    video_id: str,
    title: str,
    uploader: str,
    duration_s: float,
    timeout_s: float,
    max_chars: int,
    preferred_langs: list[str],
) -> YouTubeTranscriptResult | None:
    if not _env_bool("YOUTUBE_TRANSCRIPT_YTDLP_FALLBACK", True):
        return None

    probe_timeout = _env_float("YOUTUBE_TRANSCRIPT_YTDLP_TIMEOUT_S", 12.0)
    payload = await _run_ytdlp_probe(url, timeout_s=max(probe_timeout, timeout_s))
    if not isinstance(payload, dict):
        return None

    vid = str(payload.get("id") or "").strip() or video_id
    title = str(payload.get("title") or title or "Unknown Title")
    uploader = str(payload.get("uploader") or payload.get("channel") or uploader or "Unknown")
    try:
        duration_s = float(payload.get("duration") or duration_s or 0.0)
    except Exception:
        duration_s = float(duration_s or 0.0)

    entries = _collect_ytdlp_caption_entries(payload, "subtitles")
    entries.extend(_collect_ytdlp_caption_entries(payload, "automatic_captions"))
    if not entries:
        return None
    entries = _sort_ytdlp_caption_entries(entries, preferred=preferred_langs)

    max_segments = _env_int("YOUTUBE_TRANSCRIPT_MAX_SEGMENTS", 80)
    for entry in entries:
        transcript = await _resolve_from_caption_url(
            caption_url=entry.get("url", ""),
            timeout_s=timeout_s,
            max_chars=max_chars,
            max_segments=max_segments,
        )
        if not transcript:
            continue
        source_tag = "ytdlp_subtitles" if entry.get("source") == "subtitles" else "ytdlp_automatic_captions"
        return YouTubeTranscriptResult(
            video_id=vid,
            url=url,
            text=transcript,
            title=title,
            uploader=uploader,
            duration_s=float(duration_s or 0.0),
            source=source_tag,
            language=str(entry.get("lang") or ""),
            cache_hit=False,
            cached_at=time.time(),
        )
    return None


def _load_cache(video_id: str, ttl_s: int) -> YouTubeTranscriptResult | None:
    if ttl_s <= 0:
        return None
    path = _cache_path(video_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        cached_at = float(data.get("cached_at") or 0.0)
        if (time.time() - cached_at) > float(ttl_s):
            return None
        text = str(data.get("text") or "").strip()
        if not text:
            return None
        return YouTubeTranscriptResult(
            video_id=str(data.get("video_id") or video_id),
            url=str(data.get("url") or _watch_url(video_id)),
            text=text,
            title=str(data.get("title") or "Unknown Title"),
            uploader=str(data.get("uploader") or "Unknown"),
            duration_s=float(data.get("duration_s") or 0.0),
            source=str(data.get("source") or "captionTracks"),
            language=str(data.get("language") or ""),
            cache_hit=True,
            cached_at=cached_at,
        )
    except Exception:
        return None


def _store_cache(result: YouTubeTranscriptResult) -> None:
    path = _cache_path(result.video_id)
    payload = {
        "video_id": result.video_id,
        "url": result.url,
        "text": result.text,
        "title": result.title,
        "uploader": result.uploader,
        "duration_s": result.duration_s,
        "source": result.source,
        "language": result.language,
        "cached_at": result.cached_at,
    }
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except Exception:
        logger.debug("Failed to write YouTube transcript cache", exc_info=True)


async def resolve_youtube_transcript(url: str, force_refresh: bool = False) -> YouTubeTranscriptResult | None:
    """Resolve YouTube transcript from caption tracks (without yt-dlp/audio decode).
    Returns None when unavailable or on non-YouTube URLs.
    """
    if not is_youtube_url(url):
        return None

    video_id = extract_youtube_video_id(url)
    if not video_id:
        return None

    ttl_s = _env_int("YOUTUBE_TRANSCRIPT_CACHE_TTL_S", 86400)
    if not force_refresh:
        cached = _load_cache(video_id, ttl_s=ttl_s)
        if cached:
            return cached

    timeout_s = _env_float("YOUTUBE_TRANSCRIPT_TIMEOUT_S", 8.0)
    allow_asr = _env_bool("YOUTUBE_TRANSCRIPT_ALLOW_ASR", True)
    max_chars = _env_int("YOUTUBE_TRANSCRIPT_MAX_CHARS", 120000)
    preferred_langs = _preferred_langs()
    watch_url = _watch_url(video_id)
    title = "Unknown Title"
    uploader = "Unknown"
    duration_s = 0.0

    html_text = await _fetch_text(watch_url, timeout_s=timeout_s)
    if html_text:
        player = _extract_player_response(html_text)
        if isinstance(player, dict):
            details = player.get("videoDetails") or {}
            title = str(details.get("title") or title)
            uploader = str(details.get("author") or uploader)
            try:
                duration_s = float(details.get("lengthSeconds") or duration_s)
            except Exception:
                duration_s = float(duration_s or 0.0)

            caps = (player.get("captions") or {}).get("playerCaptionsTracklistRenderer", {}).get("captionTracks", [])
            if isinstance(caps, list) and caps:
                tracks = [track for track in caps if isinstance(track, dict)]
                tracks = _sort_tracks(tracks, preferred=preferred_langs)

                for track in tracks:
                    asr = _track_is_asr(track)
                    if asr and not allow_asr:
                        continue

                    base_url = html_lib.unescape(str(track.get("baseUrl") or "")).strip()
                    if not base_url:
                        continue

                    lang = str(track.get("languageCode") or "")
                    for caption_url in _iter_caption_urls(base_url):
                        transcript = await _resolve_from_caption_url(
                            caption_url=caption_url,
                            timeout_s=timeout_s,
                            max_chars=max_chars,
                            max_segments=_env_int("YOUTUBE_TRANSCRIPT_MAX_SEGMENTS", 80),
                        )
                        if not transcript:
                            continue
                        result = YouTubeTranscriptResult(
                            video_id=video_id,
                            url=url,
                            text=transcript,
                            title=title,
                            uploader=uploader,
                            duration_s=duration_s,
                            source="captionTracks",
                            language=lang,
                            cache_hit=False,
                            cached_at=time.time(),
                        )
                        _store_cache(result)
                        return result

    fallback = await _resolve_via_ytdlp_captions(
        url=url,
        video_id=video_id,
        title=title,
        uploader=uploader,
        duration_s=duration_s,
        timeout_s=timeout_s,
        max_chars=max_chars,
        preferred_langs=preferred_langs,
    )
    if fallback:
        _store_cache(fallback)
        return fallback
    return None
