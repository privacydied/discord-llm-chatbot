"""Unroll an X/Twitter thread from X's own logged-out status page.
[CA][REH][IV].

``https://x.com/<user>/status/<id>`` is server-rendered for logged-out
visitors: the HTML carries X's own ``TweetDetail`` conversation timeline as a
serialized JSON string. The author's continuation posts come as
``SelfThread`` items and, when the link points mid-thread, the earlier posts as
ancestor entries. No third-party service, API key or browser is involved.

``unroll_thread`` returns the author's posts in order, or None when the post
is not part of a multi-post thread (the caller then handles it as a single
post). If a page ends with a "show more" cursor, the page of the last post is
fetched to continue, up to ``MAX_PAGES``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from bot.http_client import RequestConfig, get_http_client
from bot.utils.logging import get_logger

logger = get_logger(__name__)

STATUS_URL = "https://x.com/i/status/{id}"  # x.com redirects to the canonical /<user>/status/<id>
MAX_PAGES = 4
DEFAULT_MAX_POSTS = 100
DEFAULT_MAX_CHARS = 20000
MIN_THREAD_POSTS = 2
SELF_THREAD = "SelfThread"

_SCRIPT_RE = re.compile(r"<script[^>]*>(.*?)</script>", re.S)
# Seroval-serialized loader result: {kind:"Result.Ok",value:"<JSON as a JS string>"}
_RESULT_RE = re.compile(r'kind:"Result\.Ok",value:("(?:[^"\\]|\\.)*")')
_JS_HEX_RE = re.compile(r"\\x([0-9a-fA-F]{2})")
_TRAILING_TCO_RE = re.compile(r"(?:\s*https://t\.co/\w+)+\s*$")
_STATUS_ID_RE = re.compile(r"/status(?:es)?/(\d{1,25})")


class ThreadFetchError(RuntimeError):
    """The status page could not be fetched or held no conversation data."""


@dataclass
class Post:
    id: str
    handle: str
    name: str
    text: str
    created_at: datetime | None = None
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)  # "photo", "video", "animated_gif"
    quoted: str | None = None  # "@user: text" of a quoted post
    self_thread: bool = False


@dataclass
class Thread:
    posts: list[Post]
    url: str
    truncated: bool = False

    @property
    def author(self) -> Post:
        return self.posts[0]


def status_id_from_url(url: str) -> str | None:
    match = _STATUS_ID_RE.search(url or "")
    return match.group(1) if match else None


# --- page -> JSON -------------------------------------------------------------


def page_payloads(html: str) -> list[dict[str, Any]]:
    """Every JSON loader result serialized into the page's inline scripts."""
    payloads: list[dict[str, Any]] = []
    for script in _SCRIPT_RE.finditer(html or ""):
        for match in _RESULT_RE.finditer(script.group(1)):
            literal = _JS_HEX_RE.sub(r"\\u00\1", match.group(1))
            try:
                value = json.loads(json.loads(literal))
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(value, dict):
                payloads.append(value)
    return payloads


def _timeline_entries(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for payload in payloads:
        conversation = (payload.get("data") or {}).get("threaded_conversation_with_injections_v2")
        if isinstance(conversation, dict):
            return [e for ins in conversation.get("instructions") or [] for e in ins.get("entries") or []]
    return []


def _entry_contents(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Item contents of one timeline entry (a single tweet or a conversation module)."""
    content = entry.get("content") or {}
    items = content.get("items")
    if isinstance(items, list):
        return [(i.get("item") or {}).get("content") or {} for i in items]
    return [content.get("content") or content]


# --- tweet result -> Post -----------------------------------------------------


def _result(node: dict[str, Any]) -> dict[str, Any]:
    result = ((node or {}).get("tweet_results") or {}).get("result") or node or {}
    return result.get("tweet") or result if result.get("__typename") == "TweetWithVisibilityResults" else result


def _user(result: dict[str, Any]) -> tuple[str, str]:
    user = ((result.get("core") or {}).get("user_results") or {}).get("result") or {}
    core = user.get("core") or user.get("legacy") or {}
    return str(core.get("screen_name") or ""), str(core.get("name") or "")


def _text(result: dict[str, Any], media: list[str]) -> str:
    note = (((result.get("note_tweet") or {}).get("note_tweet_results") or {}).get("result") or {}).get("text")
    details = result.get("details") or {}
    legacy = result.get("legacy") or {}
    text = str(note or details.get("full_text") or legacy.get("full_text") or "")
    urls = list(result.get("url_entities") or []) + list((legacy.get("entities") or {}).get("urls") or [])
    for entity in urls:
        if isinstance(entity, dict) and entity.get("url") and entity.get("expanded_url"):
            text = text.replace(str(entity["url"]), str(entity["expanded_url"]))
    return (_TRAILING_TCO_RE.sub("", text) if media else text).strip()


def _created(result: dict[str, Any]) -> datetime | None:
    ms = (result.get("details") or {}).get("created_at_ms")
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=UTC) if ms else None
    except (TypeError, ValueError, OverflowError):
        return None


def _media(result: dict[str, Any]) -> list[str]:
    items = result.get("media_entities2") or ((result.get("legacy") or {}).get("extended_entities") or {}).get("media") or []
    return [str(m.get("type") or "media") for m in items if isinstance(m, dict)]


def _quoted(result: dict[str, Any]) -> str | None:
    for key in ("quoted_status_results", "quoted_tweet_results", "quoted_post_results"):
        inner = _result({"tweet_results": result.get(key)}) if result.get(key) else None
        if inner and inner.get("rest_id"):
            handle, _ = _user(inner)
            return f"@{handle}: {_text(inner, _media(inner))}"
    return None


def to_post(content: dict[str, Any]) -> Post | None:
    result = _result(content)
    rest_id = result.get("rest_id")
    if not rest_id or result.get("__typename") not in (None, "Tweet"):
        return None
    handle, name = _user(result)
    media = _media(result)
    reply_to = (result.get("reply_to_results") or {}).get("rest_id") or (result.get("legacy") or {}).get("in_reply_to_status_id_str")
    return Post(
        id=str(rest_id),
        handle=handle,
        name=name,
        text=_text(result, media),
        created_at=_created(result),
        reply_to=str(reply_to) if reply_to else None,
        media=media,
        quoted=_quoted(result),
        self_thread=content.get("display_type") == SELF_THREAD,
    )


def page_posts(html: str) -> tuple[list[Post], bool]:
    """(posts in page order, whether a 'show more' cursor was present)."""
    posts: list[Post] = []
    more = False
    for entry in _timeline_entries(page_payloads(html)):
        for content in _entry_contents(entry):
            if content.get("__typename") == "TimelineTimelineCursor" or content.get("cursor_type"):
                more = more or str(content.get("cursor_type") or "").lower() in ("showmore", "bottom", "showmorethreads")
                continue
            post = to_post(content)
            if post is not None:
                posts.append(post)
    return posts, more


# --- posts -> author thread ---------------------------------------------------


def _same_author(post: Post, handle: str) -> bool:
    return bool(handle) and post.handle.lower() == handle.lower()


def _ancestors(focal: Post, by_id: dict[str, Post]) -> list[Post]:
    """The author's earlier posts that ``focal`` replies to, oldest first."""
    chain: list[Post] = []
    current = focal
    while current.reply_to and current.reply_to in by_id and _same_author(by_id[current.reply_to], focal.handle):
        current = by_id[current.reply_to]
        if current in chain:
            break
        chain.append(current)
    return chain[::-1]


def _descendants(last: Post, posts: list[Post], seen: set[str]) -> list[Post]:
    """The author's replies to ``last``, each continuing the previous one."""
    replies = {p.reply_to: p for p in posts if p.reply_to and _same_author(p, last.handle) and p.id not in seen}
    chain: list[Post] = []
    current = last
    while current.id in replies:
        current = replies.pop(current.id)
        chain.append(current)
    return chain


def assemble(focal_id: str, posts: list[Post]) -> list[Post]:
    """Author-only chain through ``focal_id``: ancestors + focal + continuations."""
    by_id = {p.id: p for p in posts}
    focal = by_id.get(focal_id)
    if focal is None:
        return []
    chain = [*_ancestors(focal, by_id), focal]
    return chain + _descendants(focal, posts, {p.id for p in chain})


# --- fetching ------------------------------------------------------------------


async def fetch_status_page(status_id: str, *, timeout_s: float) -> str:
    """HTML of X's logged-out status page. Raises ThreadFetchError."""
    cfg = RequestConfig(
        connect_timeout=min(3.0, timeout_s),
        read_timeout=timeout_s,
        total_timeout=timeout_s,
        max_retries=1,
    )
    try:
        http = await get_http_client()
        response = await http.get(STATUS_URL.format(id=status_id), config=cfg)
    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        raise ThreadFetchError(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code != 200:
        raise ThreadFetchError(f"HTTP {response.status_code}")
    return response.text


async def unroll_thread(
    url: str,
    *,
    timeout_s: float = 10.0,
    max_posts: int = DEFAULT_MAX_POSTS,
) -> Thread | None:
    """The author's whole thread around the post at ``url``; None for a single post."""
    status_id = status_id_from_url(url)
    if not status_id:
        return None
    posts, more = page_posts(await fetch_status_page(status_id, timeout_s=timeout_s))
    if not posts:
        raise ThreadFetchError("no conversation data in page")
    chain = assemble(status_id, posts)
    for _ in range(MAX_PAGES - 1):
        if not more or not chain or len(chain) >= max_posts:
            break
        extra, more = page_posts(await fetch_status_page(chain[-1].id, timeout_s=timeout_s))
        tail = _descendants(chain[-1], extra, {p.id for p in chain})
        if not tail:
            break
        chain += tail
    if len(chain) < MIN_THREAD_POSTS:
        return None
    return Thread(posts=chain[:max_posts], url=f"https://x.com/{chain[0].handle}/status/{chain[0].id}", truncated=len(chain) > max_posts or more)


# --- rendering -------------------------------------------------------------------


def _render_post(index: int, total: int, post: Post) -> str:
    lines = [f"[{index}/{total}] {post.text}".rstrip()]
    if post.media:
        kinds = ", ".join(sorted({m.replace("animated_gif", "gif") for m in post.media}))
        lines.append(f"({len(post.media)} attached: {kinds})")
    if post.quoted:
        lines.append(f"(quoting {post.quoted})")
    return "\n".join(lines)


def render_thread(thread: Thread, *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Plain-text thread for the model: header, then every post in order."""
    first = thread.author
    when = f", posted {first.created_at:%d %b %Y}" if first.created_at else ""
    total = len(thread.posts)
    header = f"X thread by @{first.handle} ({first.name}), {total} posts{when}. {thread.url}"
    body: list[str] = []
    used = len(header)
    for index, post in enumerate(thread.posts, start=1):
        block = _render_post(index, total, post)
        if used + len(block) > max_chars:
            body.append(f"[thread cut here: {total - index + 1} more posts not shown]")
            break
        body.append(block)
        used += len(block) + 2
    if thread.truncated:
        body.append("[the thread continues beyond what could be loaded]")
    return header + "\n\n" + "\n\n".join(body)
