"""Tool: look at an image posted earlier in the channel.
[CA][REH][IV][SFT][PA].

Solves the "huh, what image?" problem. The bot describes an image, the turn
ends, and the picture is gone -- both context managers persist only
``{author_id, content, timestamp}``, with no attachment URL and no message id,
so nothing can point back at it. ``channel.history()`` however returns real
Message objects with ``.attachments`` intact, so the image can simply be found
again and re-read on demand.

No temp files are involved. ``see_infer`` requires a local path, but the layer
beneath it (``generate_vl_response`` -> ``get_base64_image``) fetches http(s)
URLs directly, so this tool passes the Discord CDN URL straight through and
never touches the filesystem. [SFT]
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from bot.utils.logging import get_logger

from ..types import ToolContext, ToolResult, ToolSpec

logger = get_logger(__name__)

# How far back to hunt for an image when the model does not say. [CMV][PA]
MAX_IMAGE_LOOKBACK = 50

# Vision inference is slow; this tool needs far more than the 10s default. [PA]
VIEW_IMAGE_TIMEOUT_S = 45.0

# Trim the model's description so one image cannot dominate the prompt. [CMV]
MAX_DESCRIPTION_CHARS = 1500

DEFAULT_QUESTION = "Describe this image in detail."

PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "posts_ago": {
            "type": "integer",
            "description": (
                "Which message to take the image from, counting back from the current one. "
                "Omit this to use the most recent image in the channel, which is usually what "
                "'that image' or 'the picture you just looked at' refers to."
            ),
            "minimum": 1,
            "maximum": MAX_IMAGE_LOOKBACK,
        },
        "question": {
            "type": "string",
            "description": "What to find out about the image. Omit for a general description.",
        },
    },
    "required": [],
}

DESCRIPTION = (
    "Look at an image that was posted earlier in this channel and answer a question about it. "
    "Use this whenever the conversation refers back to a picture, screenshot, meme or photo that is "
    "not attached to the current message -- including when you described it yourself a few messages ago. "
    "Do not claim you cannot see an image before trying this."
)


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _image_refs(message: Any) -> list[Any]:
    """Image references on a message, via the shared harvester. Never raises."""
    try:
        from bot.modality import collect_image_urls_from_message

        return list(collect_image_urls_from_message(message) or [])
    except Exception as exc:  # [REH]
        logger.debug("tool.view_image.harvest_failed error=%s", exc)
        return []


async def _find_image(channel: Any, anchor: Any, posts_ago: int | None) -> tuple[Any, Any, int] | str:
    """Locate (image_ref, message, position) or return an error string. [REH]"""
    limit = posts_ago if posts_ago else MAX_IMAGE_LOOKBACK
    try:
        history = [msg async for msg in channel.history(limit=limit, before=anchor)]
    except Exception as exc:
        name = type(exc).__name__
        if name == "Forbidden":
            return "missing permission to read message history in this channel"
        logger.warning("tool.view_image.history_failed error=%s", exc)
        return f"could not read channel history ({name})"

    if posts_ago:
        if len(history) < posts_ago:
            return f"channel history does not go back {posts_ago} messages"
        target = history[posts_ago - 1]
        refs = _image_refs(target)
        if not refs:
            return f"the message {posts_ago} posts ago has no image attached"
        return refs[0], target, posts_ago

    # No position given: walk back to the most recent message carrying an image.
    for index, msg in enumerate(history):
        refs = _image_refs(msg)
        if refs:
            return refs[0], msg, index + 1
    return f"no image found in the last {len(history)} messages"


# Discord re-signs CDN URLs, so these query parameters change for the same
# image and must not enter a URL-derived cache key. [CMV]
_VOLATILE_QUERY_KEYS = frozenset({"ex", "is", "hm"})

# Discord attachment URLs are /attachments/<channel_id>/<attachment_id>/<name>.
_ATTACHMENTS_SEGMENT = "attachments"


def normalize_url(url: str) -> str:
    """A URL stripped of expiring signatures, for use as a last-resort key.

    Discord CDN links carry ``?ex=&is=&hm=`` parameters refreshed periodically,
    so the raw URL is a poor key. Other query parameters are kept, because
    hosts use them to select a rendition (``?format=png&size=4096``).
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    kept = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() not in _VOLATILE_QUERY_KEYS]
    return urlunparse(parsed._replace(query=urlencode(sorted(kept)), fragment=""))


def attachment_token(url: str, filename: str | None = None) -> str:
    """The attachment id from a Discord CDN URL, else a stable stand-in.

    The id is the third path segment after ``attachments`` and is permanent,
    unlike the signed query string wrapped around it.
    """
    try:
        segments = [s for s in (urlparse(url).path or "").split("/") if s]
    except ValueError:
        segments = []
    if _ATTACHMENTS_SEGMENT in segments:
        index = segments.index(_ATTACHMENTS_SEGMENT)
        if len(segments) > index + 2:
            return segments[index + 2]
    # Embedded/external images have no attachment id; the filename distinguishes
    # multiple images on one message, and the normalised URL is the backstop.
    return (filename or "").strip() or normalize_url(url)


def cache_identity(message: Any, ref: Any) -> str:
    """Stable identity for an image, keyed on message and attachment.

    Preferred over any URL-derived key: a host can rotate an entire URL path
    for the same picture, but ``(message_id, attachment_id)`` is permanent for
    as long as the message exists. Falls back to the normalised URL only when
    the message carries no id (in practice, tests and synthetic contexts).
    """
    url = str(getattr(ref, "url", "") or "")
    token = attachment_token(url, getattr(ref, "filename", None))
    message_id = getattr(message, "id", None)
    if message_id:
        return f"m{message_id}:{token}"
    return f"u:{normalize_url(url)}"


class _VisionUnavailable(RuntimeError):
    """Raised inside the cached computation so failures are never stored."""


async def _run_vl(url: str, question: str) -> str:
    """Call the vision model. Raises _VisionUnavailable rather than returning None.

    ``see_infer`` requires a local path, but the layer beneath it fetches
    http(s) directly, so no temp file is needed. [PA]
    """
    from bot.ai_backend import generate_vl_response

    try:
        result = await generate_vl_response(image_url=url, user_prompt=question)
    except Exception as exc:  # [REH]
        logger.warning("tool.view_image.vl_failed error=%s", exc)
        raise _VisionUnavailable(str(exc)) from exc

    text = (result or {}).get("text") if isinstance(result, dict) else None
    cleaned = str(text).strip() if text else ""
    if not cleaned:
        raise _VisionUnavailable("empty description")
    return cleaned


async def _describe(url: str, question: str, cfg: dict[str, Any], identity: str | None = None) -> str | None:
    """Describe an image, reusing a cached description when possible. [PA][REH]

    ``identity`` is the cache key source -- normally ``(message_id,
    attachment_id)`` from :func:`cache_identity`, which survives a host
    rotating the URL entirely. Falls back to the normalised URL when absent.

    Returns None on any failure, and failures are never cached -- the compute
    raises so get_or_compute stores nothing, with negative caching off.
    """
    from bot.url_safety import UrlSafetyError, validate_url_with_dns

    # Embedded images can point anywhere a user linked, so validate. [SFT]
    try:
        await validate_url_with_dns(url)
    except UrlSafetyError as exc:
        logger.warning("tool.view_image.blocked_url reason=%s", exc)
        return None
    except Exception as exc:  # [REH]
        logger.debug("tool.view_image.validate_failed error=%s", exc)
        return None

    try:
        from bot.single_flight_cache import CacheFamily, get_cache

        cache = get_cache(cfg)
    except Exception as exc:  # [REH] cache must never be load-bearing
        logger.debug("tool.view_image.cache_unavailable error=%s", exc)
        try:
            return await _run_vl(url, question)
        except _VisionUnavailable:
            return None

    # The question is part of the key: a cached "describe this" is the wrong
    # answer to "what colour is the car?". [CMV]
    key_parts = [identity or f"u:{normalize_url(url)}", question.strip().lower()]
    try:
        description, was_hit = await cache.get_or_compute(
            CacheFamily.VL_DESCRIPTION,
            key_parts,
            lambda: _run_vl(url, question),
            negative_on_exception=False,
        )
    except _VisionUnavailable:
        return None
    except Exception as exc:  # [REH]
        logger.warning("tool.view_image.cache_failed error=%s", exc)
        return None

    logger.info("tool.view_image.described cache_hit=%s", was_hit)
    return description


def _provenance(message: Any, position: int, ref: Any) -> str:
    author = getattr(getattr(message, "author", None), "display_name", None) or getattr(getattr(message, "author", None), "name", None) or "unknown"
    stamp = getattr(message, "created_at", None)
    when = stamp.strftime("%Y-%m-%d %H:%M UTC") if stamp else "unknown time"
    filename = getattr(ref, "filename", None) or "image"
    return f"Image from {position} posts ago, posted by {author} at {when} ({filename})"


async def view_image(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    """Re-read an image posted earlier in the channel. Never raises. [REH]"""
    raw_posts_ago = arguments.get("posts_ago")
    posts_ago = None if raw_posts_ago is None else _coerce_int(raw_posts_ago)
    if raw_posts_ago is not None and posts_ago is None:
        return ToolResult.failure("posts_ago must be an integer")
    if posts_ago is not None and (posts_ago < 1 or posts_ago > MAX_IMAGE_LOOKBACK):
        return ToolResult.failure(f"posts_ago must be between 1 and {MAX_IMAGE_LOOKBACK}")

    question = str(arguments.get("question") or DEFAULT_QUESTION).strip() or DEFAULT_QUESTION

    channel = ctx.channel
    if channel is None or not hasattr(channel, "history"):
        return ToolResult.failure("no channel available to look for images in")

    found = await _find_image(channel, ctx.message, posts_ago)
    if isinstance(found, str):
        return ToolResult.failure(found)
    ref, message, position = found

    url = str(getattr(ref, "url", "") or "")
    if not url:
        return ToolResult.failure("image reference had no usable URL")

    identity = cache_identity(message, ref)
    logger.info("tool.view_image.describing position=%d identity=%s", position, identity[:60])
    description = await _describe(url, question, ctx.config or {}, identity)
    if not description:
        return ToolResult.failure("could not read that image")

    if len(description) > MAX_DESCRIPTION_CHARS:
        description = description[:MAX_DESCRIPTION_CHARS] + "…"

    from bot.url_safety import wrap_untrusted_content

    body = f"{_provenance(message, position, ref)}\n\n{description}"
    return ToolResult.success(wrap_untrusted_content(body, source="discord-image"))


SPEC = ToolSpec(
    name="view_image",
    description=DESCRIPTION,
    parameters=PARAMETERS,
    handler=view_image,
    timeout_s=VIEW_IMAGE_TIMEOUT_S,
)
