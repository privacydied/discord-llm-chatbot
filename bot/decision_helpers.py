"""Pure helper functions for routing decisions.

Each helper returns explicit decisions and logs a one-line breadcrumb using the
existing structured logging schema. They are designed to be deterministic and
side-effect free so rare routing bugs can be isolated and tested easily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import re
import time

try:  # pragma: no cover - fallback when optional deps missing
    from bot.utils.logging import get_logger
except Exception:  # pragma: no cover
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore
        return logging.getLogger(name)


logger = get_logger(__name__)


@dataclass
class ScopeResult:
    """Resolved scope information."""

    case: str
    scope_id: Optional[str]


def resolve_scope(trigger: object) -> ScopeResult:
    """Determine routing scope for a trigger message.

    The function inspects common Discord attributes (``thread`` or ``reference``)
    but only relies on ``getattr`` so tests can pass in lightweight mocks.

    Returns a :class:`ScopeResult` with ``case`` in ``{"thread", "reply", "plain"}``.
    """

    case = "plain"
    scope_id: Optional[str] = None
    try:
        if (
            getattr(trigger, "channel", None)
            and getattr(getattr(trigger, "channel", None), "id", None)
            and getattr(trigger, "channel", None).__class__.__name__ == "Thread"
        ):
            case = "thread"
            scope_id = str(getattr(getattr(trigger, "channel", None), "id", None))
        elif getattr(trigger, "reference", None) and getattr(
            getattr(trigger, "reference", None), "message_id", None
        ):
            case = "reply"
            scope_id = str(
                getattr(getattr(trigger, "reference", None), "message_id", None)
            )
        else:
            scope_id = str(getattr(trigger, "id", None))
    except Exception:
        scope_id = str(getattr(trigger, "id", None))

    logger.info(
        "scope_resolved",
        extra={
            "subsys": "router",
            "event": "scope_resolved",
            "msg_id": getattr(trigger, "id", None),
            "detail": {"case": case, "scope": scope_id},
        },
    )
    return ScopeResult(case=case, scope_id=scope_id)


def extract_chat_text(messages: Iterable[str]) -> Dict[str, object]:
    """Join messages, strip mentions and report text presence."""

    cleaned: List[str] = []
    for m in messages:
        without_mentions = re.sub(r"<@!?\d+>", "", m)
        cleaned.append(without_mentions.strip())
    joined = " ".join(filter(None, cleaned))
    has_text = bool(re.search(r"\S|[!?]", joined))
    logger.info(
        "chat_text",
        extra={
            "subsys": "router",
            "event": "chat_text",
            "detail": {"has_text": has_text, "length": len(joined)},
        },
    )
    return {"text": joined, "has_text_flag": has_text}


MEDIA_RE = re.compile(r"\b(photo|image|picture|screenshot|video|media)\b", re.I)


def detect_media_intent(text: str) -> bool:
    """Return True if the text explicitly requests media handling."""

    intent = bool(MEDIA_RE.search(text))
    logger.info(
        "media_intent",
        extra={"subsys": "router", "event": "media_intent", "detail": intent},
    )
    return intent


def harvest_in_scope_io(
    scope_messages: Iterable[Dict[str, Iterable[str]]],
) -> Dict[str, List[str]]:
    """Collect URLs and attachments from in-scope messages."""

    urls: List[str] = []
    attachments: List[str] = []
    for msg in scope_messages:
        urls.extend(msg.get("urls", []))
        attachments.extend(msg.get("attachments", []))
    logger.info(
        "harvest_complete",
        extra={
            "subsys": "router",
            "event": "harvest_complete",
            "detail": {"urls": len(urls), "attachments": len(attachments)},
        },
    )
    return {"urls": urls, "attachments": attachments}


def choose_route(
    has_text: bool, media_intent: bool, harvested: Dict[str, List[str]]
) -> str:
    """Select a route: ``text``, ``media`` or ``nag``."""

    route = "text"
    has_media = bool(harvested.get("urls") or harvested.get("attachments"))
    if media_intent and not has_media:
        route = "nag"
    elif media_intent:
        route = "media"
    elif not has_text:
        route = "nag"
    logger.info(
        "route_selected",
        extra={"subsys": "router", "event": "route_selected", "detail": route},
    )
    return route


def select_reply_target(
    case: str, scope_id: Optional[str], now: float | None = None
) -> Optional[str]:
    """Decide final reply target based on case and scope id."""

    if now is None:
        now = time.time()
    target = scope_id if case in {"thread", "reply"} else None
    logger.info(
        "reply_target_ok",
        extra={
            "subsys": "router",
            "event": "reply_target_ok",
            "detail": target,
            "ts": now,
        },
    )
    return target


def compose_context(messages: List[str], limits: Dict[str, int]) -> Dict[str, object]:
    """Deduplicate and truncate context messages."""

    max_items = limits.get("max_items", len(messages))
    max_chars = limits.get("max_chars", 1000)
    deduped: List[str] = []
    seen = set()
    for m in messages:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    truncated = deduped[-max_items:]
    joined = "\n".join(truncated)[:max_chars]
    logger.info(
        "local_context",
        extra={
            "subsys": "router",
            "event": "local_context",
            "detail": {"items": len(truncated), "chars": len(joined)},
        },
    )
    return {"items": truncated, "joined_text": joined}
