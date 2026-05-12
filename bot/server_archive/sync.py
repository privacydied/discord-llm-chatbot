"""Discord backfill helpers for the raw server archive."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

try:  # pragma: no cover - discord.py is available in the bot runtime
    import discord
except Exception:  # pragma: no cover
    discord = None  # type: ignore[assignment]

from .models import (
    ArchiveAttachment,
    ArchiveChannel,
    ArchiveGuild,
    ArchiveMessage,
    ArchiveMessageBundle,
    ArchiveMention,
    ArchiveSyncState,
    ArchiveThread,
    ArchiveUser,
    utc_now_iso,
)
from .store import ServerArchiveStore

logger = logging.getLogger(__name__)

_SYNC_ERRORS = tuple(
    exc
    for exc in (
        getattr(discord, "Forbidden", None) if discord else None,
        getattr(discord, "NotFound", None) if discord else None,
        getattr(discord, "HTTPException", None) if discord else None,
    )
    if exc is not None
)
# Backward-compatible alias for older tests and callers.
_SYNC_ERROR_TYPES = _SYNC_ERRORS


def _id(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def build_bundle_from_message(
    message: Any, *, max_message_chars: int = 8000, include_bot_messages: bool = False
) -> ArchiveMessageBundle | None:
    guild = getattr(message, "guild", None)
    channel = getattr(message, "channel", None)
    author = getattr(message, "author", None)
    if guild is None or channel is None or author is None:
        return None
    if not include_bot_messages and bool(getattr(author, "bot", False)):
        return None

    guild_id = _id(getattr(guild, "id", None))
    channel_id = _id(getattr(channel, "id", None))
    author_id = _id(getattr(author, "id", None))
    if not guild_id or not channel_id or not author_id:
        return None

    is_thread = (
        hasattr(channel, "parent_id")
        and getattr(channel, "parent_id", None) is not None
    )
    thread_id = _id(getattr(channel, "id", None)) if is_thread else None
    thread = (
        ArchiveThread(
            thread_id=thread_id or channel_id,
            guild_id=guild_id,
            parent_channel_id=_id(getattr(channel, "parent_id", None)) or channel_id,
            name=str(getattr(channel, "name", "")),
            archived_at=_iso(getattr(channel, "archived_at", None)) or utc_now_iso(),
            last_synced_message_id=None,
            last_synced_at=None,
        )
        if is_thread
        else None
    )

    content = str(getattr(message, "content", "") or "")
    clean_content = str(getattr(message, "clean_content", "") or content)
    if len(content) > max_message_chars:
        content = content[:max_message_chars]
    if len(clean_content) > max_message_chars:
        clean_content = clean_content[:max_message_chars]

    attachments = []
    for index, attachment in enumerate(getattr(message, "attachments", []) or []):
        attachment_id = (
            _id(getattr(attachment, "id", None))
            or f"{getattr(message, 'id', 'msg')}:{index}"
        )
        attachments.append(
            ArchiveAttachment(
                attachment_id=attachment_id,
                message_id=_id(getattr(message, "id", None)) or "",
                filename=getattr(attachment, "filename", None),
                content_type=getattr(attachment, "content_type", None),
                size=getattr(attachment, "size", None),
                url=str(getattr(attachment, "url", "") or ""),
                proxy_url=getattr(attachment, "proxy_url", None),
            )
        )

    mentions = []
    for user in getattr(message, "mentions", []) or []:
        mentioned_user_id = _id(getattr(user, "id", None))
        if mentioned_user_id:
            mentions.append(
                ArchiveMention(
                    message_id=_id(getattr(message, "id", None)) or "",
                    mentioned_user_id=mentioned_user_id,
                )
            )

    metadata = {
        "channel_type": type(channel).__name__,
        "attachments_count": len(attachments),
        "mentions_count": len(mentions),
        "bot_author": bool(getattr(author, "bot", False)),
        "has_embeds": int(bool(getattr(message, "embeds", []) or [])),
    }
    reference = getattr(message, "reference", None)
    reply_to_message_id = (
        _id(getattr(reference, "message_id", None)) if reference is not None else None
    )

    return ArchiveMessageBundle(
        guild=ArchiveGuild(
            guild_id=guild_id,
            name=str(getattr(guild, "name", "")),
            icon_url=_id(getattr(guild, "icon", None)),
        ),
        channel=ArchiveChannel(
            channel_id=channel_id,
            guild_id=guild_id,
            parent_id=_id(getattr(channel, "parent_id", None)),
            name=str(getattr(channel, "name", "")),
            type=type(channel).__name__,
            archived_at=utc_now_iso(),
            last_synced_message_id=None,
            last_synced_at=None,
        ),
        thread=thread,
        author=ArchiveUser(
            user_id=author_id,
            username=str(
                getattr(author, "name", None)
                or getattr(author, "global_name", None)
                or author_id
            ),
            global_name=getattr(author, "global_name", None),
            display_name=getattr(author, "display_name", None),
            bot=int(bool(getattr(author, "bot", False))),
        ),
        message=ArchiveMessage(
            message_id=_id(getattr(message, "id", None)) or "",
            guild_id=guild_id,
            channel_id=channel_id,
            thread_id=thread_id,
            author_id=author_id,
            content=content,
            clean_content=clean_content,
            created_at=_iso(getattr(message, "created_at", None)) or utc_now_iso(),
            edited_at=_iso(getattr(message, "edited_at", None)),
            jump_url=getattr(message, "jump_url", None),
            reply_to_message_id=reply_to_message_id,
            has_attachments=int(bool(attachments)),
            has_embeds=int(bool(getattr(message, "embeds", []) or [])),
            metadata_json=json.dumps(metadata, separators=(",", ":"), sort_keys=True),
        ),
        attachments=tuple(attachments),
        mentions=tuple(mentions),
    )


async def _sync_history_scope(
    store: ServerArchiveStore,
    source: Any,
    *,
    guild_id: str,
    channel_id: str,
    thread_id: str | None,
    force: bool,
    batch_size: int = 100,
    max_message_chars: int = 8000,
    include_bot_messages: bool = False,
) -> int:
    state = await store.get_sync_state(
        guild_id=guild_id, channel_id=channel_id, thread_id=thread_id
    )
    if state and state.status == "running" and not force:
        return 0
    if state and state.last_message_id and not force:
        after = (
            getattr(discord, "Object", lambda id: None)(id=int(state.last_message_id))
            if discord
            else None
        )
        history_kwargs: dict[str, Any] = {
            "after": after,
            "oldest_first": True,
            "limit": None,
        }
    else:
        history_kwargs = {"oldest_first": True, "limit": None}

    await store.set_sync_state(
        ArchiveSyncState(
            scope_key=f"{guild_id}:{channel_id}:{thread_id or ''}",
            guild_id=guild_id,
            channel_id=channel_id,
            thread_id=thread_id,
            last_message_id=state.last_message_id if state else None,
            last_synced_at=utc_now_iso(),
            status="running",
        )
    )

    processed = 0
    batch: list[ArchiveMessageBundle] = []
    last_message_id = state.last_message_id if state else None
    try:
        async for message in source.history(**history_kwargs):
            bundle = build_bundle_from_message(
                message,
                max_message_chars=max_message_chars,
                include_bot_messages=include_bot_messages,
            )
            if bundle is None:
                continue
            batch.append(bundle)
            if len(batch) >= batch_size:
                processed += await store.upsert_bundles(batch)
                last_message_id = batch[-1].message.message_id
                await store.set_sync_state(
                    ArchiveSyncState(
                        scope_key=f"{guild_id}:{channel_id}:{thread_id or ''}",
                        guild_id=guild_id,
                        channel_id=channel_id,
                        thread_id=thread_id,
                        last_message_id=last_message_id,
                        last_synced_at=utc_now_iso(),
                        status="running",
                    )
                )
                batch.clear()
        if batch:
            processed += await store.upsert_bundles(batch)
            last_message_id = batch[-1].message.message_id
        await store.set_sync_state(
            ArchiveSyncState(
                scope_key=f"{guild_id}:{channel_id}:{thread_id or ''}",
                guild_id=guild_id,
                channel_id=channel_id,
                thread_id=thread_id,
                last_message_id=last_message_id,
                last_synced_at=utc_now_iso(),
                status="complete",
            )
        )
        return processed
    except _SYNC_ERRORS as exc:
        logger.warning(
            "Server archive sync skipped due to permission/fetch error",
            extra={
                "subsys": "server_archive",
                "event": "archive_sync_permission_error",
                "detail": {
                    "guild_id": guild_id,
                    "channel_id": channel_id,
                    "thread_id": thread_id,
                    "error_type": type(exc).__name__,
                },
            },
        )
        await store.set_sync_state(
            ArchiveSyncState(
                scope_key=f"{guild_id}:{channel_id}:{thread_id or ''}",
                guild_id=guild_id,
                channel_id=channel_id,
                thread_id=thread_id,
                last_message_id=last_message_id,
                last_synced_at=utc_now_iso(),
                status="permission_error",
                error=type(exc).__name__,
            )
        )
        return processed
    except Exception as exc:
        logger.exception(
            "Server archive sync failed",
            extra={
                "subsys": "server_archive",
                "event": "archive_sync_failed",
                "detail": {
                    "guild_id": guild_id,
                    "channel_id": channel_id,
                    "thread_id": thread_id,
                    "error_type": type(exc).__name__,
                },
            },
        )
        await store.set_sync_state(
            ArchiveSyncState(
                scope_key=f"{guild_id}:{channel_id}:{thread_id or ''}",
                guild_id=guild_id,
                channel_id=channel_id,
                thread_id=thread_id,
                last_message_id=last_message_id,
                last_synced_at=utc_now_iso(),
                status="error",
                error=type(exc).__name__,
            )
        )
        return processed


async def sync_channel_archive(
    store: ServerArchiveStore, channel: Any, *, force: bool = False
) -> int:
    guild = getattr(channel, "guild", None)
    if guild is None:
        return 0
    guild_id = _id(getattr(guild, "id", None))
    channel_id = _id(getattr(channel, "id", None))
    if not guild_id or not channel_id:
        return 0
    thread_id = channel_id if getattr(channel, "parent_id", None) is not None else None
    return await _sync_history_scope(
        store,
        channel,
        guild_id=guild_id,
        channel_id=channel_id,
        thread_id=thread_id,
        force=force,
    )


async def sync_thread_archive(
    store: ServerArchiveStore, thread: Any, *, force: bool = False
) -> int:
    guild = getattr(thread, "guild", None)
    if guild is None:
        return 0
    guild_id = _id(getattr(guild, "id", None))
    thread_id = _id(getattr(thread, "id", None))
    parent_id = _id(getattr(thread, "parent_id", None)) or thread_id
    if not guild_id or not thread_id:
        return 0
    return await _sync_history_scope(
        store,
        thread,
        guild_id=guild_id,
        channel_id=thread_id,
        thread_id=thread_id,
        force=force,
    )


def _guild_sync_targets(guild: Any) -> list[Any]:
    targets: list[Any] = []
    seen: set[str] = set()
    for collection_name in ("text_channels", "channels", "threads"):
        for item in list(getattr(guild, collection_name, []) or []):
            item_id = _id(getattr(item, "id", None))
            if item_id and item_id not in seen and hasattr(item, "history"):
                targets.append(item)
                seen.add(item_id)
    active_threads = getattr(guild, "active_threads", None)
    if callable(active_threads):
        try:
            for thread in list(active_threads()) or []:
                thread_id = _id(getattr(thread, "id", None))
                if thread_id and thread_id not in seen and hasattr(thread, "history"):
                    targets.append(thread)
                    seen.add(thread_id)
        except Exception:
            logger.debug("Server archive active_threads lookup failed", exc_info=True)
    return targets


async def sync_guild_archive(
    store: ServerArchiveStore, guild: Any, *, force: bool = False
) -> int:
    guild_id = _id(getattr(guild, "id", None))
    if not guild_id:
        return 0

    state = await store.get_sync_state(guild_id=guild_id)
    if state and state.status == "running" and not force:
        logger.info(
            "Server archive guild sync already running",
            extra={
                "subsys": "server_archive",
                "event": "archive_guild_sync_running",
                "detail": {"guild_id": guild_id},
            },
        )
        return 0

    targets = _guild_sync_targets(guild)

    await store.set_sync_state(
        ArchiveSyncState(
            scope_key=f"{guild_id}::",
            guild_id=guild_id,
            last_synced_at=utc_now_iso(),
            status="running",
        )
    )

    processed = 0
    had_errors = False
    semaphore = asyncio.Semaphore(2)

    async def _run_target(target: Any) -> int:
        async with semaphore:
            try:
                return await sync_channel_archive(store, target, force=force)
            except Exception:
                logger.exception(
                    "Server archive target sync failed",
                    extra={
                        "subsys": "server_archive",
                        "event": "archive_target_sync_failed",
                        "detail": {
                            "guild_id": guild_id,
                            "target_id": _id(getattr(target, "id", None)),
                        },
                    },
                )
                return -1

    for target in targets:
        count = await _run_target(target)
        processed += max(0, count)
        target_id = _id(getattr(target, "id", None))
        target_state = await store.get_sync_state(
            guild_id=guild_id,
            channel_id=target_id,
            thread_id=target_id
            if getattr(target, "parent_id", None) is not None
            else None,
        )
        if count < 0 or (
            target_state is not None and target_state.status not in {"complete", "idle"}
        ):
            had_errors = True

    await store.set_sync_state(
        ArchiveSyncState(
            scope_key=f"{guild_id}::",
            guild_id=guild_id,
            last_synced_at=utc_now_iso(),
            status="complete_with_errors" if had_errors else "complete",
            error="permission_or_fetch_errors" if had_errors else None,
        )
    )
    return processed
