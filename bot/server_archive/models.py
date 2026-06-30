"""Data models for the raw server archive."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    return str(value)


@dataclass(slots=True)
class ArchiveGuild:
    guild_id: str
    name: str
    icon_url: str | None = None
    first_seen_at: str = field(default_factory=utc_now_iso)
    last_seen_at: str = field(default_factory=utc_now_iso)

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArchiveChannel:
    channel_id: str
    guild_id: str
    parent_id: str | None = None
    name: str = ""
    type: str = "text"
    archived_at: str = field(default_factory=utc_now_iso)
    last_synced_message_id: str | None = None
    last_synced_at: str | None = None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArchiveThread:
    thread_id: str
    guild_id: str
    parent_channel_id: str
    name: str = ""
    archived_at: str = field(default_factory=utc_now_iso)
    last_synced_message_id: str | None = None
    last_synced_at: str | None = None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArchiveUser:
    user_id: str
    username: str
    global_name: str | None = None
    display_name: str | None = None
    bot: int = 0
    last_seen_at: str = field(default_factory=utc_now_iso)
    avatar: str | None = None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArchiveAttachment:
    attachment_id: str
    message_id: str
    filename: str | None = None
    content_type: str | None = None
    size: int | None = None
    url: str = ""
    proxy_url: str | None = None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArchiveMessage:
    message_id: str
    guild_id: str
    channel_id: str
    thread_id: str | None
    author_id: str
    content: str = ""
    clean_content: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    edited_at: str | None = None
    deleted_at: str | None = None
    jump_url: str | None = None
    reply_to_message_id: str | None = None
    has_attachments: int = 0
    has_embeds: int = 0
    metadata_json: str = "{}"

    def to_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["has_attachments"] = int(bool(self.has_attachments))
        payload["has_embeds"] = int(bool(self.has_embeds))
        return payload


@dataclass(slots=True)
class ArchiveMention:
    message_id: str
    mentioned_user_id: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArchiveMessageBundle:
    guild: ArchiveGuild
    channel: ArchiveChannel
    author: ArchiveUser
    message: ArchiveMessage
    thread: ArchiveThread | None = None
    attachments: tuple[ArchiveAttachment, ...] = ()
    mentions: tuple[ArchiveMention, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "guild": self.guild.to_row(),
            "channel": self.channel.to_row(),
            "author": self.author.to_row(),
            "message": self.message.to_row(),
            "thread": self.thread.to_row() if self.thread else None,
            "attachments": [item.to_row() for item in self.attachments],
            "mentions": [item.to_row() for item in self.mentions],
        }


@dataclass(slots=True)
class ArchiveSyncState:
    scope_key: str
    guild_id: str
    channel_id: str | None = None
    thread_id: str | None = None
    last_message_id: str | None = None
    last_synced_at: str | None = None
    status: str = "idle"
    error: str | None = None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> ArchiveSyncState:
        data = dict(row)
        return cls(
            scope_key=str(data["scope_key"]),
            guild_id=str(data["guild_id"]),
            channel_id=data.get("channel_id"),
            thread_id=data.get("thread_id"),
            last_message_id=data.get("last_message_id"),
            last_synced_at=data.get("last_synced_at"),
            status=str(data.get("status") or "idle"),
            error=data.get("error"),
        )


@dataclass(slots=True)
class ArchiveSearchResult:
    message_id: str
    guild_id: str
    channel_id: str
    thread_id: str | None
    author_id: str
    author_name: str | None
    channel_name: str | None
    content: str
    clean_content: str
    snippet: str
    created_at: str
    edited_at: str | None
    jump_url: str | None
    score: float

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> ArchiveSearchResult:
        data = dict(row)
        return cls(
            message_id=str(data["message_id"]),
            guild_id=str(data["guild_id"]),
            channel_id=str(data["channel_id"]),
            thread_id=data.get("thread_id"),
            author_id=str(data["author_id"]),
            author_name=data.get("author_name"),
            channel_name=data.get("channel_name"),
            content=str(data.get("content") or ""),
            clean_content=str(data.get("clean_content") or ""),
            snippet=str(data.get("snippet") or ""),
            created_at=str(data.get("created_at") or ""),
            edited_at=data.get("edited_at"),
            jump_url=data.get("jump_url"),
            score=float(data.get("score") or 0.0),
        )
