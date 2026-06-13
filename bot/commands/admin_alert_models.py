"""Data models for the admin DM alert system."""

from dataclasses import dataclass, field
from enum import Enum


class AlertSessionStatus(Enum):
    COMPOSING = "composing"
    READY = "ready"
    POSTING = "posting"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class AlertDestination:
    guild_id: int | None
    channel_id: int | None
    channel_name: str | None
    guild_name: str | None = None
    permissions_valid: bool = True
    permission_issues: list[str] = field(default_factory=list)


@dataclass
class AlertSession:
    user_id: int
    session_id: str
    status: AlertSessionStatus
    created_at: float
    expires_at: float
    content: str = ""
    embed_title: str = ""
    embed_description: str = ""
    destinations: list[AlertDestination] = field(default_factory=list)
    mention_everyone: bool = False
    current_step: str = "select_channels"
    composer_message_id: int | None = None
    composer_ready: bool = False
    # Guild navigation pagination
    guild_page: int = 0
    selected_guild_id: int | None = None
    channel_page: int = 0
    guilds_list: list = field(default_factory=list)
    selection_message_id: int | None = None
    channel_message_id: int | None = None
