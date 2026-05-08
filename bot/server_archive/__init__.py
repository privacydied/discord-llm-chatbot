"""Raw guild message archive: SQLite storage, FTS search, and sync helpers."""

from .models import (
    ArchiveAttachment,
    ArchiveChannel,
    ArchiveGuild,
    ArchiveMention,
    ArchiveMessage,
    ArchiveMessageBundle,
    ArchiveSearchResult,
    ArchiveSyncState,
    ArchiveThread,
    ArchiveUser,
)
from .service import (
    ServerArchiveService,
    enqueue_live_message,
    get_server_archive_service,
    get_server_archive_status,
    search_archive,
    start_server_archive_service,
    stop_server_archive_service,
)

__all__ = [
    "ArchiveAttachment",
    "ArchiveChannel",
    "ArchiveGuild",
    "ArchiveMention",
    "ArchiveMessage",
    "ArchiveMessageBundle",
    "ArchiveSearchResult",
    "ArchiveSyncState",
    "ArchiveThread",
    "ArchiveUser",
    "ServerArchiveService",
    "enqueue_live_message",
    "get_server_archive_service",
    "get_server_archive_status",
    "search_archive",
    "start_server_archive_service",
    "stop_server_archive_service",
]
