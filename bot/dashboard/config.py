"""Dashboard-specific configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from bot.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8011
DEFAULT_RATE_LIMIT_SENDS_PER_MINUTE = 5
DEFAULT_MAX_MESSAGE_CHARS = 1800
DEFAULT_DM_RETENTION_DAYS = 180
DEFAULT_AUDIT_RETENTION_DAYS = 180
DEFAULT_SESSION_TTL_HOURS = 8
DEFAULT_PAGE_SIZE = 50
DEFAULT_MAX_PAGE_SIZE = 200
DEFAULT_SUMMARY_TTL_SECONDS = 3
DEFAULT_STATIC_CACHE_SECONDS = 300
DEFAULT_RATE_LIMIT_BACKFILLS_PER_HOUR = 5
DEFAULT_MESSAGE_DB_PATH = "./data/dashboard_messages.db"
DEFAULT_BACKFILL_DB_PATH = "./data/dashboard_backfill.db"
DEFAULT_MESSAGE_RETENTION_DAYS = 180
DEFAULT_BACKFILL_MAX_MESSAGES_PER_CHANNEL = 500
DEFAULT_BACKFILL_MAX_CHANNELS_PER_RUN = 50
DEFAULT_BACKFILL_SLEEP_MS = 500
DEFAULT_MESSAGE_PAGE_SIZE = 50
DEFAULT_MESSAGE_PAGE_SIZE_MAX = 200


@dataclass(frozen=True)
class DashboardConfig:
    """Immutable dashboard configuration."""

    enabled: bool = False
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    public_bind: bool = False
    auth_token: Optional[str] = None
    session_secret: Optional[str] = None
    owner_ids: set[int] = field(default_factory=set)
    rate_limit_sends_per_minute: int = DEFAULT_RATE_LIMIT_SENDS_PER_MINUTE
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS
    dm_archive_enabled: bool = True
    dm_retention_days: int = DEFAULT_DM_RETENTION_DAYS
    audit_retention_days: int = DEFAULT_AUDIT_RETENTION_DAYS
    audit_db_path: str = "./data/dashboard_audit.db"
    session_ttl_hours: int = DEFAULT_SESSION_TTL_HOURS
    page_size: int = DEFAULT_PAGE_SIZE
    max_page_size: int = DEFAULT_MAX_PAGE_SIZE
    summary_ttl_seconds: int = DEFAULT_SUMMARY_TTL_SECONDS
    show_message_previews: bool = True
    # New config options
    static_cache_seconds: int = DEFAULT_STATIC_CACHE_SECONDS
    rate_limit_backfills_per_hour: int = DEFAULT_RATE_LIMIT_BACKFILLS_PER_HOUR
    message_db_path: str = DEFAULT_MESSAGE_DB_PATH
    backfill_db_path: str = DEFAULT_BACKFILL_DB_PATH
    guild_archive_enabled: bool = True
    message_retention_days: int = DEFAULT_MESSAGE_RETENTION_DAYS
    backfill_enabled: bool = True
    backfill_max_messages_per_channel: int = DEFAULT_BACKFILL_MAX_MESSAGES_PER_CHANNEL
    backfill_max_channels_per_run: int = DEFAULT_BACKFILL_MAX_CHANNELS_PER_RUN
    backfill_sleep_ms: int = DEFAULT_BACKFILL_SLEEP_MS
    redact_secrets: bool = True
    require_csrf: bool = True
    suppress_mentions: bool = True
    allow_everyone_mentions: bool = False
    message_page_size: int = DEFAULT_MESSAGE_PAGE_SIZE
    message_page_size_max: int = DEFAULT_MESSAGE_PAGE_SIZE_MAX

    def __post_init__(self) -> None:
        if self.public_bind and self.host == DEFAULT_HOST:
            logger.warning(
                "DASHBOARD_PUBLIC_BIND=true but DASHBOARD_HOST is still %s. Set DASHBOARD_HOST=0.0.0.0 explicitly for external access.",
                self.host,
            )


def _parse_bool_str(raw: Optional[str], default: bool) -> bool:
    if raw is None:
        return default
    s = str(raw).strip().lower()
    return s in {"1", "true", "yes", "on", "enabled", "enable"}


def _safe_int(raw: Optional[str], default: int, name: str) -> int:
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (ValueError, AttributeError):
        logger.warning("Invalid %s value '%s', using default %s", name, raw, default)
        return default


def _parse_ids(raw: Optional[str]) -> set[int]:
    if not raw:
        return set()
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            logger.warning("Invalid DASHBOARD_OWNER_IDS entry: %s", part)
    return ids


def load_dashboard_config() -> DashboardConfig:
    """Load dashboard configuration from environment variables."""
    enabled = _parse_bool_str(os.getenv("DASHBOARD_ENABLED"), False)
    host = os.getenv("DASHBOARD_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
    port = _safe_int(os.getenv("DASHBOARD_PORT"), DEFAULT_PORT, "DASHBOARD_PORT")
    public_bind = _parse_bool_str(os.getenv("DASHBOARD_PUBLIC_BIND"), False)
    auth_token = os.getenv("DASHBOARD_AUTH_TOKEN") or None
    session_secret = os.getenv("DASHBOARD_SESSION_SECRET") or None
    owner_ids = _parse_ids(os.getenv("DASHBOARD_OWNER_IDS"))

    # Merge DASHBOARD_OWNER_IDS with existing OWNER_IDS if available
    existing_owner_ids = _parse_ids(os.getenv("OWNER_IDS"))
    owner_ids = owner_ids | existing_owner_ids

    rate_limit = _safe_int(
        os.getenv("DASHBOARD_RATE_LIMIT_SENDS_PER_MINUTE"),
        DEFAULT_RATE_LIMIT_SENDS_PER_MINUTE,
        "DASHBOARD_RATE_LIMIT_SENDS_PER_MINUTE",
    )
    max_chars = _safe_int(
        os.getenv("DASHBOARD_MAX_MESSAGE_CHARS"),
        DEFAULT_MAX_MESSAGE_CHARS,
        "DASHBOARD_MAX_MESSAGE_CHARS",
    )
    dm_enabled = _parse_bool_str(os.getenv("DASHBOARD_DM_ARCHIVE_ENABLED"), True)
    dm_retention = _safe_int(
        os.getenv("DASHBOARD_DM_RETENTION_DAYS"),
        DEFAULT_DM_RETENTION_DAYS,
        "DASHBOARD_DM_RETENTION_DAYS",
    )
    audit_retention = _safe_int(
        os.getenv("DASHBOARD_AUDIT_RETENTION_DAYS"),
        DEFAULT_AUDIT_RETENTION_DAYS,
        "DASHBOARD_AUDIT_RETENTION_DAYS",
    )
    audit_db = os.getenv("DASHBOARD_AUDIT_DB_PATH", "./data/dashboard_audit.db").strip()
    session_ttl = _safe_int(
        os.getenv("DASHBOARD_SESSION_TTL_HOURS"),
        DEFAULT_SESSION_TTL_HOURS,
        "DASHBOARD_SESSION_TTL_HOURS",
    )
    show_previews = _parse_bool_str(os.getenv("DASHBOARD_SHOW_MESSAGE_PREVIEWS"), True)

    # New config values
    static_cache = _safe_int(
        os.getenv("DASHBOARD_STATIC_CACHE_SECONDS"),
        DEFAULT_STATIC_CACHE_SECONDS,
        "DASHBOARD_STATIC_CACHE_SECONDS",
    )
    rate_limit_backfills = _safe_int(
        os.getenv("DASHBOARD_RATE_LIMIT_BACKFILLS_PER_HOUR"),
        DEFAULT_RATE_LIMIT_BACKFILLS_PER_HOUR,
        "DASHBOARD_RATE_LIMIT_BACKFILLS_PER_HOUR",
    )
    message_db = os.getenv("DASHBOARD_MESSAGE_DB_PATH", DEFAULT_MESSAGE_DB_PATH).strip()
    backfill_db = os.getenv("DASHBOARD_BACKFILL_DB_PATH", DEFAULT_BACKFILL_DB_PATH).strip()
    guild_archive = _parse_bool_str(os.getenv("DASHBOARD_GUILD_ARCHIVE_ENABLED"), True)
    message_retention = _safe_int(
        os.getenv("DASHBOARD_MESSAGE_RETENTION_DAYS"),
        DEFAULT_MESSAGE_RETENTION_DAYS,
        "DASHBOARD_MESSAGE_RETENTION_DAYS",
    )
    backfill_enabled = _parse_bool_str(os.getenv("DASHBOARD_BACKFILL_ENABLED"), True)
    backfill_max_msgs = _safe_int(
        os.getenv("DASHBOARD_BACKFILL_MAX_MESSAGES_PER_CHANNEL"),
        DEFAULT_BACKFILL_MAX_MESSAGES_PER_CHANNEL,
        "DASHBOARD_BACKFILL_MAX_MESSAGES_PER_CHANNEL",
    )
    backfill_max_chs = _safe_int(
        os.getenv("DASHBOARD_BACKFILL_MAX_CHANNELS_PER_RUN"),
        DEFAULT_BACKFILL_MAX_CHANNELS_PER_RUN,
        "DASHBOARD_BACKFILL_MAX_CHANNELS_PER_RUN",
    )
    backfill_sleep = _safe_int(
        os.getenv("DASHBOARD_BACKFILL_SLEEP_MS"),
        DEFAULT_BACKFILL_SLEEP_MS,
        "DASHBOARD_BACKFILL_SLEEP_MS",
    )
    redact_secrets = _parse_bool_str(os.getenv("DASHBOARD_REDACT_SECRETS"), True)
    require_csrf = _parse_bool_str(os.getenv("DASHBOARD_REQUIRE_CSRF"), True)
    suppress_mentions = _parse_bool_str(os.getenv("DASHBOARD_SUPPRESS_MENTIONS"), True)
    allow_everyone = _parse_bool_str(os.getenv("DASHBOARD_ALLOW_EVERYONE_MENTIONS"), False)
    msg_page_size = _safe_int(
        os.getenv("DASHBOARD_MESSAGE_PAGE_SIZE"),
        DEFAULT_MESSAGE_PAGE_SIZE,
        "DASHBOARD_MESSAGE_PAGE_SIZE",
    )
    msg_page_size_max = _safe_int(
        os.getenv("DASHBOARD_MESSAGE_PAGE_SIZE_MAX"),
        DEFAULT_MESSAGE_PAGE_SIZE_MAX,
        "DASHBOARD_MESSAGE_PAGE_SIZE_MAX",
    )

    # Validate: if enabled, require auth_token
    if enabled and not auth_token:
        logger.error("DASHBOARD_ENABLED=true but DASHBOARD_AUTH_TOKEN is not set. Dashboard will NOT start. Set DASHBOARD_AUTH_TOKEN to a strong random value.")
        enabled = False

    if enabled and not session_secret:
        # Generate a one-time session secret, printed to console only
        import secrets

        session_secret = secrets.token_hex(32)
        logger.warning(
            "DASHBOARD_SESSION_SECRET not set. Generated a one-time session secret. Set DASHBOARD_SESSION_SECRET in .env for persistent sessions. Secret: %s",
            session_secret,
        )

    cfg = DashboardConfig(
        enabled=enabled,
        host=host,
        port=port,
        public_bind=public_bind,
        auth_token=auth_token,
        session_secret=session_secret,
        owner_ids=owner_ids,
        rate_limit_sends_per_minute=rate_limit,
        max_message_chars=max_chars,
        dm_archive_enabled=dm_enabled,
        dm_retention_days=dm_retention,
        audit_retention_days=audit_retention,
        audit_db_path=audit_db,
        session_ttl_hours=session_ttl,
        show_message_previews=show_previews,
        # New config values
        static_cache_seconds=static_cache,
        rate_limit_backfills_per_hour=rate_limit_backfills,
        message_db_path=message_db,
        backfill_db_path=backfill_db,
        guild_archive_enabled=guild_archive,
        message_retention_days=message_retention,
        backfill_enabled=backfill_enabled,
        backfill_max_messages_per_channel=backfill_max_msgs,
        backfill_max_channels_per_run=backfill_max_chs,
        backfill_sleep_ms=backfill_sleep,
        redact_secrets=redact_secrets,
        require_csrf=require_csrf,
        suppress_mentions=suppress_mentions,
        allow_everyone_mentions=allow_everyone,
        message_page_size=msg_page_size,
        message_page_size_max=msg_page_size_max,
    )

    if enabled:
        bind_addr = host if public_bind else DEFAULT_HOST
        logger.info(
            "Dashboard enabled: http://%s:%d (public_bind=%s, owner_ids=%s)",
            bind_addr,
            port,
            public_bind,
            sorted(owner_ids),
        )

    return cfg
