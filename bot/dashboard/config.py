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
DEFAULT_DM_RETENTION_DAYS = 90
DEFAULT_AUDIT_RETENTION_DAYS = 180
DEFAULT_SESSION_TTL_HOURS = 8
DEFAULT_PAGE_SIZE = 50
DEFAULT_MAX_PAGE_SIZE = 200
DEFAULT_SUMMARY_TTL_SECONDS = 3


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
