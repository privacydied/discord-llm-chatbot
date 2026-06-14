"""Dashboard aiohttp server with lifecycle management."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from aiohttp import web

from bot.utils.logging import get_logger

from .auth import SessionStore

if TYPE_CHECKING:
    from .audit_store import AuditStore
    from .backfill import BackfillJobStore, BackfillService
    from .config import DashboardConfig
    from .dm_store import DMStore
    from .message_store import MessageStore
    from .services import DashboardServices

logger = get_logger(__name__)


class DashboardServer:
    """Manages the aiohttp dashboard server lifecycle."""

    def __init__(
        self,
        config: DashboardConfig,
        services: DashboardServices,
        audit_store: AuditStore,
        dm_store: DMStore,
        message_store: MessageStore,
        backfill_store: BackfillJobStore,
        backfill_service: BackfillService,
    ) -> None:
        self._config = config
        self._services = services
        self._audit_store = audit_store
        self._dm_store = dm_store
        self._message_store = message_store
        self._backfill_store = backfill_store
        self._backfill_service = backfill_service
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the dashboard aiohttp server."""
        if not self._config.enabled:
            logger.info("Dashboard is disabled, not starting server")
            return

        # Build host for binding
        host = self._config.host if self._config.public_bind else "127.0.0.1"
        port = self._config.port

        # Create aiohttp app
        app = web.Application()
        app["dashboard_config"] = self._config
        app["session_store"] = SessionStore(session_ttl_hours=self._config.session_ttl_hours)
        app["audit_store"] = self._audit_store
        app["dm_store"] = self._dm_store
        app["message_store"] = self._message_store
        app["backfill_store"] = self._backfill_store
        app["backfill_service"] = self._backfill_service

        # Register routes
        from .routes import setup_routes

        setup_routes(app, self._services)

        # Create runner and site
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        self._app = app
        self._runner = runner
        self._site = site

        logger.info("Dashboard server started on %s:%d", host, port)

        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop the dashboard server gracefully."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

        # Clean up any running backfill jobs
        try:
            stale = await self._backfill_store.reset_stale_jobs()
            if stale:
                logger.info("Reset %d stale backfill jobs on shutdown", stale)
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.warning("Backfill cleanup on shutdown failed: %s", e)

        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
            self._app = None
            logger.info("Dashboard server stopped")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired sessions and old records."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                if self._app is None:
                    break
                session_store: SessionStore = self._app["session_store"]
                removed = session_store.cleanup()
                if removed:
                    logger.debug("Cleaned up %d expired sessions", removed)

                try:
                    audit_deleted = await self._audit_store.cleanup_retention()
                    if audit_deleted:
                        logger.info("Audit retention cleanup: %d records removed", audit_deleted)
                except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                    logger.warning("Audit retention cleanup failed: %s", e)

                try:
                    dm_deleted = await self._dm_store.cleanup_retention()
                    if dm_deleted:
                        logger.info("DM retention cleanup: %d records removed", dm_deleted)
                except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                    logger.warning("DM retention cleanup failed: %s", e)

                try:
                    msg_deleted = await self._message_store.cleanup_retention()
                    if msg_deleted:
                        logger.info("Message store retention cleanup: %d records removed", msg_deleted)
                except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                    logger.warning("Message store retention cleanup failed: %s", e)

            except asyncio.CancelledError:
                break
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                logger.warning("Dashboard cleanup loop error: %s", e)

    @property
    def is_running(self) -> bool:
        return self._runner is not None

    @property
    def app(self) -> web.Application | None:
        return self._app

    async def hot_reload_config(self, new_config: DashboardConfig) -> None:
        """Apply updated dashboard config without restarting the server.

        Mutable at runtime: auth_token, owner_ids, rate_limit, retention_days,
        max_message_chars, show_message_previews, dm_archive_enabled,
        session_ttl_hours.

        NOT mutable: host/port/public_bind (require restart), enabled (requires
        restart), audit_db_path (requires restart).
        """
        old = self._config

        # Update config reference on server and services
        self._config = new_config
        self._services._config = new_config

        # Update rate limiter
        self._services._rate_limiter._sends_per_minute = new_config.rate_limit_sends_per_minute

        # Update app state
        if self._app is not None:
            self._app["dashboard_config"] = new_config

            # Replace session store with new TTL
            old_store = self._app.get("session_store")
            if old_store and old_store._session_ttl_hours != new_config.session_ttl_hours:
                from .auth import SessionStore

                new_store = SessionStore(session_ttl_hours=new_config.session_ttl_hours)
                # Migrate valid sessions from old store
                for sid, sess in old_store._sessions.items():
                    new_store._sessions[sid] = sess
                self._app["session_store"] = new_store

        # Update retention on stores
        self._audit_store._retention_days = new_config.audit_retention_days
        self._dm_store._retention_days = new_config.dm_retention_days
        self._message_store._retention_days = new_config.message_retention_days

        # DM archive toggle
        if not new_config.dm_archive_enabled:
            self._dm_store._initialized = False
        elif not getattr(self._dm_store, "_initialized", False):
            await self._dm_store.initialize()

        # Log what changed
        changed = []
        if old.auth_token != new_config.auth_token:
            changed.append("auth_token=rotated")
        if old.owner_ids != new_config.owner_ids:
            changed.append(f"owner_ids={len(new_config.owner_ids)}")
        if old.rate_limit_sends_per_minute != new_config.rate_limit_sends_per_minute:
            changed.append(f"rate_limit={new_config.rate_limit_sends_per_minute}/min")
        if old.max_message_chars != new_config.max_message_chars:
            changed.append(f"max_chars={new_config.max_message_chars}")
        if old.show_message_previews != new_config.show_message_previews:
            changed.append(f"previews={new_config.show_message_previews}")
        if old.session_ttl_hours != new_config.session_ttl_hours:
            changed.append(f"session_ttl={new_config.session_ttl_hours}h")
        if old.audit_retention_days != new_config.audit_retention_days:
            changed.append(f"audit_retention={new_config.audit_retention_days}d")
        if old.dm_retention_days != new_config.dm_retention_days:
            changed.append(f"dm_retention={new_config.dm_retention_days}d")
        if old.message_retention_days != new_config.message_retention_days:
            changed.append(f"message_retention={new_config.message_retention_days}d")

        if changed:
            logger.info("Dashboard config hot-reloaded: %s", ", ".join(changed))

            await self._audit_store.record(
                event_type="dashboard.config.reload",
                result="success",
                metadata={"changes": changed},
            )
