"""Dashboard aiohttp server with lifecycle management."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

from aiohttp import web

from bot.utils.logging import get_logger

from .auth import SessionStore
from .config import DashboardConfig

if TYPE_CHECKING:
    from .audit_store import AuditStore
    from .dm_store import DMStore
    from .services import DashboardServices

logger = get_logger(__name__)


class DashboardServer:
    """Manages the aiohttp dashboard server lifecycle."""

    def __init__(
        self,
        config: DashboardConfig,
        services: "DashboardServices",
        audit_store: "AuditStore",
        dm_store: "DMStore",
    ) -> None:
        self._config = config
        self._services = services
        self._audit_store = audit_store
        self._dm_store = dm_store
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._cleanup_task: Optional[asyncio.Task] = None

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
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
            self._app = None
            logger.info("Dashboard server stopped")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired sessions and old audit/DM records."""
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
                except Exception as e:
                    logger.warning("Audit retention cleanup failed: %s", e)

                try:
                    dm_deleted = await self._dm_store.cleanup_retention()
                    if dm_deleted:
                        logger.info("DM retention cleanup: %d records removed", dm_deleted)
                except Exception as e:
                    logger.warning("DM retention cleanup failed: %s", e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Dashboard cleanup loop error: %s", e)

    @property
    def is_running(self) -> bool:
        return self._runner is not None

    @property
    def app(self) -> Optional[web.Application]:
        return self._app
