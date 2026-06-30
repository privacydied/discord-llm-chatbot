"""Background task management for the Discord bot."""

import asyncio
import logging
from typing import Any

from discord.ext import commands, tasks

from bot.server_archive import (
    start_server_archive_service as start_native_server_archive_service,
)
from bot.server_archive import (
    stop_server_archive_service as stop_native_server_archive_service,
)

from .config import load_config
from .janitor import start_janitor, stop_janitor
from .memory import (
    save_all_profiles,
    save_all_server_profiles,
    start_memory_distiller,
    start_memory_service,
    stop_memory_distiller,
    stop_memory_service,
)

logger = logging.getLogger(__name__)
# Global task registry
_background_tasks: dict[str, tasks.Loop] = {}
_running_tasks: list[asyncio.Task] = []


def _reclaim_memory(cfg: dict[str, Any]) -> float:
    """Best-effort memory reclaim when the health check sees high RSS. [PA][REH]

    Three steps, cheapest/safest first: collect Python-level cycles, evict
    non-default STT whisper models (each pins tens-to-hundreds of MB of
    weights -- see bot/stt.py's LRU cache), then ask glibc to release freed
    arena pages back to the OS via malloc_trim(). Returns MB reclaimed
    (0 if measurement or trimming isn't available); never raises.
    """
    import gc

    import psutil

    process = psutil.Process()
    before_mb = process.memory_info().rss / 1024 / 1024
    gc.collect()

    if cfg.get("MEMORY_EVICT_STT_CACHE_ON_WARNING", True):
        try:
            from .stt import get_stt_manager_if_initialized

            manager = get_stt_manager_if_initialized()
            if manager is not None:
                manager.evict_idle_models()
        except Exception as exc:
            logger.debug(f"STT cache eviction during memory reclaim skipped: {exc}")

    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass

    after_mb = process.memory_info().rss / 1024 / 1024
    return max(0.0, before_mb - after_mb)


def _persist_profiles_sync() -> tuple[bool, bool]:
    """Persist profile caches to disk."""
    return save_all_profiles(), save_all_server_profiles()


async def _persist_profiles_nonblocking() -> tuple[bool, bool]:
    """Persist profile caches without blocking the event loop."""
    return await asyncio.to_thread(_persist_profiles_sync)


def setup_memory_save_task(bot: commands.Bot) -> tasks.Loop:
    """Set up a task to periodically save memory profiles.

    Args:
        bot: The bot instance

    Returns:
        The task loop that can be started/stopped

    """
    initial_config = load_config()
    interval_minutes = initial_config.get("PROFILE_AUTOSAVE_INTERVAL", 10)

    @tasks.loop(minutes=interval_minutes)
    async def memory_save_task() -> None:
        try:
            current_config = load_config()
            target_interval = current_config.get("PROFILE_AUTOSAVE_INTERVAL", 10)
            if target_interval != memory_save_task.current_interval:
                memory_save_task.change_interval(minutes=target_interval)
                memory_save_task.current_interval = target_interval
                logger.debug(
                    "Updated memory save interval",
                    extra={
                        "subsys": "memory",
                        "event": "autosave_interval_update",
                        "detail": {"minutes": target_interval},
                    },
                )

            user_ok, server_ok = await _persist_profiles_nonblocking()
            if not user_ok or not server_ok:
                logger.warning(
                    "Auto-save completed with failures",
                    extra={
                        "subsys": "memory",
                        "event": "autosave_partial_failure",
                        "detail": {"user_ok": user_ok, "server_ok": server_ok},
                    },
                )
            logger.debug(
                "Auto-saved all profiles",
                extra={"subsys": "memory", "event": "autosave"},
            )
        except Exception as e:
            logger.exception(
                f"Error during profile autosave: {e}",
                extra={"subsys": "memory", "event": "autosave_error"},
            )

    memory_save_task.current_interval = interval_minutes

    # Register the task
    _background_tasks["memory_save"] = memory_save_task

    return memory_save_task


class TaskManager:
    """Manages background tasks for the bot."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.tasks = {}
        self.running = False

    async def start_all_tasks(self) -> None:
        """Start all background tasks."""
        if self.running:
            logger.warning("Tasks are already running")
            return

        try:
            # Start profile auto-save task
            await self._start_profile_autosave()

            # Start cleanup tasks
            await self._start_cleanup_tasks()

            # Start health check task
            await self._start_health_check()

            # Start curated memory service
            await start_memory_service(self.bot)

            # Start memory distiller service
            await start_memory_distiller(self.bot)

            # Start server archive service
            await start_native_server_archive_service(self.bot)

            # Start janitor task
            await self._start_janitor()

            self.running = True
            logger.info("All background tasks started successfully")

        except Exception as e:
            logger.error(f"Error starting background tasks: {e}", exc_info=True)
            raise

    async def stop_all_tasks(self) -> None:
        """Stop all background tasks."""
        if not self.running:
            return

        logger.info("Stopping all background tasks...")

        # Stop janitor first
        try:
            await stop_janitor()
        except Exception as e:
            logger.warning(f"Error stopping janitor: {e}")

        # Stop curated memory service
        try:
            await stop_memory_service()
        except Exception as e:
            logger.warning(f"Error stopping curated memory service: {e}")

        # Stop memory distiller service
        try:
            await stop_memory_distiller()
        except Exception as e:
            logger.warning(f"Error stopping memory distiller service: {e}")

        # Stop server archive service
        try:
            await stop_native_server_archive_service()
        except Exception as e:
            logger.warning(f"Error stopping server archive service: {e}")
        for task_name, task in self.tasks.items():
            try:
                task.cancel()
                logger.debug(f"Cancelled task: {task_name}")
            except Exception as e:
                logger.warning(f"Error cancelling task {task_name}: {e}")

        # Wait for tasks to complete
        if _running_tasks:
            await asyncio.gather(*_running_tasks, return_exceptions=True)

        self.running = False
        logger.info("All background tasks stopped")

    async def _start_profile_autosave(self) -> None:
        """Start the profile auto-save task."""
        cfg = load_config()
        interval_minutes = cfg.get("PROFILE_AUTOSAVE_INTERVAL", 10)

        @tasks.loop(minutes=interval_minutes)
        async def profile_autosave() -> None:
            """Automatically save user and server profiles."""
            try:
                current_cfg = load_config()
                target_minutes = current_cfg.get("PROFILE_AUTOSAVE_INTERVAL", 10)
                if target_minutes != profile_autosave.current_interval:
                    profile_autosave.change_interval(minutes=target_minutes)
                    profile_autosave.current_interval = target_minutes
                    logger.debug(
                        "Updated profile autosave interval",
                        extra={
                            "subsys": "memory",
                            "event": "autosave_interval_update",
                            "detail": {"minutes": target_minutes},
                        },
                    )

                user_ok, server_ok = await _persist_profiles_nonblocking()
                if not user_ok or not server_ok:
                    logger.warning(
                        "Auto-save completed with failures",
                        extra={
                            "subsys": "memory",
                            "event": "autosave_partial_failure",
                            "detail": {"user_ok": user_ok, "server_ok": server_ok},
                        },
                    )
                logger.debug("Auto-saved all profiles")
            except Exception as e:
                logger.error(f"Error during profile autosave: {e}", exc_info=True)

        profile_autosave.current_interval = interval_minutes
        profile_autosave.start()
        self.tasks["profile_autosave"] = profile_autosave
        logger.info("Profile auto-save task started")

    async def _start_cleanup_tasks(self) -> None:
        """Start cleanup tasks."""
        cfg = load_config()
        interval_hours = cfg.get("CLEANUP_INTERVAL_HOURS", 24)

        @tasks.loop(hours=interval_hours)
        async def cleanup_old_logs() -> None:
            """Clean up old log files."""
            try:
                import time
                from pathlib import Path

                current_cfg = load_config()
                target_hours = current_cfg.get("CLEANUP_INTERVAL_HOURS", 24)
                if target_hours != cleanup_old_logs.current_interval:
                    cleanup_old_logs.change_interval(hours=target_hours)
                    cleanup_old_logs.current_interval = target_hours
                    logger.debug(
                        "Updated cleanup interval",
                        extra={
                            "subsys": "maintenance",
                            "event": "cleanup_interval_update",
                            "detail": {"hours": target_hours},
                        },
                    )

                logs_dir = current_cfg.get("USER_LOGS_DIR")
                if not logs_dir or not Path(logs_dir).exists():
                    return

                # Clean up files older than 30 days
                cutoff_time = time.time() - (30 * 24 * 60 * 60)

                for log_file in Path(logs_dir).rglob("*.log"):
                    try:
                        if log_file.stat().st_mtime < cutoff_time:
                            log_file.unlink()
                            logger.debug(f"Deleted old log file: {log_file}")
                    except Exception as e:
                        logger.warning(f"Error deleting log file {log_file}: {e}")

                logger.info("Log cleanup completed")

            except Exception as e:
                logger.error(f"Error during log cleanup: {e}", exc_info=True)

        cleanup_old_logs.current_interval = interval_hours
        cleanup_old_logs.start()
        self.tasks["cleanup_old_logs"] = cleanup_old_logs
        logger.info("Cleanup tasks started")

    async def _start_health_check(self) -> None:
        """Start health check task."""
        cfg = load_config()
        interval_minutes = cfg.get("HEALTH_CHECK_INTERVAL", 5)

        @tasks.loop(minutes=interval_minutes)
        async def health_check() -> None:
            """Perform health checks."""
            try:
                current_cfg = load_config()
                target_minutes = current_cfg.get("HEALTH_CHECK_INTERVAL", 5)
                if target_minutes != health_check.current_interval:
                    health_check.change_interval(minutes=target_minutes)
                    health_check.current_interval = target_minutes
                    logger.debug(
                        "Updated health check interval",
                        extra={
                            "subsys": "healthcheck",
                            "event": "health_interval_update",
                            "detail": {"minutes": target_minutes},
                        },
                    )

                # Check bot connection
                if not self.bot.is_ready():
                    logger.warning("Bot is not ready")
                    return

                # Check guild count
                guild_count = len(self.bot.guilds)
                logger.debug(f"Health check: Connected to {guild_count} guilds")

                # Check memory usage
                import psutil

                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024

                warning_threshold = current_cfg.get("MEMORY_WARNING_THRESHOLD", 1200)
                if memory_mb > warning_threshold:
                    logger.warning(f"High memory usage: {memory_mb:.1f}MB")
                    reclaimed_mb = _reclaim_memory(current_cfg)
                    if reclaimed_mb > 0:
                        logger.info(
                            "healthcheck.memory_reclaim | reclaimed_mb=%.1f",
                            reclaimed_mb,
                            extra={
                                "subsys": "healthcheck",
                                "event": "healthcheck.memory_reclaim",
                                "detail": {"reclaimed_mb": reclaimed_mb},
                            },
                        )

                # Update bot status if needed
                if guild_count == 0:
                    logger.warning("Bot is not connected to any guilds")

            except Exception as e:
                logger.error(f"Error during health check: {e}", exc_info=True)

        health_check.current_interval = interval_minutes
        health_check.start()
        self.tasks["health_check"] = health_check
        logger.info("Health check task started")

    async def _start_janitor(self) -> None:
        """Start the cache and log janitor task."""
        try:
            await start_janitor()
            logger.info("Janitor task started")
        except Exception as e:
            logger.error(f"Error starting janitor: {e}", exc_info=True)


# Global task manager instance
_task_manager: TaskManager | None = None


async def spawn_background_tasks(bot: commands.Bot) -> None:
    """Initialize and start all background tasks."""
    global _task_manager

    if _task_manager is not None:
        logger.warning("Background tasks already initialized")
        return

    try:
        _task_manager = TaskManager(bot)
        await _task_manager.start_all_tasks()
        logger.info("Background tasks spawned successfully")

    except Exception as e:
        logger.error(f"Failed to spawn background tasks: {e}", exc_info=True)
        raise


async def stop_background_tasks() -> None:
    """Stop all background tasks."""
    global _task_manager

    if _task_manager is None:
        logger.warning("No background tasks to stop")
        return

    try:
        await _task_manager.stop_all_tasks()
        _task_manager = None
        logger.info("Background tasks stopped successfully")

    except Exception as e:
        logger.error(f"Error stopping background tasks: {e}", exc_info=True)


def get_task_status() -> dict[str, Any]:
    """Get the status of all background tasks."""
    if _task_manager is None:
        return {"status": "not_initialized", "tasks": {}}

    task_status = {}
    for name, task in _task_manager.tasks.items():
        task_status[name] = {
            "running": not task.is_being_cancelled(),
            "failed": task.failed(),
            "next_iteration": task.next_iteration.isoformat() if task.next_iteration else None,
        }

    return {
        "status": "running" if _task_manager.running else "stopped",
        "tasks": task_status,
    }
