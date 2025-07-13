"""
Background task management for the Discord bot.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta

from discord.ext import commands, tasks

from .memory import save_all_profiles, save_all_server_profiles
from .config import load_config

logger = logging.getLogger(__name__)
config = load_config()

# Global task registry
_background_tasks: Dict[str, tasks.Loop] = {}
_running_tasks: List[asyncio.Task] = []


class TaskManager:
    """Manages background tasks for the bot."""
    
    def __init__(self, bot: commands.Bot):
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
        
        # Cancel all registered tasks
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
        @tasks.loop(minutes=config.get("PROFILE_AUTOSAVE_INTERVAL", 10))
        async def profile_autosave():
            """Automatically save user and server profiles."""
            try:
                save_all_profiles()
                save_all_server_profiles()
                logger.debug("Auto-saved all profiles")
            except Exception as e:
                logger.error(f"Error during profile autosave: {e}", exc_info=True)
        
        profile_autosave.start()
        self.tasks["profile_autosave"] = profile_autosave
        logger.info("Profile auto-save task started")
    
    async def _start_cleanup_tasks(self) -> None:
        """Start cleanup tasks."""
        @tasks.loop(hours=config.get("CLEANUP_INTERVAL_HOURS", 24))
        async def cleanup_old_logs():
            """Clean up old log files."""
            try:
                from pathlib import Path
                import time
                
                logs_dir = config.get("USER_LOGS_DIR")
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
        
        cleanup_old_logs.start()
        self.tasks["cleanup_old_logs"] = cleanup_old_logs
        logger.info("Cleanup tasks started")
    
    async def _start_health_check(self) -> None:
        """Start health check task."""
        @tasks.loop(minutes=config.get("HEALTH_CHECK_INTERVAL", 5))
        async def health_check():
            """Perform health checks."""
            try:
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
                
                if memory_mb > config.get("MEMORY_WARNING_THRESHOLD", 500):
                    logger.warning(f"High memory usage: {memory_mb:.1f}MB")
                
                # Update bot status if needed
                if guild_count == 0:
                    logger.warning("Bot is not connected to any guilds")
                
            except Exception as e:
                logger.error(f"Error during health check: {e}", exc_info=True)
        
        health_check.start()
        self.tasks["health_check"] = health_check
        logger.info("Health check task started")


# Global task manager instance
_task_manager: Optional[TaskManager] = None


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


def get_task_status() -> Dict[str, Any]:
    """Get the status of all background tasks."""
    if _task_manager is None:
        return {"status": "not_initialized", "tasks": {}}
    
    task_status = {}
    for name, task in _task_manager.tasks.items():
        task_status[name] = {
            "running": not task.is_being_cancelled(),
            "failed": task.failed(),
            "next_iteration": task.next_iteration.isoformat() if task.next_iteration else None
        }
    
    return {
        "status": "running" if _task_manager.running else "stopped",
        "tasks": task_status
    }
