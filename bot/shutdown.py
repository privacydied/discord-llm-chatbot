"""
Graceful shutdown handling for the Discord bot.
"""
import asyncio
import logging
import signal
import sys
from typing import Callable, Optional, Any

from discord.ext import commands

from .memory import save_all_profiles, save_all_server_profiles
from .tts import cleanup_tts
from .ollama import ollama_client
from .tasks import stop_background_tasks

logger = logging.getLogger(__name__)

# Global shutdown state
_shutdown_in_progress = False
_shutdown_timeout = 30  # seconds


class GracefulShutdown:
    """Handles graceful shutdown of the bot."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.shutdown_tasks = []
        self.shutdown_in_progress = False
    
    def add_shutdown_task(self, task: Callable, description: str = "") -> None:
        """Add a task to be executed during shutdown."""
        self.shutdown_tasks.append({
            'task': task,
            'description': description or task.__name__
        })
    
    async def execute_shutdown(self, signal_num: Optional[int] = None) -> None:
        """Execute graceful shutdown sequence."""
        if self.shutdown_in_progress:
            logger.warning("Shutdown already in progress")
            return
        
        self.shutdown_in_progress = True
        
        if signal_num:
            logger.info(f"Received signal {signal_num}, initiating graceful shutdown...")
        else:
            logger.info("Initiating graceful shutdown...")
        
        try:
            # Execute shutdown tasks with timeout
            await asyncio.wait_for(
                self._execute_shutdown_tasks(),
                timeout=_shutdown_timeout
            )
            
            logger.info("Graceful shutdown completed successfully")
            
        except asyncio.TimeoutError:
            logger.error(f"Shutdown timed out after {_shutdown_timeout} seconds")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
        finally:
            # Force exit if needed
            if self.bot.is_closed():
                logger.info("Bot connection already closed")
            else:
                logger.info("Closing bot connection...")
                await self.bot.close()
    
    async def _execute_shutdown_tasks(self) -> None:
        """Execute all registered shutdown tasks."""
        logger.info("Executing shutdown tasks...")
        
        # Save all profiles
        await self._save_all_data()
        
        # Stop background tasks
        await self._stop_background_tasks()
        
        # Clean up TTS resources
        await self._cleanup_tts()
        
        # Close external clients
        await self._cleanup_external_clients()
        
        # Execute custom shutdown tasks
        for task_info in self.shutdown_tasks:
            try:
                task = task_info['task']
                description = task_info['description']
                
                logger.debug(f"Executing shutdown task: {description}")
                
                if asyncio.iscoroutinefunction(task):
                    await task()
                else:
                    task()
                
                logger.debug(f"Completed shutdown task: {description}")
                
            except Exception as e:
                logger.error(f"Error in shutdown task {description}: {e}", exc_info=True)
        
        logger.info("All shutdown tasks completed")
    
    async def _save_all_data(self) -> None:
        """Save all persistent data."""
        try:
            logger.info("Saving all profiles...")
            save_all_profiles()
            save_all_server_profiles()
            logger.info("All profiles saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving profiles during shutdown: {e}", exc_info=True)
    
    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks."""
        try:
            logger.info("Stopping background tasks...")
            await stop_background_tasks()
            logger.info("Background tasks stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping background tasks: {e}", exc_info=True)
    
    async def _cleanup_tts(self) -> None:
        """Clean up TTS resources."""
        try:
            logger.info("Cleaning up TTS resources...")
            await cleanup_tts()
            logger.info("TTS cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during TTS cleanup: {e}", exc_info=True)
    
    async def _cleanup_external_clients(self) -> None:
        """Clean up external API clients."""
        try:
            logger.info("Closing external clients...")
            
            # Close Ollama client
            if ollama_client:
                await ollama_client.close()
                logger.debug("Ollama client closed")
            
            logger.info("External clients closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing external clients: {e}", exc_info=True)


# Global shutdown manager
_shutdown_manager: Optional[GracefulShutdown] = None


def graceful_shutdown(bot: commands.Bot) -> Callable:
    """Return a signal handler for graceful bot shutdown."""
    global _shutdown_manager
    
    if _shutdown_manager is None:
        _shutdown_manager = GracefulShutdown(bot)
    
    def signal_handler(signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        global _shutdown_in_progress
        
        if _shutdown_in_progress:
            logger.warning(f"Received signal {signum} during shutdown, forcing exit...")
            sys.exit(1)
        
        _shutdown_in_progress = True
        
        # Schedule the shutdown coroutine
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If event loop is running, schedule the shutdown
            asyncio.create_task(_shutdown_manager.execute_shutdown(signum))
        else:
            # If event loop is not running, run the shutdown directly
            asyncio.run(_shutdown_manager.execute_shutdown(signum))
        
        # Exit after shutdown
        sys.exit(0)
    
    return signal_handler


def setup_signal_handlers(bot: commands.Bot) -> None:
    """Set up signal handlers for graceful shutdown."""
    shutdown_handler = graceful_shutdown(bot)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    logger.info("Signal handlers configured for graceful shutdown")


def add_shutdown_task(task: Callable, description: str = "") -> None:
    """Add a custom shutdown task."""
    global _shutdown_manager
    
    if _shutdown_manager is None:
        logger.warning("Shutdown manager not initialized, cannot add shutdown task")
        return
    
    _shutdown_manager.add_shutdown_task(task, description)
    logger.debug(f"Added shutdown task: {description or task.__name__}")


async def emergency_shutdown(bot: commands.Bot, reason: str = "Emergency shutdown") -> None:
    """Perform an emergency shutdown without full cleanup."""
    logger.critical(f"Emergency shutdown triggered: {reason}")
    
    try:
        # Quick save of critical data
        save_all_profiles()
        save_all_server_profiles()
        
        # Close bot connection
        if not bot.is_closed():
            await bot.close()
        
        logger.critical("Emergency shutdown completed")
        
    except Exception as e:
        logger.critical(f"Error during emergency shutdown: {e}", exc_info=True)
    
    finally:
        sys.exit(1)


def is_shutdown_in_progress() -> bool:
    """Check if shutdown is currently in progress."""
    return _shutdown_in_progress
