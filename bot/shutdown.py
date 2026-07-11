"""Graceful shutdown handling for the Discord bot."""

import asyncio
import logging
import os
import signal
import sys
from collections.abc import Callable
from typing import Any

from discord.ext import commands

from .memory import save_all_profiles, save_all_server_profiles
from .ollama import ollama_client
from .tasks import stop_background_tasks


def _save_all_data_sync() -> tuple[bool, bool]:
    """Persist all profile data off the event loop."""
    return save_all_profiles(), save_all_server_profiles()


logger = logging.getLogger(__name__)

# Global shutdown state
_shutdown_in_progress = False
_shutdown_timeout = 30  # seconds


class GracefulShutdown:
    """Handles graceful shutdown of the bot."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.shutdown_tasks = []
        self.shutdown_in_progress = False
        self._shutdown_event = asyncio.Event()

    async def shutdown_with_timeout(self, timeout: float = 30.0) -> None:
        """Shutdown with timeout to prevent hanging."""
        if self.shutdown_in_progress:
            return

        self.shutdown_in_progress = True
        logger.info(f"Starting graceful shutdown with {timeout}s timeout")

        try:
            # Set shutdown event
            self._shutdown_event.set()

            # Perform shutdown with timeout
            await asyncio.wait_for(self._perform_shutdown(), timeout=timeout)

        except TimeoutError:
            logger.warning(f"Shutdown timed out after {timeout}s, forcing exit")
            # Force close any remaining resources
            await self._force_cleanup()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
            await self._force_cleanup()

    async def _perform_shutdown(self) -> None:
        """Perform the actual shutdown sequence."""
        # Close the bot properly
        if self.bot and not self.bot.is_closed():
            await self.bot.close()

        # Run any additional shutdown tasks
        for task_func in self.shutdown_tasks:
            try:
                if asyncio.iscoroutinefunction(task_func):
                    await task_func()
                else:
                    task_func()
            except Exception as e:
                logger.exception(f"Error in shutdown task: {e}")

    async def _force_cleanup(self) -> None:
        """Force cleanup of resources when normal shutdown fails."""
        try:
            # Kill any running subprocesses first
            await self._cleanup_subprocesses()

            # Cancel all tasks
            tasks = [t for t in asyncio.all_tasks() if not t.done()]
            for task in tasks:
                task.cancel()

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.exception(f"Error in force cleanup: {e}")

    def add_shutdown_task(self, task: Callable, description: str = "") -> None:
        """Add a task to be executed during shutdown."""
        self.shutdown_tasks.append({"task": task, "description": description or task.__name__})

    async def execute_shutdown(self, signal_num: int | None = None) -> None:
        """Execute graceful shutdown sequence."""
        if self.shutdown_in_progress:
            logger.warning("Shutdown already in progress")
            return

        self.shutdown_in_progress = True
        shutdown_start_time = asyncio.get_event_loop().time()

        if signal_num:
            logger.info(f"🔄 Received signal {signal_num}, initiating graceful shutdown...")
        else:
            logger.info("🔄 Initiating graceful shutdown...")

        try:
            # Set shutdown flag to prevent RAG initialization during shutdown
            try:
                from bot.rag.hybrid_search import set_shutdown_flag

                set_shutdown_flag(True)
            except (ImportError, AttributeError, RuntimeError, OSError) as e:
                logger.warning(f"Failed to set RAG shutdown flag: {e}")

            # Execute shutdown tasks with timeout
            logger.debug(f"⏱️ Starting shutdown tasks with {_shutdown_timeout}s timeout")
            await asyncio.wait_for(self._execute_shutdown_tasks(), timeout=_shutdown_timeout)

            shutdown_duration = asyncio.get_event_loop().time() - shutdown_start_time
            logger.info(f"✅ Graceful shutdown completed successfully in {shutdown_duration:.2f}s")

        except TimeoutError:
            shutdown_duration = asyncio.get_event_loop().time() - shutdown_start_time
            logger.exception(f"❌ Shutdown timed out after {_shutdown_timeout} seconds (actual: {shutdown_duration:.2f}s)")
        except (RuntimeError, OSError, ValueError, AttributeError, TimeoutError) as e:
            shutdown_duration = asyncio.get_event_loop().time() - shutdown_start_time
            logger.error(
                f"❌ Error during shutdown after {shutdown_duration:.2f}s: {e}",
                exc_info=True,
            )
        finally:
            # Force exit if needed
            if self.bot.is_closed():
                logger.debug("✅ Bot connection already closed")
            else:
                logger.warning("⚠️ Bot connection still open after shutdown")
                await self.bot.close()

    async def _execute_shutdown_tasks(self) -> None:
        """Execute all registered shutdown tasks."""
        logger.info("🔧 Executing shutdown tasks...")
        task_start_time = asyncio.get_event_loop().time()

        # Save all data first
        logger.debug("💾 Starting data save...")
        save_start = asyncio.get_event_loop().time()
        await self._save_all_data()
        save_duration = asyncio.get_event_loop().time() - save_start
        logger.debug(f"✅ Data save completed in {save_duration:.2f}s")

        # Stop background tasks
        logger.debug("🛑 Stopping background tasks...")
        bg_start = asyncio.get_event_loop().time()
        await self._stop_background_tasks()
        bg_duration = asyncio.get_event_loop().time() - bg_start
        logger.debug(f"✅ Background tasks stopped in {bg_duration:.2f}s")

        # Clean up TTS resources
        logger.debug("🎵 Cleaning up TTS resources...")
        tts_start = asyncio.get_event_loop().time()
        await self._cleanup_tts()
        tts_duration = asyncio.get_event_loop().time() - tts_start
        logger.debug(f"✅ TTS cleanup completed in {tts_duration:.2f}s")

        # Clean up RAG system resources
        logger.debug("🧠 Cleaning up RAG system...")
        rag_start = asyncio.get_event_loop().time()
        await self._cleanup_rag_system()
        rag_duration = asyncio.get_event_loop().time() - rag_start
        logger.debug(f"✅ RAG cleanup completed in {rag_duration:.2f}s")

        # Clean up external API clients
        logger.debug("🌐 Cleaning up external clients...")
        ext_start = asyncio.get_event_loop().time()
        await self._cleanup_external_clients()
        ext_duration = asyncio.get_event_loop().time() - ext_start
        logger.debug(f"✅ External clients cleanup completed in {ext_duration:.2f}s")

        # Execute any custom shutdown tasks
        if self.shutdown_tasks:
            logger.debug(f"🔧 Executing {len(self.shutdown_tasks)} custom shutdown tasks...")
            for i, task_info in enumerate(self.shutdown_tasks, 1):
                try:
                    task = task_info["task"]
                    description = task_info["description"]

                    logger.debug(f"🔧 [{i}/{len(self.shutdown_tasks)}] Executing: {description}")
                    custom_start = asyncio.get_event_loop().time()

                    if asyncio.iscoroutinefunction(task):
                        await task()
                    else:
                        task()

                    custom_duration = asyncio.get_event_loop().time() - custom_start
                    logger.debug(f"✅ [{i}/{len(self.shutdown_tasks)}] Completed '{description}' in {custom_duration:.2f}s")

                except Exception as e:
                    logger.error(f"❌ Error in shutdown task '{description}': {e}", exc_info=True)

        total_duration = asyncio.get_event_loop().time() - task_start_time
        logger.info(f"✅ All shutdown tasks completed in {total_duration:.2f}s")

    async def _save_all_data(self) -> None:
        """Save all persistent data."""
        try:
            logger.info("Saving all profiles...")
            await asyncio.to_thread(_save_all_data_sync)
            logger.info("All profiles saved successfully")

        except Exception as e:
            logger.error(f"Error saving profiles during shutdown: {e}", exc_info=True)

        # Flush context managers -- append()/append_message() now only mark an
        # in-memory dirty flag instead of writing on every message (see
        # bot/tasks.py's context_autosave task), so shutdown must force one last
        # flush or the last <=CONTEXT_AUTOSAVE_INTERVAL_SECONDS of conversation
        # context would be lost on a clean stop, not just a crash. [PA]
        try:
            context_manager = getattr(self.bot, "context_manager", None)
            if context_manager is not None:
                flushed = await context_manager.flush_if_dirty()
                logger.info(f"Context manager flushed on shutdown: {flushed}")

            enhanced_context_manager = getattr(self.bot, "enhanced_context_manager", None)
            if enhanced_context_manager is not None:
                flushed = await enhanced_context_manager.flush_if_dirty()
                logger.info(f"Enhanced context manager flushed on shutdown: {flushed}")

        except Exception as e:
            logger.error(f"Error flushing context managers during shutdown: {e}", exc_info=True)

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
            if hasattr(self.bot, "tts_manager") and self.bot.tts_manager:
                # Check if TTS manager has a close method
                if hasattr(self.bot.tts_manager, "close"):
                    close_method = self.bot.tts_manager.close
                    if asyncio.iscoroutinefunction(close_method):
                        await close_method()
                    else:
                        close_method()
                    logger.info("TTS manager closed successfully.")
                # Check if TTS manager has a cleanup method
                elif hasattr(self.bot.tts_manager, "cleanup"):
                    cleanup_method = self.bot.tts_manager.cleanup
                    if asyncio.iscoroutinefunction(cleanup_method):
                        await cleanup_method()
                    else:
                        cleanup_method()
                    logger.info("TTS manager cleanup completed.")
                else:
                    # Just set to None to release resources
                    self.bot.tts_manager = None
                    logger.info("TTS manager reference cleared.")
            else:
                logger.info("TTS manager not available, skipping cleanup.")

        except Exception as e:
            logger.error(f"Error during TTS cleanup: {e}", exc_info=True)

    async def _cleanup_rag_system(self) -> None:
        """Clean up RAG system resources."""
        try:
            logger.info("Cleaning up RAG system...")

            # Cancel any active long-running RAG tasks
            if hasattr(self.bot, "_active_long_running_tasks"):
                active_tasks = dict(self.bot._active_long_running_tasks)  # Copy to avoid modification during iteration
                if active_tasks:
                    logger.info(f"Cancelling {len(active_tasks)} active RAG tasks...")
                    for task_id, task in active_tasks.items():
                        try:
                            if not task.done():
                                task.cancel()
                                logger.debug(f"Cancelled RAG task: {task_id}")
                        except (RuntimeError, OSError, ValueError, AttributeError) as e:
                            logger.warning(f"Error cancelling RAG task {task_id}: {e}")

                    # Wait briefly for tasks to clean up
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(
                                *[task for task in active_tasks.values() if not task.done()],
                                return_exceptions=True,
                            ),
                            timeout=3.0,
                        )
                    except TimeoutError:
                        logger.warning("Some RAG tasks did not complete within timeout")

                    # Clear the task tracking
                    self.bot._active_long_running_tasks.clear()
                    if hasattr(self.bot, "_task_metadata"):
                        self.bot._task_metadata.clear()

            # Clean up RAG search engine if available (WITHOUT triggering initialization)
            try:
                # Import the module but do NOT call get_hybrid_search() as it triggers initialization
                from bot.rag import hybrid_search

                # Check if there's an existing search engine instance to clean up
                if hasattr(hybrid_search, "_hybrid_search") and hybrid_search._hybrid_search:
                    search_engine = hybrid_search._hybrid_search
                    logger.debug("Found existing RAG search engine, cleaning up...")

                    # Close any database connections or resources
                    if hasattr(search_engine, "close"):
                        try:
                            if asyncio.iscoroutinefunction(search_engine.close):
                                await asyncio.wait_for(search_engine.close(), timeout=5.0)
                            else:
                                search_engine.close()
                            logger.debug("RAG search engine closed")
                        except TimeoutError:
                            logger.warning("RAG search engine close timed out")
                        except (RuntimeError, OSError, ValueError, AttributeError, TimeoutError) as e:
                            logger.debug(f"RAG search engine close warning: {e}")

                    # Clean up ChromaDB client if available
                    if hasattr(search_engine, "rag_backend") and search_engine.rag_backend:
                        try:
                            backend = search_engine.rag_backend
                            if hasattr(backend, "client") and backend.client:
                                if hasattr(backend.client, "close"):
                                    backend.client.close()
                                logger.debug("ChromaDB client cleaned up")
                        except (RuntimeError, OSError, ValueError, AttributeError) as e:
                            logger.debug(f"ChromaDB client cleanup warning: {e}")

                    # Clear the global reference
                    hybrid_search._hybrid_search = None
                    logger.debug("Cleared global RAG search engine reference")
                else:
                    logger.debug("No existing RAG search engine found to clean up")

            except ImportError:
                logger.debug("RAG system not available for cleanup")
            except (RuntimeError, OSError, ValueError, AttributeError, ImportError) as e:
                logger.warning(f"Error during RAG search engine cleanup: {e}")

            logger.info("✅ RAG system cleanup completed")

        except (RuntimeError, OSError, ValueError, AttributeError, ImportError) as e:
            logger.error(f"Error during RAG system cleanup: {e}", exc_info=True)

    async def _cleanup_subprocesses(self) -> None:
        """Clean up any running subprocesses to prevent hanging."""
        try:
            import psutil

            current_process = psutil.Process()
            children = current_process.children(recursive=True)

            if children:
                logger.info(f"Found {len(children)} child processes, terminating...")

                # First try graceful termination
                for child in children:
                    try:
                        if child.is_running():
                            child.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # Wait a bit for graceful termination
                await asyncio.sleep(1.0)

                # Force kill any remaining processes
                for child in children:
                    try:
                        if child.is_running():
                            logger.warning(f"Force killing subprocess PID {child.pid}")
                            child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                logger.info("All subprocesses cleaned up")

        except ImportError:
            logger.warning("psutil not available, cannot clean up subprocesses")
        except Exception as e:
            logger.exception(f"Error cleaning up subprocesses: {e}")

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
_shutdown_manager: GracefulShutdown | None = None


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
            # Force exit immediately on second signal
            os._exit(1)

        _shutdown_in_progress = True
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")

        try:
            # Get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule shutdown and let the event loop handle it
                # Don't call sys.exit immediately - let the shutdown complete
                task = asyncio.create_task(_shutdown_manager.execute_shutdown(signum))

                # Add a callback to exit when shutdown is complete
                def on_shutdown_complete(future) -> None:
                    try:
                        future.result()  # Check for exceptions
                        logger.info("Graceful shutdown completed, exiting...")
                    except Exception as e:
                        logger.exception(f"Shutdown failed: {e}")
                    finally:
                        # Use os._exit to avoid any cleanup issues
                        os._exit(0)

                task.add_done_callback(on_shutdown_complete)
            else:
                # If event loop is not running, run the shutdown directly
                try:
                    asyncio.run(_shutdown_manager.execute_shutdown(signum))
                    logger.info("Graceful shutdown completed, exiting...")
                except Exception as e:
                    logger.exception(f"Shutdown failed: {e}")
                finally:
                    os._exit(0)
        except Exception as e:
            logger.exception(f"Error in signal handler: {e}")
            os._exit(1)

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
