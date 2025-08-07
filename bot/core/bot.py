"""Core bot implementation for Discord LLM Chatbot."""

from __future__ import annotations
import asyncio
import os
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Any

import discord
import io
from discord.ext import commands

from bot.config import load_system_prompts
from bot.util.logging import get_logger
from bot.metrics import NullMetrics
from bot.memory import load_all_profiles
from bot.memory.context_manager import ContextManager
from bot.memory.enhanced_context_manager import EnhancedContextManager

if TYPE_CHECKING:
    from bot.router import Router, BotAction
    from bot.tts import TTSManager


class LLMBot(commands.Bot):
    """Main bot class that extends the base Bot class with LLM capabilities."""

    def __init__(self, *args, config: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.logger = get_logger(__name__)
        self.metrics = NullMetrics()
        self.user_profiles = {}
        self.server_profiles = {}
        self.memory_save_task = None
        self.tts_manager: Optional[TTSManager] = None
        self.router: Optional[Router] = None
        self.background_tasks = []
        self._is_ready = asyncio.Event()
        self.system_prompts = {}
        self._processed_messages = set()
        self._dispatch_lock = asyncio.Lock()  # Global lock for processed messages tracking
        self._user_queues: Dict[str, asyncio.Queue] = {}  # Per-user message queues
        self._user_processors: Dict[str, asyncio.Task] = {}  # Per-user processing tasks
        
        # Track active long-running tasks for cancellation
        self._active_long_running_tasks: Dict[str, asyncio.Task] = {}  # task_id -> task
        self._task_metadata: Dict[str, Dict[str, Any]] = {}  # task_id -> metadata
        
        # Idempotency guard to prevent duplicate initialization [DRY][REH]
        self._boot_completed = False
        
        self.context_manager = ContextManager(
            self,
            filepath=self.config.get("CONTEXT_FILE_PATH", "context.json"),
            max_messages=self.config.get("MAX_CONTEXT_MESSAGES", 10)
        )
        # Enhanced context manager for multi-user conversation tracking
        self.enhanced_context_manager = EnhancedContextManager(
            self,
            filepath=self.config.get("ENHANCED_CONTEXT_FILE_PATH", "enhanced_context.json"),
            history_window=int(os.getenv("HISTORY_WINDOW", "10")),
            max_token_limit=self.config.get("MAX_CONTEXT_TOKENS", 4000)
        )

    async def setup_hook(self) -> None:
        """Asynchronous setup phase for the bot."""
        # Prevent duplicate initialization [DRY][REH]
        if self._boot_completed:
            self.logger.debug("🔄 Setup hook called but boot already completed, skipping")
            return
        
        self._boot_completed = True
        self.logger.info("🔧 Starting bot setup")
        
        try:
            # Initialize metrics
            try:
                from bot.metrics.prometheus import PrometheusMetrics
                self.metrics = PrometheusMetrics()
                self.logger.info("✅ Prometheus metrics initialized")
            except Exception:
                self.logger.warning("⚠️  Prometheus metrics not available, using NullMetrics")

            # Load system prompts
            self.system_prompts = load_system_prompts()
            self.logger.info("✅ Loaded system prompts")
            
            # Load user and server profiles
            await self.load_profiles()
            self.logger.info("✅ Loaded user profiles")
            
            # Set up background tasks
            self.setup_background_tasks()
            self.logger.info("✅ Background tasks configured")
            
            # Initialize TTS if configured
            await self.setup_tts()
            self.logger.info("✅ TTS system initialized")
            
            # Set up message router
            await self.setup_router()
            self.logger.info("✅ Message router configured")
            
            # Load command extensions
            await self.load_extensions()
            self.logger.info("✅ Command extensions loaded")
            
            self.logger.info("🚀 Bot setup complete")
            
        except Exception as e:
            self.logger.error(f"❌ Fatal error during bot setup: {e}", exc_info=True)
            self._boot_completed = False  # Reset flag on failure to allow retry
            raise

    async def on_ready(self):
        """Called when the bot is ready and connected to Discord."""
        # Simple ready state logging - all setup is handled in setup_hook() [DRY]
        if not self._is_ready.is_set():
            self.logger.info(f"🤖 Logged in as {self.user} (ID: {self.user.id})")
            self._is_ready.set()
            self.logger.info("🎉 Bot is ready to receive commands!")

    def _get_user_queue(self, user_id: str) -> asyncio.Queue:
        """Get or create a message queue for a specific user."""
        if user_id not in self._user_queues:
            self._user_queues[user_id] = asyncio.Queue()
        return self._user_queues[user_id]
    
    async def _ensure_user_processor(self, user_id: str):
        """Ensure a message processor task is running for the user."""
        if user_id not in self._user_processors or self._user_processors[user_id].done():
            self._user_processors[user_id] = asyncio.create_task(
                self._process_user_messages(user_id)
            )
            self.logger.debug(f"Started message processor for user {user_id}")
    
    async def _process_user_messages(self, user_id: str):
        """Process messages for a specific user in order, preventing lockout."""
        queue = self._get_user_queue(user_id)
        
        try:
            while True:
                # Wait for next message with timeout to allow cleanup
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=300)  # 5 minute timeout
                except asyncio.TimeoutError:
                    # No messages for 5 minutes, cleanup processor
                    self.logger.debug(f"Message processor for user {user_id} timed out, cleaning up")
                    break
                
                try:
                    await self._process_single_message(message)
                    self.logger.debug(f"Processed message {message.id} for user {user_id}")
                except Exception as e:
                    self.logger.error(f"Error processing message {message.id} for user {user_id}: {e}", exc_info=True)
                finally:
                    queue.task_done()
                    
        except Exception as e:
            self.logger.error(f"User message processor for {user_id} failed: {e}", exc_info=True)
        finally:
            # Cleanup processor reference
            if user_id in self._user_processors:
                del self._user_processors[user_id]
            self.logger.debug(f"Message processor for user {user_id} stopped")
    
    async def _process_single_message(self, message: discord.Message):
        """Process a single message through the full pipeline."""
        try:
            # Append message to context (both managers for backward compatibility)
            if self.context_manager:
                self.context_manager.append(message)
            
            # Enhanced context tracking for multi-user conversations
            if self.enhanced_context_manager:
                await self.enhanced_context_manager.append_message(message, role="user")

            guild_info = 'DM' if isinstance(message.channel, discord.DMChannel) else f"guild:{message.guild.id}"
            self.logger.info(
                f"Processing queued message: msg_id:{message.id} author:{message.author.id} in:{guild_info} len:{len(message.content)}"
            )

            # The router decides if this is a command, a direct message, or something to ignore.
            if self.router:
                action = await self.router.dispatch_message(message)
                if action:
                    if action.meta.get('delegated_to_cog'):
                        self.logger.info(f"Message {message.id} delegated to command processor.")
                        await self.process_commands(message)
                    elif action.has_payload:
                        await self._execute_action(message, action)
                    # If no payload and not delegated, the router decided to do nothing.
                else:
                    # Fallback for messages that don't trigger the router (e.g. standard commands)
                    await self.process_commands(message)
            else:
                self.logger.error("Router not initialized, falling back to command processing.")
                await self.process_commands(message)
                
        except Exception as e:
            self.logger.error(f"Error in _process_single_message for {message.id}: {e}", exc_info=True)

    def _is_long_running_admin_command(self, message: discord.Message) -> bool:
        """Check if this is a long-running admin command that should run out-of-band."""
        if not message.content:
            return False
        
        content = message.content.strip().lower()
        
        # List of long-running admin commands that should not block user queues
        long_running_commands = [
            '!rag bootstrap',
            '!rag refresh', 
            '!rag update',
            '!rag scan',
            # Add other potentially long-running commands here
        ]
        
        for cmd in long_running_commands:
            if content.startswith(cmd):
                return True
        
        return False
    
    def _generate_task_id(self, message: discord.Message) -> str:
        """Generate a unique task ID for tracking."""
        return f"{message.author.id}_{message.id}_{message.content.split()[0] if message.content else 'unknown'}"
    
    def _register_long_running_task(self, task_id: str, task: asyncio.Task, message: discord.Message, command: str) -> None:
        """Register a long-running task for tracking and cancellation."""
        self._active_long_running_tasks[task_id] = task
        self._task_metadata[task_id] = {
            'user_id': message.author.id,
            'channel_id': message.channel.id,
            'guild_id': message.guild.id if message.guild else None,
            'command': command,
            'started_at': asyncio.get_event_loop().time(),
            'message_id': message.id
        }
        
        # Add callback to clean up when task completes
        def cleanup_task(future):
            if task_id in self._active_long_running_tasks:
                del self._active_long_running_tasks[task_id]
            if task_id in self._task_metadata:
                del self._task_metadata[task_id]
        
        task.add_done_callback(cleanup_task)
        
        self.logger.info(f"Registered long-running task: {task_id} for command: {command}")
    
    def get_active_tasks_for_user(self, user_id: int) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all active long-running tasks for a specific user."""
        user_tasks = []
        for task_id, metadata in self._task_metadata.items():
            if metadata['user_id'] == user_id:
                user_tasks.append((task_id, metadata))
        return user_tasks
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a long-running task gracefully."""
        if task_id not in self._active_long_running_tasks:
            return False
        
        task = self._active_long_running_tasks[task_id]
        metadata = self._task_metadata.get(task_id, {})
        
        self.logger.info(f"Cancelling long-running task: {task_id} (command: {metadata.get('command', 'unknown')})")
        
        # Cancel the task
        task.cancel()
        
        try:
            # Wait a bit for graceful cancellation
            await asyncio.wait_for(task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            # Expected - task was cancelled or timed out
            pass
        except Exception as e:
            self.logger.warning(f"Error during task cancellation: {e}")
        
        # Clean up tracking
        if task_id in self._active_long_running_tasks:
            del self._active_long_running_tasks[task_id]
        if task_id in self._task_metadata:
            del self._task_metadata[task_id]
        
        return True
    
    async def _execute_out_of_band_command(self, message: discord.Message):
        """Execute a long-running command outside the user's message queue."""
        try:
            guild_info = 'DM' if isinstance(message.channel, discord.DMChannel) else f"guild:{message.guild.id}"
            self.logger.info(
                f"Executing out-of-band command: msg_id:{message.id} author:{message.author.id} in:{guild_info} cmd:{message.content[:50]}..."
            )
            
            # Process the command directly without queuing
            await self._process_single_message(message)
            
        except Exception as e:
            self.logger.error(f"Error in out-of-band command execution for {message.id}: {e}", exc_info=True)
            # Send error message to user
            try:
                await message.reply(f"❌ Error executing command: {str(e)[:100]}...", mention_author=True)
            except Exception:
                pass  # Don't let error handling errors crash the bot

    async def on_message(self, message: discord.Message):
        # DD-TODO: Use a more sophisticated cache like LRU to prevent memory growth.
        async with self._dispatch_lock:
            if message.id in self._processed_messages:
                self.logger.warning(f"Duplicate dispatch prevented for msg_id: {message.id}")
                return
            # Limit the size of the set to avoid unbounded memory growth
            if len(self._processed_messages) > 1000:
                self._processed_messages.pop()
            self._processed_messages.add(message.id)

        await self._is_ready.wait() # Wait until the bot is ready

        # Early returns before processing
        if message.author == self.user:
            return

        if (not message.content or not message.content.strip()) and not message.attachments:
            return

        # Check if this is a long-running admin command
        if self._is_long_running_admin_command(message):
            # Execute long-running commands out-of-band to not block user's message queue
            task_id = self._generate_task_id(message)
            command = message.content.split()[0] if message.content else 'unknown'
            
            # Create and register the task
            task = asyncio.create_task(self._execute_out_of_band_command(message))
            self._register_long_running_task(task_id, task, message, command)
            return

        # Queue regular messages for per-user processing to prevent lockout and ensure proper ordering
        user_id = str(message.author.id)
        
        # Get user's message queue and ensure processor is running
        user_queue = self._get_user_queue(user_id)
        await self._ensure_user_processor(user_id)
        
        # Add message to user's queue for sequential processing
        await user_queue.put(message)
        
        guild_info = 'DM' if isinstance(message.channel, discord.DMChannel) else f"guild:{message.guild.id}"
        self.logger.info(
            f"Message queued: msg_id:{message.id} author:{message.author.id} in:{guild_info} len:{len(message.content)} queue_size:{user_queue.qsize()}"
        )

    async def _execute_action(self, message: discord.Message, action: BotAction):
        """Executes a BotAction by sending its content to the appropriate channel."""
        self.logger.info(f"Executing action with meta: {action.meta} for message {message.id}")

        files = None
        # If action requires TTS, process it.
        if action.meta.get('requires_tts'):
            self.logger.info(f"Action requires TTS, processing... (msg_id: {message.id})")
            if not self.tts_manager:
                self.logger.error(f"TTS Manager not available, cannot process TTS action. (msg_id: {message.id})")
                action.content = "I tried to respond with voice, but the TTS service is not working."
            else:
                action = await self.tts_manager.process(action)

        # If action has an audio path after processing, prepare it for sending.
        if action.audio_path:
            if os.path.exists(action.audio_path):
                files = [discord.File(action.audio_path, filename="voice_message.ogg")]
            else:
                self.logger.error(f"Audio file not found: {action.audio_path} (msg_id: {message.id})")
                action.content = "I tried to send a voice message, but the audio file was missing."
        elif action.meta.get('requires_tts'): # Log error only if TTS was expected but failed
            self.logger.error(f"Audio file not generated after TTS processing. (msg_id: {message.id})")
            action.content = "I tried to send a voice message, but the audio file was missing."

        content = action.content
        # Note: User mentions are handled by Discord's mention_author=True in message.reply()
        # No manual mention prepending needed to avoid duplicate mentions

        # Discord has a 2000 character limit.
        if content and len(content) > 2000:
            self.logger.warning(f"Message content for {message.id} exceeds 2000 characters. Attaching as file.")
            file_content = io.BytesIO(content.encode('utf-8'))
            text_file = discord.File(fp=file_content, filename="full_response.txt")
            if files is None:
                files = []
            files.append(text_file)

            # The message becomes a notification about the attached file.
            if action.meta.get('is_reply'):
                content = f"{message.author.mention}\n\nThe response was too long to post directly. The full content is attached as a file."
            else:
                content = "The response was too long to post directly. The full content is attached as a file."

        # Show typing indicator while sending the message
        async with message.channel.typing():
            try:
                if content or action.embeds or files:
                    # Use message.reply() for contextual replies.
                    sent_message = await message.reply(content=content, embeds=action.embeds, files=files, mention_author=True)
                    
                    # Track bot response in enhanced context manager
                    if self.enhanced_context_manager and sent_message:
                        await self.enhanced_context_manager.append_message(sent_message, role="bot")
            except discord.errors.HTTPException as e:
                # If replying fails (e.g., original message deleted), send a normal message instead.
                if e.code == 50035: # Unknown Message
                    self.logger.warning(f"Replying to message {message.id} failed (likely deleted). Falling back to channel send.")
                    sent_message = await message.channel.send(content=content, embeds=action.embeds, files=files)
                    
                    # Track bot response in enhanced context manager
                    if self.enhanced_context_manager and sent_message:
                        await self.enhanced_context_manager.append_message(sent_message, role="bot")
                else:
                    self.logger.error(f"Failed to send message: {e} (msg_id: {message.id})")
                    raise # Re-raise other HTTP exceptions

    async def load_profiles(self) -> None:
        """Load user and server memory profiles."""
        try:
            self.logger.info("Loading memory profiles...")
            self.user_profiles, self.server_profiles = load_all_profiles()
            self.logger.info(f"Loaded {len(self.user_profiles)} user and {len(self.server_profiles)} server profiles.")
        except Exception as e:
            self.logger.error(f"Failed to load profiles: {e}", exc_info=True)

    def setup_background_tasks(self) -> None:
        """Set up background tasks for the bot."""
        try:
            from bot.tasks import setup_memory_save_task
            self.memory_save_task = setup_memory_save_task(self)
            self.memory_save_task.start()
        except Exception as e:
            self.logger.error(f"Failed to set up background tasks: {e}", exc_info=True)

    async def setup_tts(self) -> None:
        """Set up TTS manager if configured."""
        try:
            from bot.tts.interface import TTSManager
            self.tts_manager = TTSManager(self)
            self.logger.info("TTS manager initialized")
        except Exception as e:
            self.logger.error(f"Failed to set up TTS: {e}", exc_info=True)

    async def setup_router(self) -> None:
        """Set up message router."""
        try:
            from bot.router import Router
            self.router = Router(self)
            self.logger.info("Router initialized")
        except Exception as e:
            self.logger.error(f"Failed to set up router: {e}", exc_info=True)

    async def load_extensions(self) -> None:
        """Load command extensions."""
        try:
            from bot.commands import setup_commands
            await setup_commands(self)
            self.logger.info("Commands loaded")
        except Exception as e:
            self.logger.error(f"Failed to load extensions: {e}", exc_info=True)

    async def connect(self, *, reconnect: bool = True) -> None:
        """Connect to Discord."""
        try:
            await super().connect(reconnect=reconnect)
        except discord.ConnectionClosed as e:
            self.logger.error(f"Connection closed: {e}")
            # Attempt to reconnect after a delay
            await asyncio.sleep(5)
            await self.connect(reconnect=reconnect)

    async def close(self) -> None:
        """Clean up resources before shutdown."""
        self.logger.info("Bot is shutting down...")
        
        try:
            # CRITICAL: Close Discord connection FIRST to stop heartbeat thread
            # This prevents the "Event loop is closed" error from heartbeat thread
            if not self.is_closed():
                self.logger.info("Closing Discord connection...")
                try:
                    # Use a timeout to prevent hanging on Discord close
                    await asyncio.wait_for(super().close(), timeout=8.0)
                    self.logger.info("Discord connection closed successfully")
                except asyncio.TimeoutError:
                    self.logger.warning("Discord close timed out, forcing closure")
                    # Force close by setting internal state
                    if hasattr(self, '_closed'):
                        self._closed = True
                except Exception as e:
                    self.logger.warning(f"Error closing Discord connection: {e}")
            
            # Cancel user message processors
            if self._user_processors:
                self.logger.info(f"Cancelling {len(self._user_processors)} user message processors")
                for user_id, processor in self._user_processors.items():
                    if not processor.done():
                        processor.cancel()
                
                # Wait for processors to cancel with short timeout
                if self._user_processors:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*self._user_processors.values(), return_exceptions=True),
                            timeout=3.0
                        )
                    except asyncio.TimeoutError:
                        self.logger.warning("User processors did not cancel within timeout")
                
                self._user_processors.clear()
                self._user_queues.clear()
            
            # Cancel memory save task
            if self.memory_save_task and not self.memory_save_task.done():
                self.memory_save_task.cancel()
                try:
                    await asyncio.wait_for(self.memory_save_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for background tasks to complete with timeout
            if self.background_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self.background_tasks, return_exceptions=True),
                        timeout=3.0
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Background tasks did not complete within timeout")
            
            # Close TTS manager
            if self.tts_manager:
                try:
                    await asyncio.wait_for(self.tts_manager.close(), timeout=2.0)
                except asyncio.TimeoutError:
                    self.logger.warning("TTS manager close timed out")
            
            # Close global ollama client first
            try:
                from .ollama import ollama_client
                if ollama_client and hasattr(ollama_client, 'close'):
                    self.logger.debug("Closing global ollama client")
                    await asyncio.wait_for(ollama_client.close(), timeout=2.0)
            except Exception as e:
                self.logger.debug(f"Error closing global ollama client: {e}")
            
            # Close aiohttp sessions
            await self._close_all_aiohttp_sessions()
            
            # Close the database connection
            if hasattr(self, 'db') and self.db.is_connected:
                self.db.close()
                
            # Close any other resources
            if hasattr(self, 'rag_system'):
                try:
                    await asyncio.wait_for(self.rag_system.close(), timeout=2.0)
                except asyncio.TimeoutError:
                    self.logger.warning("RAG system close timed out")
            
            # Cancel remaining tasks with aggressive timeout
            current_task = asyncio.current_task()
            tasks = [t for t in asyncio.all_tasks() 
                    if t is not current_task and not t.done() and not t.cancelled()]
            
            if tasks:
                self.logger.info(f"Cancelling {len(tasks)} remaining background tasks")
                for task in tasks:
                    task.cancel()
                
                # Very short timeout for remaining tasks
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Some tasks did not cancel within final timeout")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)
        finally:
            self.logger.info("Bot shutdown complete")
    
    async def _close_all_aiohttp_sessions(self) -> None:
        """Close all aiohttp sessions to prevent shutdown warnings."""
        try:
            self.logger.debug("Closing all aiohttp sessions...")
            
            # Close bot's own session if it exists
            if hasattr(self, 'session') and self.session and not self.session.closed:
                self.logger.debug("Closing bot session")
                await self.session.close()
                await asyncio.sleep(0.05)  # Small delay for cleanup
            
            # Close ollama backend session
            if hasattr(self, 'router') and self.router:
                if hasattr(self.router, 'ollama_backend') and self.router.ollama_backend:
                    self.logger.debug("Closing ollama backend session")
                    await self.router.ollama_backend.close()
                    await asyncio.sleep(0.05)
            
            # Close any HTTP client sessions in various modules
            modules_to_check = [
                'ollama',  # Global ollama client
                'utils',   # Any utility HTTP clients
                'rag',     # RAG system HTTP clients
            ]
            
            for module_name in modules_to_check:
                try:
                    if hasattr(self, module_name):
                        module = getattr(self, module_name)
                        if hasattr(module, 'close'):
                            self.logger.debug(f"Closing {module_name} HTTP sessions")
                            await module.close()
                            await asyncio.sleep(0.05)
                except Exception as e:
                    self.logger.debug(f"Error closing {module_name} sessions: {e}")
            
            # Find and close any remaining aiohttp sessions using garbage collection
            import gc
            import aiohttp
            
            # Force garbage collection to expose any remaining sessions
            gc.collect()
            
            # Find any remaining ClientSession objects
            remaining_sessions = []
            for obj in gc.get_objects():
                if isinstance(obj, aiohttp.ClientSession) and not obj.closed:
                    remaining_sessions.append(obj)
            
            if remaining_sessions:
                self.logger.warning(f"Found {len(remaining_sessions)} unclosed aiohttp sessions, closing them")
                for session in remaining_sessions:
                    try:
                        await session.close()
                        await asyncio.sleep(0.01)
                    except Exception as e:
                        self.logger.debug(f"Error closing remaining session: {e}")
            
            # Final garbage collection
            gc.collect()
            
            # Give more time for all sessions to properly close
            await asyncio.sleep(0.3)
            
            self.logger.debug("All aiohttp sessions closed")
            
        except Exception as e:
            self.logger.warning(f"Error closing aiohttp sessions: {e}")
