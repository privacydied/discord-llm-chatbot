"""Core bot implementation for Discord LLM Chatbot."""

from __future__ import annotations
import asyncio
import os
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Any

import discord
import io
import uuid
from discord.ext import commands
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel

from bot.config import load_system_prompts
from bot.util.logging import get_logger
from bot.metrics import NullMetrics
from bot.memory import load_all_profiles
from bot.memory.context_manager import ContextManager
from bot.memory.enhanced_context_manager import EnhancedContextManager
from bot.events import setup_command_error_handler

if TYPE_CHECKING:
    from bot.router import Router, BotAction
    from bot.tts import TTSManager


def log_commands_setup(
    console: Console,
    command_modules: List[Tuple[str, bool]],
    command_cogs: List[Tuple[str, bool]],
    total_commands: int
) -> None:
    """
    Log command setup progress using Rich's Tree and Panel for visual reporting.
    
    This function creates a structured visual report of the command setup process,
    showing the status of module imports and cog registrations in a tree format.
    
    Args:
        console: Rich Console instance for output
        command_modules: List of (module_name, success_status) tuples for imports
        command_cogs: List of (cog_name, success_status) tuples for cog registration
        total_commands: Total number of commands registered across all cogs
    
    The output includes:
    - A root node titled "🎬 Commands Setup"
    - A branch "📦 Import modules" with ✅/❌ status for each module
    - A branch "⚙️ Load cogs" with ✅/❌ status for each cog
    - A summary with success/failure counts and total command count
    """
    # Create the main tree structure
    tree = Tree("🎬 Commands Setup")
    
    # Add modules branch
    modules_branch = tree.add("📦 Import modules")
    modules_success = 0
    modules_failed = 0
    
    for module_name, module_ok in command_modules:
        status_icon = "✅" if module_ok else "❌"
        modules_branch.add(f"{status_icon} {module_name}")
        if module_ok:
            modules_success += 1
        else:
            modules_failed += 1
    
    # Add cogs branch
    cogs_branch = tree.add("⚙️ Load cogs")
    cogs_success = 0
    cogs_failed = 0
    
    for cog_name, cog_ok in command_cogs:
        status_icon = "✅" if cog_ok else "❌"
        cogs_branch.add(f"{status_icon} {cog_name}")
        if cog_ok:
            cogs_success += 1
        else:
            cogs_failed += 1
    
    # Add summary branch
    total_success = modules_success + cogs_success
    total_failed = modules_failed + cogs_failed
    
    summary_branch = tree.add("📊 Summary")
    summary_branch.add(f"🎉 Complete: {total_success} loaded, {total_failed} failed")
    summary_branch.add(f"📋 Total commands registered: {total_commands}")
    
    # Create panel and print
    panel = Panel(
        tree,
        title="Command Setup Report",
        border_style="blue",
        padding=(1, 2)
    )
    
    console.print(panel)


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
        
        # Rich console for enhanced command setup logging
        self.console = Console()
        
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
            
            # Initialize RAG system (with eager loading if configured)
            await self.setup_rag()
            self.logger.info("✅ RAG system configured")
            
            # Load command extensions
            await self.load_extensions()
            self.logger.info("✅ Command extensions loaded")
            
            # Setup global command error handler
            self.command_error_handler = await setup_command_error_handler(self)
            self.logger.info("✅ Global command error handler configured")
            
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
        """Executes a BotAction by sending its content to the appropriate channel.
        Adds detailed dispatch instrumentation, readiness gating, and empty-content guardrails.
        """
        # Metadata for logging
        guild_id = getattr(message.guild, 'id', None)
        channel_id = getattr(message.channel, 'id', None)
        user_id = getattr(message.author, 'id', None)
        is_dm = isinstance(message.channel, discord.DMChannel)
        debug_token = f"d{message.id}-{uuid.uuid4().hex[:8]}"

        base_extra = {"guild_id": guild_id, "user_id": user_id, "msg_id": message.id}
        self.logger.info(
            f"dispatch:pre | channel_id={channel_id} is_dm={is_dm} ready={self._is_ready.is_set()} "
            f"embeds={len(action.embeds) if action.embeds else 0} meta={action.meta}",
            extra={**base_extra, "event": "dispatch.send.pre"},
        )

        # Gate on bot readiness (avoid sending before on_ready); wait briefly if needed
        if not self._is_ready.is_set():
            try:
                await asyncio.wait_for(self._is_ready.wait(), timeout=5.0)
                self.logger.info(
                    f"dispatch:ready | ready=True",
                    extra={**base_extra, "event": "dispatch.ready.wait.ok"},
                )
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"dispatch:ready | ready=False",
                    extra={**base_extra, "event": "dispatch.ready.wait.timeout"},
                )

        files = None
        # If action requires TTS, process it.
        if action.meta.get('requires_tts'):
            self.logger.info(
                "TTS requested, processing…",
                extra={**base_extra, "event": "tts.process.start"},
            )
            if not self.tts_manager:
                self.logger.error(
                    "tts:missing",
                    extra={**base_extra, "event": "tts.manager.missing"},
                )
                action.content = "I tried to respond with voice, but the TTS service is not working."
            else:
                action = await self.tts_manager.process(action)

        # If action has an audio path after processing, prepare it for sending.
        if action.audio_path:
            if os.path.exists(action.audio_path):
                files = [discord.File(action.audio_path, filename="voice_message.ogg")]
            else:
                self.logger.error(
                    f"tts:file_missing | path={action.audio_path}",
                    extra={**base_extra, "event": "tts.file.missing"},
                )
                action.content = "I tried to send a voice message, but the audio file was missing."
        elif action.meta.get('requires_tts'):  # TTS expected but failed
            self.logger.error(
                "tts:no_audio",
                extra={**base_extra, "event": "tts.audio.not_generated"},
            )
            action.content = "I tried to send a voice message, but the audio file was missing."

        content = action.content or ""
        embed_count = len(action.embeds) if action.embeds else 0
        file_count = len(files) if files else 0

        # Discord has a 2000 character limit: attach overflow as file
        if content and len(content) > 2000:
            self.logger.warning(
                f"dispatch:overflow | length={len(content)}",
                extra={**base_extra, "event": "dispatch.content.overflow"},
            )
            file_content = io.BytesIO(content.encode('utf-8'))
            text_file = discord.File(fp=file_content, filename="full_response.txt")
            if files is None:
                files = []
            files.append(text_file)
            file_count = len(files)

            # The message becomes a notification about the attached file.
            if action.meta.get('is_reply'):
                content = f"{message.author.mention}\n\nThe response was too long to post directly. The full content is attached as a file."
            else:
                content = "The response was too long to post directly. The full content is attached as a file."

        # Guardrail: if everything is empty, synthesize a fallback message
        if (not content.strip()) and embed_count == 0 and file_count == 0:
            content = (
                f"ℹ️ I generated an empty response. I've logged this so it can be fixed. "
                f"Please try again. [ref: {debug_token}]"
            )
            self.logger.warning(
                f"dispatch:empty | ref={debug_token}",
                extra={**base_extra, "event": "dispatch.guard.empty"},
            )

        # Prepare preview for logs
        preview = content.replace("\n", " ")[:120] if content else ""
        self.logger.info(
            f"dispatch:attempt | content_len={len(content)} preview=\"{preview}\" embeds={embed_count} files={file_count}",
            extra={**base_extra, "event": "dispatch.send.attempt"},
        )

        sent_message = None
        # Show typing indicator while sending the message
        async with message.channel.typing():
            try:
                if content or action.embeds or files:
                    # Use message.reply() for contextual replies.
                    sent_message = await message.reply(
                        content=content,
                        embeds=action.embeds,
                        files=files,
                        mention_author=True,
                    )

                    self.logger.info(
                        f"dispatch:ok | discord_msg_id={getattr(sent_message, 'id', None)} embeds={embed_count} files={file_count}",
                        extra={**base_extra, "event": "dispatch.send.ok"},
                    )
                    
                    # Track bot response in enhanced context manager
                    if self.enhanced_context_manager and sent_message:
                        await self.enhanced_context_manager.append_message(sent_message, role="bot")
            except discord.errors.HTTPException as e:
                # If replying fails (e.g., original message deleted), send a normal message instead.
                if e.code == 50035:  # Unknown Message
                    self.logger.warning(
                        "dispatch:fallback | reason=unknown_message",
                        extra={**base_extra, "event": "dispatch.send.reply_fallback"},
                    )
                    sent_message = await message.channel.send(content=content, embeds=action.embeds, files=files)
                    
                    self.logger.info(
                        f"dispatch:ok | discord_msg_id={getattr(sent_message, 'id', None)} embeds={embed_count} files={file_count}",
                        extra={**base_extra, "event": "dispatch.send.ok"},
                    )
                    # Track bot response in enhanced context manager
                    if self.enhanced_context_manager and sent_message:
                        await self.enhanced_context_manager.append_message(sent_message, role="bot")
                else:
                    self.logger.error(
                        f"dispatch:error | code={e.code} status={getattr(e, 'status', 'n/a')} details={str(e)}",
                        extra={**base_extra, "event": "dispatch.send.error"},
                        exc_info=True,
                    )
                    raise  # Re-raise other HTTP exceptions
            finally:
                self.logger.debug(
                    f"dispatch:finalize | sent={(sent_message is not None)}",
                    extra={**base_extra, "event": "dispatch.send.finalize"},
                )

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
            self.router = Router(self)  # Pass bot instance, not config dict
            self.logger.debug("✅ Message router initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize message router: {e}", exc_info=True)
            raise
    
    async def setup_rag(self) -> None:
        """Set up RAG system to enable eager loading if configured."""
        try:
            # Check if RAG is configured and eager loading is enabled
            rag_enabled = self.config.get('rag_enabled', True)
            if not rag_enabled:
                self.logger.info("⚠️  RAG system disabled via configuration")
                return
                
            # Import RAG config and check eager loading setting
            from bot.rag.config import get_rag_config
            rag_config = get_rag_config()
            
            if rag_config.eager_vector_load:
                self.logger.info("🚀 RAG eager loading enabled - initializing RAG system at startup")
                
                # Initialize the RAG system to trigger eager loading
                from bot.rag.hybrid_search import get_hybrid_search
                search_engine = await get_hybrid_search()
                
                self.logger.info("✅ RAG system initialized with eager vector loading")
            else:
                self.logger.info("⏱️  RAG lazy loading enabled - deferring initialization until first use")
                
        except Exception as e:
            # RAG initialization failure should not crash the bot
            self.logger.warning(f"⚠️  RAG system initialization failed (bot will continue without RAG): {e}")
            if self.config.get('debug', False):
                self.logger.debug("RAG initialization traceback:", exc_info=True)

    async def load_extensions(self) -> None:
        """Load command extensions using Rich visual reporting."""
        import importlib
        
        # Define command modules and their corresponding cog classes
        module_definitions = [
            ("test_cmds", "TestCommands"),
            ("memory_cmds", "MemoryCommands"),
            ("tts_cmds", "TTSCommands"),
            ("config_commands", "ConfigCommands"),
            ("video_commands", "VideoCommands"),
            ("rag_commands", "RAGCommands")
        ]
        
        command_modules = []  # List of (module_name, success_status)
        command_cogs = []     # List of (cog_name, success_status)
        loaded_modules = {}   # Store successfully loaded modules
        
        # Phase 1: Import command modules
        for module_name, cog_class_name in module_definitions:
            try:
                self.logger.debug(f"Importing {module_name}...")
                module = importlib.import_module(f"bot.commands.{module_name}")
                loaded_modules[cog_class_name] = module
                command_modules.append((module_name, True))
                self.logger.debug(f"✅ Successfully imported {module_name}")
            except Exception as import_error:
                command_modules.append((module_name, False))
                self.logger.error(f"❌ Failed to import {module_name}: {import_error}", exc_info=True)
        
        # Phase 2: Load and register cogs
        for cog_class_name, module in loaded_modules.items():
            try:
                # Check if cog is already loaded to avoid duplicates
                if self.get_cog(cog_class_name):
                    self.logger.debug(f"Skipping already loaded cog: {cog_class_name}")
                    command_cogs.append((cog_class_name, True))
                    continue
                
                self.logger.debug(f"Loading {cog_class_name} cog...")
                
                # Check if module has setup function
                if hasattr(module, 'setup'):
                    await module.setup(self)
                    
                    # Verify the cog was actually loaded
                    if self.get_cog(cog_class_name):
                        command_cogs.append((cog_class_name, True))
                        self.logger.debug(f"✅ {cog_class_name} loaded successfully")
                    else:
                        command_cogs.append((cog_class_name, False))
                        self.logger.error(f"❌ {cog_class_name} setup completed but cog not found")
                else:
                    command_cogs.append((cog_class_name, False))
                    self.logger.error(f"❌ {cog_class_name} module missing setup function")
                    
            except Exception as cog_error:
                command_cogs.append((cog_class_name, False))
                self.logger.error(f"❌ Failed to load {cog_class_name}: {cog_error}", exc_info=True)
        
        # Count total registered commands across all cogs
        total_commands = 0
        for cog in self.cogs.values():
            total_commands += len([cmd for cmd in cog.get_commands()])
        
        # Generate Rich visual report
        log_commands_setup(
            self.console,
            command_modules,
            command_cogs,
            total_commands
        )
        
        # Log summary at appropriate level
        successful_modules = sum(1 for _, success in command_modules if success)
        failed_modules = sum(1 for _, success in command_modules if not success)
        successful_cogs = sum(1 for _, success in command_cogs if success)
        failed_cogs = sum(1 for _, success in command_cogs if not success)
        
        if failed_modules > 0 or failed_cogs > 0:
            self.logger.warning(f"⚠️ Command setup completed with failures: {failed_modules + failed_cogs} failed")
        else:
            self.logger.info(f"✅ Command setup completed successfully: {successful_modules + successful_cogs} loaded")

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
