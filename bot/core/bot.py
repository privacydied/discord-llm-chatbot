"""Core bot implementation for Discord LLM Chatbot."""

from __future__ import annotations
import asyncio
import os
import time
from typing import TYPE_CHECKING, Optional, List

import discord
from discord import Intents
from discord.ext import commands

from bot.util.logging import get_logger, init_logging
from bot.metrics import NullMetrics
from bot.tts import TTSManager
from bot.memory import load_all_profiles

if TYPE_CHECKING:
    from bot.router import BotAction


class LLMBot(commands.Bot):
    """Main bot class that extends the base Bot class with LLM capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_logging()
        self.logger = get_logger(__name__)
        try:
            from bot.metrics.prometheus import PrometheusMetrics
            self.metrics = PrometheusMetrics()
        except Exception:
            self.metrics = NullMetrics()
        
        self.config = {
            'tts': {
                'voice_model_path': os.getenv('TTS_MODEL_PATH'),
                'voice_style_path': os.getenv('TTS_VOICES_PATH'),
                'tokenizer_alias': os.getenv('TTS_TOKENISER', 'default')
            }
        }
        self.user_profiles = {}
        self.server_profiles = {}
        self.memory_save_task = None
        self.tts_manager = None
        self.background_tasks = []



    async def setup_hook(self) -> None:
        """Called when the bot is starting up."""
        self.logger.info("Setting up bot...", extra={"subsys": "core", "event": "setup_start"})
        
        # Load memory profiles
        await self.load_profiles()
        
        # Set up background tasks
        self.setup_background_tasks()
        
        # Set up TTS manager if configured
        await self.setup_tts()
        
        # Set up router
        await self.setup_router()
        
        # Load extensions
        await self.load_extensions()
        
        # Custom message handler is now implemented directly in on_message
        # No need to register separate listener
        
        self.logger.info("Bot setup complete", extra={"subsys": "core", "event": "setup_complete"})
        
    async def on_message(self, message: discord.Message):
        # Skip messages from self
        if message.author == self.user:
            return

        # Guard: Ensure message has content or attachments
        if (not message.content or len(message.content.strip()) == 0) and not message.attachments:
            self.logger.warning(
                "GUARD-FAIL",
                extra={
                    "subsys": "core",
                    "event": "empty_message",
                    "message_id": message.id,
                    "author_id": message.author.id
                }
            )
            return

        # Log incoming message
        channel_type = 'DM' if isinstance(message.channel, discord.DMChannel) else str(message.guild.id)
        self.logger.info(
            "EVENT-RECV",
            extra={
                "subsys": "core",
                "event": "message_received",
                "message_id": message.id,
                "guild_or_dm": channel_type,
                "author_id": message.author.id,
                "raw_length": len(message.content)
            }
        )

        # Get normalized prefixes
        prefixes = self.get_prefixes(message)

        # --- BEGIN DIAGNOSTIC LOGGING ---
        is_mentioned = self.user in message.mentions
        has_prefix = any(message.content.startswith(p) for p in prefixes)
        self.logger.info(
            "[DIAGNOSTIC] Pre-dispatch check",
            extra={
                "subsys": "core",
                "event": "pre_dispatch_check",
                "message_id": message.id,
                "is_mentioned": is_mentioned,
                "has_prefix": has_prefix,
                "prefixes": prefixes,
            }
        )
        # --- END DIAGNOSTIC LOGGING ---

        # In DMs, process the message directly. In guilds, check for mention/prefix.
        if isinstance(message.channel, discord.DMChannel) or is_mentioned or has_prefix:
            self.logger.info("[DIAGNOSTIC] Pre-dispatch check", extra={'subsys': 'bot', 'guild_id': message.guild.id if message.guild else 'DM', 'user_id': message.author.id, 'msg_id': message.id})
            action = await self.process_message(message)
            if action and action.has_payload:
                self.logger.info(f"SENDING-MSG: {action.content[:50]}...", extra={'subsys': 'bot', 'guild_id': message.guild.id if message.guild else 'DM', 'user_id': message.author.id, 'msg_id': message.id})
                await message.channel.send(content=action.content, embeds=action.embeds, files=action.files)
        else:
            self.logger.info(
                "Message ignored: No trigger (mention/prefix) in guild channel",
                extra={"subsys": "core", "event": "ignore_no_trigger", "message_id": message.id}
            )

    def get_prefixes(self, message: discord.Message) -> tuple[str, ...]:
        """Normalize command prefixes to tuple of strings"""
        # Guard: Ensure message is valid
        if not message or not isinstance(message, discord.Message):
            self.logger.error(
                "GUARD-FAIL",
                extra={
                    "subsys": "core",
                    "event": "invalid_message_object",
                    "error": "None or invalid message object"
                }
            )
            return ()

        # Handle callable prefix
        if callable(self.command_prefix):
            result = self.command_prefix(self, message)
        else:
            result = self.command_prefix
            
        # Normalize to tuple of strings
        if isinstance(result, str):
            return (result,)
        elif isinstance(result, (list, tuple)):
            return tuple(str(p) for p in result)
        else:
            self.logger.warning(
                "GUARD-FAIL",
                extra={
                    "subsys": "core",
                    "event": "invalid_prefix_type",
                    "type": type(result).__name__
                }
            )
            return ()

    async def load_profiles(self) -> None:
        """Load user and server memory profiles."""
        try:
            self.logger.info("Loading memory profiles...", extra={"subsys": "memory", "event": "load_start"})
            self.user_profiles, self.server_profiles = load_all_profiles()
            self.logger.info(
                f"Loaded {len(self.user_profiles)} user profiles and {len(self.server_profiles)} server profiles",
                extra={"subsys": "memory", "event": "load_complete"}
            )
        except Exception as e:
            self.logger.error(f"Failed to load profiles: {e}", exc_info=True, 
                             extra={"subsys": "memory", "event": "load_error"})

    def setup_background_tasks(self) -> None:
        """Set up background tasks for the bot."""
        try:
            from bot.tasks import setup_memory_save_task
            self.memory_save_task = setup_memory_save_task(self)
            self.memory_save_task.start()
        except Exception as e:
            self.logger.error(f"Failed to set up background tasks: {e}", exc_info=True,
                             extra={"subsys": "tasks", "event": "setup_error"})

    async def setup_tts(self) -> None:
        """Set up TTS manager if configured."""
        try:
            from bot.tts.interface import TTSManager
            self.tts_manager = TTSManager(self)
            self.logger.info("TTS manager initialized", extra={"subsys": "tts", "event": "setup_complete"})
        except Exception as e:
            self.logger.error(f"Failed to set up TTS manager: {e}", exc_info=True,
                             extra={"subsys": "tts", "event": "setup_error"})
            self.tts_manager = None

    async def setup_router(self) -> None:
        """Set up message router."""
        try:
            from bot.router import setup_router
            # setup_router returns a Router instance directly, no need to await
            self.router = setup_router(self)
            self.logger.info("Router initialized", extra={"subsys": "router", "event": "setup_complete"})
        except Exception as e:
            self.logger.error(f"Failed to set up router: {e}", exc_info=True,
                             extra={"subsys": "router", "event": "setup_error"})

    async def load_extensions(self) -> None:
        """Load command extensions."""
        try:
            from bot.commands import setup_commands
            await setup_commands(self)
            self.logger.info("Commands loaded", extra={"subsys": "commands", "event": "setup_complete"})
        except Exception as e:
            self.logger.error(f"Failed to load extensions: {e}", exc_info=True,
                             extra={"subsys": "commands", "event": "setup_error"})

    async def close(self) -> None:
        """Clean up resources when the bot is shutting down."""
        self.logger.info("Bot is shutting down...", extra={"subsys": "core", "event": "shutdown_start"})
        
        # Cancel background tasks
        if self.memory_save_task and self.memory_save_task.is_running():
            self.memory_save_task.cancel()
        
        # Cancel any other background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Close TTS manager if it exists
        if self.tts_manager:
            await self.tts_manager.close()
        
        await super().close()
        self.logger.info("Bot shutdown complete", extra={"subsys": "core", "event": "shutdown_complete"})

    async def process_message(self, message: discord.Message) -> Optional[BotAction]:
        """Calls the router to dispatch the message and returns the resulting action."""
        self.logger.debug(f"Routing message {message.id}...", extra={"message_id": message.id})
        try:
            action = await self.router.dispatch_message(message)
            if action and action.has_payload:
                self.logger.info(f"Router returned action for message {message.id}", extra={"message_id": message.id})
            else:
                self.logger.info(f"Router ignored message {message.id}", extra={"message_id": message.id})
            return action
        except Exception as e:
            self.logger.error(
                f"Error dispatching message {message.id}: {e}",
                exc_info=True,
                extra={"message_id": message.id, "event": "dispatch_error"}
            )
            return None

    @commands.command(name='testflow')
    async def test_flow(self, ctx):
        """Test command to verify message flow"""
        self.logger.debug(" [TEST-FLOW] Command received")
        await ctx.send(" Message flow working correctly!")
        self.logger.debug("[TEST-FLOW] Response sent")
