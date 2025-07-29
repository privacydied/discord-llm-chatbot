"""Core bot implementation for Discord LLM Chatbot."""

from __future__ import annotations
import asyncio
import os
from typing import TYPE_CHECKING, Optional

import discord
from discord.ext import commands

from bot.util.logging import get_logger
from bot.metrics import NullMetrics
from bot.memory import load_all_profiles

if TYPE_CHECKING:
    from bot.router import Router, BotAction
    from bot.tts import TTSManager


class LLMBot(commands.Bot):
    """Main bot class that extends the base Bot class with LLM capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(__name__)
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
        self.tts_manager: Optional[TTSManager] = None
        self.router: Optional[Router] = None
        self.background_tasks = []
        self._is_ready = asyncio.Event()

    async def setup_hook(self) -> None:
        """Asynchronous setup phase for the bot."""
        self.logger.info("Setting up bot...")
        try:
            from bot.metrics.prometheus import PrometheusMetrics
            self.metrics = PrometheusMetrics()
        except Exception:
            self.logger.warning("Prometheus metrics not available, using NullMetrics.")

        await self.load_profiles()
        self.setup_background_tasks()
        await self.setup_tts()
        await self.setup_router()
        await self.load_extensions()
        self.logger.info("Bot setup complete")

    async def on_ready(self):
        """Called when the bot is ready and connected to Discord."""
        if not self._is_ready.is_set():
            self.logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
            self.logger.info('Running setup_hook...')
            await self.setup_hook()
            self._is_ready.set()
            self.logger.info("Bot is ready to receive commands.")

    async def on_message(self, message: discord.Message):
        await self._is_ready.wait() # Wait until the bot is ready
        if message.author == self.user:
            return

        if (not message.content or not message.content.strip()) and not message.attachments:
            return

        guild_info = 'DM' if isinstance(message.channel, discord.DMChannel) else f"guild:{message.guild.id}"
        self.logger.info(
            f"Message received: msg_id:{message.id} author:{message.author.id} in:{guild_info} len:{len(message.content)}"
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

        try:
            if action.content or action.embeds or files:
                # Use message.reply() if the action was triggered by a reply
                if action.meta.get('is_reply'):
                    await message.reply(content=action.content, embeds=action.embeds, files=files)
                else:
                    await message.channel.send(content=action.content, embeds=action.embeds, files=files)
        except discord.errors.HTTPException as e:
            self.logger.error(f"Failed to send message: {e} (msg_id: {message.id})")

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

    async def close(self) -> None:
        """Clean up resources when the bot is shutting down."""
        self.logger.info("Bot is shutting down...")
        if self.memory_save_task and self.memory_save_task.is_running():
            self.memory_save_task.cancel()
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        if self.tts_manager:
            await self.tts_manager.close()
        await super().close()
        self.logger.info("Bot shutdown complete")

