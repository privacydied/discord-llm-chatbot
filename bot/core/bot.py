"""Core bot implementation for Discord LLM Chatbot."""

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable

import discord
from discord.ext import commands, tasks

from bot.core.client import Bot
from bot.logger import get_logger
from bot.memory import load_all_profiles


class LLMBot(Bot):
    """Main bot class that extends the base Bot class with LLM capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(__name__)
        self.config = {}
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
        
        self.logger.info("Bot setup complete", extra={"subsys": "core", "event": "setup_complete"})

    async def load_profiles(self) -> None:
        """Load user and server memory profiles."""
        try:
            self.logger.info("Loading memory profiles...", extra={"subsys": "memory", "event": "load_start"})
            self.user_profiles, self.server_profiles = await asyncio.to_thread(load_all_profiles)
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
            from bot.tts import TTSManager
            self.tts_manager = TTSManager(self)
            self.logger.info("TTS manager initialized", extra={"subsys": "tts", "event": "setup_complete"})
        except Exception as e:
            self.logger.error(f"Failed to set up TTS manager: {e}", exc_info=True,
                             extra={"subsys": "tts", "event": "setup_error"})

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
