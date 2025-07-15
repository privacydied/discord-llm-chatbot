"""
Defines the core LLMBot class.
"""
import discord

from .client import Bot
from bot.config import load_config
from bot.memory import load_all_profiles, load_all_server_profiles
from bot.logger import get_logger
from bot.router import setup_router
from bot.events import setup as setup_events
from bot.tasks import spawn_background_tasks
from bot.tts import TTSManager


class LLMBot(Bot):
    """Minimal bot class with bootstrap functionality only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.startup_time = None
        self.config = load_config()
        self.logger = get_logger(self.__class__.__name__)

    async def setup_hook(self) -> None:
        """Bootstrap setup - load cogs and start services."""
        self.logger.info("--- Starting Bot Setup Hook ---", extra={'subsys': 'core', 'event': 'setup_hook_start'})
        self.startup_time = discord.utils.utcnow()

        try:
            # Initialize TTS Manager
            self.tts_manager = TTSManager(self)
            self.logger.info("TTS Manager initialized.", extra={'subsys': 'core', 'event': 'tts_manager_init'})

            # Set up the router
            self.router = setup_router(self)
            self.logger.info("Router initialized.", extra={'subsys': 'core', 'event': 'router_init'})

            # Register all commands
            from bot.commands import setup_commands
            await setup_commands(self)
            self.logger.info("Commands registered.", extra={'subsys': 'core', 'event': 'commands_registered'})

            # Spawn background tasks
            await spawn_background_tasks(self)
            self.logger.info("Background tasks spawned.", extra={'subsys': 'core', 'event': 'tasks_spawned'})

            # Register event handlers
            await setup_events(self)
            self.logger.info("Event handlers registered.", extra={'subsys': 'core', 'event': 'events_registered'})

        except Exception as e:
            self.logger.critical(f"Error during setup hook: {e}", exc_info=True, extra={'subsys': 'core', 'event': 'setup_hook_fail'})
            await self.close()

        self.logger.info("--- Bot Setup Hook Complete ---", extra={'subsys': 'core', 'event': 'setup_hook_pass'})

    async def on_message(self, message: discord.Message) -> None:
        """Primary gateway for all incoming messages."""
        if message.author.bot:
            return  # Ignore messages from other bots

        # Log every message received that is not from a bot
        self.logger.debug(
            f"Message received: {message.id}",
            extra={'subsys': 'events', 'event': 'on_message', 'msg_id': message.id}
        )

        # Pass the message to the router for processing
        await self.router.dispatch_message(message)

    async def on_ready(self):
        """Called when the bot is done preparing the data received from Discord."""
        self.logger.info(f'Logged in as {self.user} (ID: {self.user.id})', extra={'subsys': 'core', 'event': 'login_success', 'user_id': self.user.id})
        self.logger.info('------', extra={'subsys': 'core', 'event': 'separator'})

        # Load profiles after bot is ready
        await self._load_profiles()



    async def _load_profiles(self):
        """Load all user and server profiles."""
        self.logger.info("Loading user and server profiles...", extra={'subsys': 'core', 'event': 'profile_load_start'})
        try:
            load_all_profiles()
            load_all_server_profiles()
            self.logger.info("Profiles loaded successfully.", extra={'subsys': 'core', 'event': 'profile_load_pass'})
        except Exception as e:
            self.logger.error(f"Error loading profiles: {e}", exc_info=True, extra={'subsys': 'core', 'event': 'profile_load_fail'})

