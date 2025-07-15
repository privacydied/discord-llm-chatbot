"""
Defines the core LLMBot class.
"""
import discord

from bot.config import load_config
from bot.logger import get_logger
from bot.router import setup_router
from bot.tts import TTSManager
from bot.commands import setup_commands
from bot.tasks import spawn_background_tasks
from bot.memory import load_all_profiles, load_all_server_profiles
from bot.events import cache_maintenance_task
from .client import Bot


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

            # Initialize router
            setup_router(self)
            self.logger.info("Router initialized.", extra={'subsys': 'core', 'event': 'router_init'})

            # Register all commands
            await setup_commands(self)
            self.logger.info("Commands registered.", extra={'subsys': 'core', 'event': 'commands_registered'})

            # Start background tasks
            spawn_background_tasks(self)
            self.logger.info("Background tasks spawned.", extra={'subsys': 'core', 'event': 'tasks_spawned'})

        except Exception as e:
            self.logger.critical(f"Error during setup hook: {e}", exc_info=True, extra={'subsys': 'core', 'event': 'setup_hook_fail'})
            await self.close()

        self.logger.info("--- Bot Setup Hook Complete ---", extra={'subsys': 'core', 'event': 'setup_hook_pass'})

    async def on_ready(self):
        """Called when the bot is done preparing the data received from Discord."""
        self.logger.info(f'Logged in as {self.user} (ID: {self.user.id})', extra={'subsys': 'core', 'event': 'login_success', 'user_id': self.user.id})
        self.logger.info('------', extra={'subsys': 'core', 'event': 'separator'})

        # Load profiles after bot is ready
        await self._load_profiles()

        # Start cache maintenance task
        cache_maintenance_task.start(self)

    async def _load_profiles(self):
        """Load all user and server profiles."""
        self.logger.info("Loading user and server profiles...", extra={'subsys': 'core', 'event': 'profile_load_start'})
        try:
            await load_all_profiles()
            await load_all_server_profiles()
            self.logger.info("Profiles loaded successfully.", extra={'subsys': 'core', 'event': 'profile_load_pass'})
        except Exception as e:
            self.logger.error(f"Error loading profiles: {e}", exc_info=True, extra={'subsys': 'core', 'event': 'profile_load_fail'})

