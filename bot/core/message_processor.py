"""Message processor: extracts per-user queue, dedup guard, and
single-message orchestration logic from LLMBot into a standalone class.

LLMBot delegates to this class in on_message while keeping the rest of its
state (router, tts_manager, config, etc) accessible via the bot reference.
"""

from __future__ import annotations

import asyncio
import collections
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Dict

import discord

from bot.utils.logging import get_logger

if TYPE_CHECKING:
    from .bot import LLMBot

logger = get_logger(__name__)

# Max entries in the FIFO dedup guard before oldest are evicted
_DEDUP_MAX = 1000

# Per-user queue idle timeout (seconds)
_QUEUE_TIMEOUT = 300


class MessageProcessor:
    """Owns the per-user queue machinery, dedup guard, and
    _process_single_message orchestration.

    Created by LLMBot during setup_hook.  All public methods are
    async-safe and do not block the event loop.
    """

    def __init__(self, bot: LLMBot) -> None:
        self.bot = bot
        self.logger = logger

        # Per-user message queues and processor tasks
        self._user_queues: Dict[str, asyncio.Queue] = {}
        self._user_processors: Dict[str, asyncio.Task] = {}

        # Dedup guard – OrderedDict for FIFO eviction
        self._processed_messages: collections.OrderedDict = collections.OrderedDict()
        self._dispatch_lock = asyncio.Lock()

        self._typing_suppressed_until: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public entry point – called from LLMBot.on_message
    # ------------------------------------------------------------------

    async def on_message(self, message: discord.Message) -> None:
        """Dedup-check, archive-enqueue, then route the message."""
        author = getattr(message, "author", None)
        try:
            author_is_bot = bool(getattr(author, "bot", False)) if author else False
        except Exception:
            author_is_bot = True
        try:
            author_is_self = bool(
                author
                and getattr(author, "id", None)
                == getattr(getattr(self.bot, "user", None), "id", None)
            )
        except Exception:
            author_is_self = False
        if author_is_bot or author_is_self:
            return

        # Dedup guard under lock
        async with self._dispatch_lock:
            if message.id in self._processed_messages:
                self.logger.warning(
                    f"Duplicate dispatch prevented for msg_id: {message.id}"
                )
                return
            while len(self._processed_messages) >= _DEDUP_MAX:
                self._processed_messages.popitem(last=False)
            self._processed_messages[message.id] = True

        # Best-effort archive enqueue (fire-and-forget)
        await self._archive_enqueue(message)

        # Skip empty messages
        if (
            not message.content or not message.content.strip()
        ) and not message.attachments:
            return

        # Alert-session suppression — early return gate
        await self._alert_suppression_gate(message)

        return True  # Message passed all early checks; caller will enqueue

    async def enqueue(self, message: discord.Message) -> None:
        """Queue a message for per-user processing.

        This is the public method called by LLMBot when all gate checks
        have passed.
        """
        user_id = str(getattr(message.author, "id", None))
        self._get_queue(user_id).put_nowait(message)
        await self._ensure_user_processor(user_id)

    # ------------------------------------------------------------------
    # Per-user queue + processor
    # ------------------------------------------------------------------

    def _get_queue(self, user_id: str) -> asyncio.Queue:
        """Get or create the message queue for *user_id*."""
        if user_id not in self._user_queues:
            self._user_queues[user_id] = asyncio.Queue()
        return self._user_queues[user_id]

    async def _ensure_user_processor(self, user_id: str) -> None:
        """Ensure a background task drains the queue for *user_id*."""
        if (
            user_id not in self._user_processors
            or self._user_processors[user_id].done()
        ):
            self._user_processors[user_id] = asyncio.create_task(
                self._process_user_messages(user_id)
            )

    async def _process_user_messages(self, user_id: str) -> None:
        """Drain the queue for one user, one message at a time."""
        queue = self._get_queue(user_id)

        try:
            while True:
                try:
                    message = await asyncio.wait_for(
                        queue.get(), timeout=_QUEUE_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    break

                try:
                    async with self._optional_typing(message.channel):
                        await self.bot._process_single_message(message)
                except Exception:
                    self.logger.error(
                        f"Error processing message {message.id} for user {user_id}",
                        exc_info=True,
                    )
                finally:
                    queue.task_done()

        except Exception:
            self.logger.error(
                f"User message processor for {user_id} failed", exc_info=True
            )
        finally:
            self._user_processors.pop(user_id, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _archive_enqueue(self, message: discord.Message) -> None:
        archive_service = getattr(self.bot, "archive_service", None)
        if archive_service is None or not getattr(
            archive_service, "enabled", True
        ):
            return

        async def _enqueue() -> None:
            await archive_service.enqueue_live_message(message)

        task = asyncio.create_task(
            _enqueue(), name=f"server_archive_enqueue_{message.id}"
        )
        task.add_done_callback(
            lambda t: (
                t.exception()
                if t.done() and not t.cancelled()
                else None
            )
        )

    async def _alert_suppression_gate(self, message: discord.Message) -> None:
        """Suppress normal text flow while an admin alert session is active in DMs."""
        try:
            cog = self.bot.get_cog("AdminAlertCommands")
            if cog is not None and cog.alert_manager.is_dm_channel(
                message.channel
            ):
                active_session = cog.alert_manager.get_session(
                    message.author.id
                )
                if active_session is not None:
                    await self._handle_alert_dm(message, cog)
        except Exception:
            pass

    async def _handle_alert_dm(
        self, message: discord.Message, cog
    ) -> None:
        prefixes = await self.bot.get_prefix(message)
        if isinstance(prefixes, (list, tuple)):
            for p in prefixes:
                if p and message.content.startswith(p):
                    rest = (message.content[len(p):] or "").strip()
                    if rest.split(" ", 1)[0].lower() == "alert":
                        try:
                            await message.channel.send(
                                "⚠️ An alert session is already active."
                            )
                        except Exception:
                            pass
                        return
        elif prefixes:
            p = prefixes
            if message.content.startswith(p):
                rest = (message.content[len(p):] or "").strip()
                if rest.split(" ", 1)[0].lower() == "alert":
                    try:
                        await message.channel.send(
                            "⚠️ An alert session is already active."
                        )
                    except Exception:
                        pass
                    return

    @asynccontextmanager
    async def _optional_typing(self, channel):
        """Type indicator context manager, copied from LLMBot."""
        typing_factory = getattr(channel, "typing", None)
        if not callable(typing_factory):
            yield
            return

        channel_key = getattr(channel, "id", None) or id(channel)
        now = time.monotonic()
        if now < self._typing_suppressed_until.get(channel_key, 0.0):
            yield
            return

        ctx = None
        entered = False
        try:
            ctx = typing_factory()
            await ctx.__aenter__()
            entered = True
        except discord.HTTPException:
            self._typing_suppressed_until[channel_key] = now + 300.0
            yield
            return

        try:
            yield
        finally:
            if entered and ctx is not None:
                try:
                    await ctx.__aexit__(None, None, None)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Cancel all active user-processor tasks."""
        for task in list(self._user_processors.values()):
            task.cancel()
            try:
                await task
            except Exception:
                pass
        self._user_processors.clear()
