"""Reminder commands: !remindme, !reminders, !timezone.
[CA][REH][IV].

``!remindme to post a fit pic on may 2027`` pings the author in this channel
on 1 May 2027 at 09:00 *their* time. The timezone is asked for once and kept
(``!timezone London``). The same reminders can be made in plain words by
addressing the bot ("@bot remind me to ... friday"); see bot/reminders/chat.py.
"""

from __future__ import annotations

from discord.ext import commands

from bot.config import load_config
from bot.core.output import safe_reply
from bot.reminders import messages
from bot.reminders.chat import origin_from, set_zone_reply
from bot.reminders.scheduler import ReminderScheduler
from bot.reminders.service import DEFAULT_MAX_PER_USER, ReminderService
from bot.reminders.store import DEFAULT_PATH, ReminderStore
from bot.utils.logging import get_logger

logger = get_logger(__name__)

CANCEL_ALL_WORDS = frozenset({"all", "everything", "*"})


def build_service(cfg: dict) -> ReminderService:
    store = ReminderStore(str(cfg.get("REMINDERS_FILE") or DEFAULT_PATH))
    store.load()
    return ReminderService(store, max_per_user=int(cfg.get("REMINDERS_MAX_PER_USER") or DEFAULT_MAX_PER_USER))


class ReminderCommands(commands.Cog):
    """Personal reminders that ping you at your own local time."""

    def __init__(self, bot: commands.Bot, service: ReminderService | None = None) -> None:
        self.bot = bot
        self.service = service or build_service(load_config())
        self.scheduler = ReminderScheduler(bot, self.service)
        # The router and the set_reminder tool find the service here. [CA]
        setattr(bot, "reminder_service", self.service)

    async def cog_load(self) -> None:
        self.scheduler.start()

    async def cog_unload(self) -> None:
        await self.scheduler.stop()
        if getattr(self.bot, "reminder_service", None) is self.service:
            setattr(self.bot, "reminder_service", None)

    @commands.command(name="remindme", aliases=["remind", "reminder", "setreminder"], help="Set a reminder. Usage: !remindme to post a fit pic on may 2027")
    @commands.cooldown(10, 60, type=commands.BucketType.user)
    async def remindme(self, ctx: commands.Context, *, text: str = "") -> None:
        if not text.strip():
            await safe_reply(ctx.message, messages.no_time())
            return
        outcome = await self.service.create(origin_from(ctx.message), text)
        await safe_reply(ctx.message, outcome.render())

    @commands.group(name="reminders", aliases=["myreminders"], invoke_without_command=True, help="List your reminders. !reminders cancel <id|all>")
    async def reminders(self, ctx: commands.Context) -> None:
        await safe_reply(ctx.message, messages.listing(self.service.list_for(ctx.author.id)))

    @reminders.command(name="cancel", aliases=["delete", "remove", "del", "rm"])
    async def reminders_cancel(self, ctx: commands.Context, target: str = "") -> None:
        await safe_reply(ctx.message, await self._cancel(ctx.author.id, target))

    @commands.command(name="unremind", aliases=["cancelreminder", "delreminder"], help="Cancel a reminder. Usage: !unremind <id|all>")
    async def unremind(self, ctx: commands.Context, target: str = "") -> None:
        await safe_reply(ctx.message, await self._cancel(ctx.author.id, target))

    async def _cancel(self, user_id: int, target: str) -> str:
        target = target.strip().lstrip("#").lower()
        if target in CANCEL_ALL_WORDS:
            ids = [r.id for r in self.service.list_for(user_id)]
            removed = [r for r in [await self.service.cancel(user_id, i) for i in ids] if r]
            return f"🗑️ cancelled {len(removed)} reminder(s)."
        if not target.isdigit():
            return "usage: `!reminders cancel <id>` (ids are in `!reminders`) or `!reminders cancel all`"
        removed = await self.service.cancel(user_id, int(target))
        return f"🗑️ cancelled reminder #{removed.id}: {removed.task or 'no message'}" if removed else f"you have no reminder #{target}."

    @commands.command(name="timezone", aliases=["tz", "settimezone", "settz", "mytimezone"], help="Set your timezone for reminders. Usage: !timezone London")
    async def timezone(self, ctx: commands.Context, *, zone: str = "") -> None:
        if not zone.strip():
            await safe_reply(ctx.message, messages.zone_current(self.service.timezone_of(ctx.author.id)))
            return
        await safe_reply(ctx.message, await set_zone_reply(self.service, ctx.author.id, zone))


async def setup(bot: commands.Bot) -> None:
    """Set up the ReminderCommands cog."""
    try:
        if bot.get_cog("ReminderCommands"):
            logger.warning("[Reminders Setup] ReminderCommands already loaded, removing old cog")
            await bot.remove_cog("ReminderCommands")
        await bot.add_cog(ReminderCommands(bot))
        loaded = bot.get_cog("ReminderCommands")
        names = [cmd.name for cmd in loaded.get_commands()] if loaded else []
        logger.info(f"✅ ReminderCommands cog loaded; commands: {names}")
    except Exception as e:
        logger.error(f"❌ Failed to set up ReminderCommands cog: {e}", exc_info=True)
        raise
