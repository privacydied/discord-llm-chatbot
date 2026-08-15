"""News digest command.
[CA][REH][IV][PA][CMV].

Explicit counterpart to the ambient "what's happening in the news" path in the
router: same licensed headline source, invoked deliberately.
"""

from __future__ import annotations

import discord
from discord.ext import commands

from bot.config import load_config
from bot.core.output import safe_send
from bot.news.headlines import DEFAULT_LIMIT, fetch_headlines
from bot.news.intent import DAYS_GENERAL_DEFAULT, DAYS_TOPIC_DEFAULT
from bot.utils.logging import get_logger

logger = get_logger(__name__)

# Discord embed ceilings, enforced here because the outbound sanitizer does not
# resize embeds. [CMV][REH]
EMBED_TITLE_LIMIT = 256
EMBED_FIELD_NAME_LIMIT = 256
EMBED_FIELD_VALUE_LIMIT = 1024
MAX_FIELDS = 8


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _build_embed(headlines: list, topic: str | None, days: int) -> discord.Embed:
    window = "last 24 hours" if days <= 1 else f"last {days} days"
    title = f"📰 Headlines — {topic}" if topic else "📰 Headlines"
    embed = discord.Embed(
        title=_truncate(title, EMBED_TITLE_LIMIT),
        description=f"The Guardian · {window} · {len(headlines)} stories",
        color=discord.Color.dark_teal(),
    )
    for headline in headlines[:MAX_FIELDS]:
        name = _truncate(headline.title, EMBED_FIELD_NAME_LIMIT)
        body = headline.summary.strip()
        value = f"{body}\n{headline.url}" if body else headline.url
        embed.add_field(name=name, value=_truncate(value, EMBED_FIELD_VALUE_LIMIT), inline=False)
    return embed


class NewsCommands(commands.Cog):
    """Fetch current headlines from licensed sources."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.cfg = load_config()
        logger.info("[News] ✔ NewsCommands initialized")

    @commands.command(name="news", help="Current headlines. Usage: !news [topic]")
    @commands.cooldown(5, 60, type=commands.BucketType.user)
    async def news(self, ctx: commands.Context, *, topic: str | None = None) -> None:  # type: ignore[override]
        """Post recent headlines, optionally filtered to a topic."""
        topic = (topic or "").strip() or None
        days = DAYS_TOPIC_DEFAULT if topic else DAYS_GENERAL_DEFAULT

        if not (self.cfg.get("GUARDIAN_API_KEY") or "").strip():
            await safe_send(
                ctx.channel,
                "📰 News lookup isn't configured — set `GUARDIAN_API_KEY` in `.env` to enable it.",
            )
            return

        try:
            headlines = await fetch_headlines(
                topic,
                cfg=self.cfg,
                days=days,
                limit=self.cfg.get("NEWS_DIGEST_LIMIT", DEFAULT_LIMIT),
            )
        except Exception as exc:  # [REH]
            logger.error(f"[News] fetch failed topic={topic}: {exc}", exc_info=True)
            await safe_send(ctx.channel, "⚠️ Couldn't reach the news service just now. Try again shortly.")
            return

        if not headlines:
            subject = f" on **{topic}**" if topic else ""
            await safe_send(ctx.channel, f"📰 Nothing found{subject} in the {days}-day window.")
            return

        logger.info(f"[News] served topic={topic} count={len(headlines)}")
        await safe_send(ctx.channel, embed=_build_embed(headlines, topic, days))


async def setup(bot: commands.Bot) -> None:
    """Set up News commands cog."""
    try:
        logger.info("[News Setup] Initializing NewsCommands cog...")
        existing = bot.get_cog("NewsCommands")
        if existing:
            logger.warning("[News Setup] NewsCommands already loaded, removing old cog")
            await bot.remove_cog("NewsCommands")

        cog = NewsCommands(bot)
        await bot.add_cog(cog)

        loaded = bot.get_cog("NewsCommands")
        if loaded:
            names = [cmd.name for cmd in loaded.get_commands()]
            logger.info(f"✅ NewsCommands cog loaded successfully; commands: {names}")
        else:
            logger.error("❌ NewsCommands failed to load - cog not found after adding")
    except Exception as e:
        logger.error(f"❌ Failed to set up NewsCommands cog: {e}", exc_info=True)
        raise
