"""
Search commands for online web search using pluggable providers.
[CA][REH][IV][PA]
"""
from __future__ import annotations

from typing import Optional, List

import discord
from discord.ext import commands

from ..util.logging import get_logger
from ..config import load_config
from ..search.factory import get_search_provider
from ..search.types import SearchQueryParams, SafeSearch, SearchResult

logger = get_logger(__name__)

# Discord embed limits
DISCORD_EMBED_DESCRIPTION_LIMIT = 4096
DISCORD_EMBED_FIELD_VALUE_LIMIT = 1024
DISCORD_EMBED_TOTAL_LIMIT = 6000


def _truncate(text: Optional[str], limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 6)] + "... ⚠️"


def _format_result_field(result: SearchResult) -> str:
    parts: List[str] = []
    # URL first for easy click
    parts.append(result.url)
    if result.snippet:
        parts.append("")
        parts.append(_truncate(result.snippet, DISCORD_EMBED_FIELD_VALUE_LIMIT - len(result.url) - 10))
    return "\n".join(parts)


class SearchCommands(commands.Cog):
    """Commands for online web search via configured provider."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.cfg = load_config()
        logger.info("[Search] ✔ SearchCommands initialized")

    @commands.command(name="search", help="Search the web. Usage: !search <query>")
    async def search(self, ctx: commands.Context, *, query: Optional[str] = None):  # type: ignore[override]
        """Execute a web search using the configured provider and return top results."""
        try:
            if not query or not query.strip():
                embed = discord.Embed(
                    title="🔎 Web Search",
                    description="Usage: `!search <query>`",
                    color=discord.Color.blurple(),
                )
                await ctx.send(embed=embed)
                return

            provider_name = self.cfg.get("SEARCH_PROVIDER", "ddg")
            max_results = self.cfg.get("SEARCH_MAX_RESULTS", 5)
            locale = self.cfg.get("SEARCH_LOCALE") or None

            # Map safe level
            safe_str = str(self.cfg.get("SEARCH_SAFE", "moderate")).lower()
            try:
                safesearch = SafeSearch(safe_str)
            except Exception:
                safesearch = SafeSearch.MODERATE

            timeout_ms = int(self.cfg.get("DDG_TIMEOUT_MS", 5000)) if provider_name == "ddg" else int(
                self.cfg.get("CUSTOM_SEARCH_TIMEOUT_MS", 8000)
            )

            params = SearchQueryParams(
                query=query.strip(),
                max_results=max_results,
                safesearch=safesearch,
                locale=locale,
                timeout_ms=timeout_ms,
            )

            provider = get_search_provider()
            results = await provider.search(params)

            if not results:
                embed = discord.Embed(
                    title="🔎 Web Search",
                    description=f"No results found for: `{_truncate(query.strip(), 256)}`",
                    color=discord.Color.orange(),
                )
                embed.set_footer(text=f"Provider: {provider_name}")
                await ctx.send(embed=embed)
                return

            embed = discord.Embed(
                title="🔎 Web Search Results",
                description=f"Query: `{_truncate(query.strip(), 256)}`",
                color=discord.Color.green(),
            )
            embed.set_footer(text=f"Provider: {provider_name} • Safe: {safesearch.value}")

            # Add top results as fields
            for idx, r in enumerate(results, start=1):
                name = _truncate(f"{idx}. {r.title}", 256)
                value = _format_result_field(r)
                embed.add_field(name=name, value=_truncate(value, DISCORD_EMBED_FIELD_VALUE_LIMIT), inline=False)

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"[Search] ❌ Search command failed: {e}", exc_info=True)
            embed = discord.Embed(
                title="❌ Search Error",
                description="An error occurred while performing the search. Please try again later.",
                color=discord.Color.red(),
            )
            await ctx.send(embed=embed)


async def setup(bot: commands.Bot):
    """Set up Search commands cog."""
    try:
        logger.info("[Search Setup] Initializing SearchCommands cog...")
        existing = bot.get_cog("SearchCommands")
        if existing:
            logger.warning("[Search Setup] SearchCommands already loaded, removing old cog")
            await bot.remove_cog("SearchCommands")

        cog = SearchCommands(bot)
        await bot.add_cog(cog)

        loaded = bot.get_cog("SearchCommands")
        if loaded:
            logger.info("✅ SearchCommands cog loaded successfully")
            names = [cmd.name for cmd in loaded.get_commands()]
            logger.info(f"[Search Setup] Registered commands: {names}")
        else:
            logger.error("❌ SearchCommands failed to load - cog not found after adding")
    except Exception as e:
        logger.error(f"❌ Failed to set up SearchCommands cog: {e}", exc_info=True)
        raise
