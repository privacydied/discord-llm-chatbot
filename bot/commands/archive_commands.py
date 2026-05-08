"""Discord commands for the raw server archive."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import discord
from discord.ext import commands

from bot.server_archive import get_server_archive_service, get_server_archive_status, search_archive, start_server_archive_service
from bot.server_archive.search import sanitize_snippet


class ArchiveCommands(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._service: Any | None = None

    async def cog_load(self) -> None:
        self._service = await start_server_archive_service(self.bot)

    async def _service_or_raise(self):
        if self._service is None:
            self._service = await get_server_archive_service(self.bot)
        return self._service

    async def _archive_enabled(self, ctx: commands.Context) -> bool:
        service = await self._service_or_raise()
        if getattr(service, "enabled", False):
            return True
        await ctx.reply("Server archive is disabled on this bot.", mention_author=False)
        return False

    def _archive_admin_allowed(self, ctx: commands.Context) -> bool:
        admin_only = True
        if self._service is not None:
            admin_only = bool(getattr(self._service, "admin_only", True))
        if not admin_only:
            return True
        perms = getattr(ctx.author, "guild_permissions", None)
        return bool(getattr(perms, "administrator", False))

    async def _require_permission(self, ctx: commands.Context) -> bool:
        if self._archive_admin_allowed(ctx):
            return True
        await ctx.reply("Archive commands are admin-only on this server.", mention_author=False)
        return False

    def _short_time(self, iso_value: str | None) -> str:
        if not iso_value:
            return "unknown"
        try:
            return datetime.fromisoformat(iso_value).astimezone().strftime("%Y-%m-%d %H:%M")
        except Exception:
            return iso_value

    @commands.guild_only()
    @commands.command(name="archive-status")
    async def archive_status(self, ctx: commands.Context) -> None:
        if not await self._require_permission(ctx):
            return
        status = await get_server_archive_status(guild_id=str(ctx.guild.id))
        counts = status.get("counts", {})
        sync_states = status.get("sync_states", [])
        last_sync = sync_states[0].get("last_synced_at") if sync_states else None
        current_state = sync_states[0].get("status") if sync_states else "idle"
        lines = [
            f"enabled: {status.get('enabled')} paused: {status.get('paused')} db: {status.get('db_path')}",
            f"queue: {status.get('queue_size', 0)}/{status.get('queue_max', 0)} dropped:{status.get('stats', {}).get('dropped', 0)} batches:{status.get('batch_size', 0)}",
            f"messages: {counts.get('messages', 0)} indexed:{counts.get('indexed_messages', counts.get('messages', 0))} guilds:{counts.get('guilds', 0)} channels:{counts.get('channels', 0)} threads:{counts.get('threads', 0)}",
            f"sync: {current_state} last: {self._short_time(last_sync)} running: {status.get('sync_running')}",
        ]
        await ctx.reply("\n".join(lines), mention_author=False)

    @commands.guild_only()
    @commands.command(name="archive-search")
    async def archive_search(self, ctx: commands.Context, *, query: str) -> None:
        if not await self._require_permission(ctx):
            return
        if not await self._archive_enabled(ctx):
            return
        service = await self._service_or_raise()
        limit = min(10, max(1, int(getattr(service, "search_limit", 10))))
        results = await search_archive(query, guild_id=str(ctx.guild.id), limit=limit)
        if not results:
            await ctx.reply("No archive matches found.", mention_author=False)
            return
        lines = []
        for result in results[:limit]:
            channel_name = result.channel_name or f"#{result.channel_id}"
            author_name = result.author_name or result.author_id
            snippet = sanitize_snippet(result.snippet or result.clean_content or result.content, limit=180)
            url = result.jump_url or ""
            lines.append(
                f"[{self._short_time(result.created_at)}] {channel_name} · {author_name}\n{snippet}"
                + (f"\n{url}" if url else "")
            )
        payload = "\n\n".join(lines)
        if len(payload) > 1900:
            payload = payload[:1890] + "…"
        await ctx.reply(payload, mention_author=False)

    @commands.guild_only()
    @commands.command(name="archive-sync")
    async def archive_sync(self, ctx: commands.Context) -> None:
        if not await self._require_permission(ctx):
            return
        if not await self._archive_enabled(ctx):
            return
        service = await self._service_or_raise()
        guild = ctx.guild
        status = await service.get_status(guild_id=str(guild.id))
        if status.get("sync_running"):
            await ctx.reply("Archive sync is already running for this guild.", mention_author=False)
            return
        asyncio.create_task(service.sync_guild(guild), name=f"archive-sync-{guild.id}")
        await ctx.reply("Archive sync started for this guild.", mention_author=False)

    @commands.guild_only()
    @commands.command(name="archive-sync-channel")
    async def archive_sync_channel(
        self, ctx: commands.Context, channel: discord.TextChannel | discord.Thread | None = None
    ) -> None:
        if not await self._require_permission(ctx):
            return
        if not await self._archive_enabled(ctx):
            return
        service = await self._service_or_raise()
        target = channel or ctx.channel
        if getattr(target, "guild", None) is not ctx.guild:
            await ctx.reply("Archive sync targets must belong to this guild.", mention_author=False)
            return
        key = f"{ctx.guild.id}:{getattr(target, 'id', '')}"
        if key in service._channel_sync_tasks and not service._channel_sync_tasks[key].done():
            await ctx.reply("Archive sync is already running for this channel/thread.", mention_author=False)
            return
        asyncio.create_task(service.sync_channel(target), name=f"archive-sync-channel-{key}")
        await ctx.reply("Archive sync started for the target channel/thread.", mention_author=False)

    @commands.guild_only()
    @commands.command(name="archive-pause")
    async def archive_pause(self, ctx: commands.Context) -> None:
        if not await self._require_permission(ctx):
            return
        if not await self._archive_enabled(ctx):
            return
        service = await self._service_or_raise()
        service.pause()
        await ctx.reply("Archive background full sync paused.", mention_author=False)

    @commands.guild_only()
    @commands.command(name="archive-resume")
    async def archive_resume(self, ctx: commands.Context) -> None:
        if not await self._require_permission(ctx):
            return
        if not await self._archive_enabled(ctx):
            return
        service = await self._service_or_raise()
        service.resume()
        await ctx.reply("Archive background full sync resumed.", mention_author=False)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(ArchiveCommands(bot))
