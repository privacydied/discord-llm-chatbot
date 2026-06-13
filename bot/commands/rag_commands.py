"""RAG system management commands for Discord bot."""

import asyncio
import contextlib
import re
from pathlib import Path

import discord
from discord.ext import commands

from bot.rag.config import get_rag_environment_info, validate_rag_environment
from bot.rag.hybrid_search import get_hybrid_search
from bot.server_features import is_server_feature_enabled
from bot.utils.logging import get_logger

logger = get_logger(__name__)


def is_admin_user():
    """Custom check that allows admin users in both guilds and DMs.

    Delegates to bot.core.permissions.is_admin_user for centralized logic.
    """
    from bot.core.permissions import is_admin_user as _check

    async def predicate(ctx):
        logger.info(f"[RAG Admin Check] User {ctx.author.id} ({ctx.author.name}) attempting RAG command")

        allowed = await _check(ctx.author, ctx.bot)

        # DM context: only owners/configured admins
        if isinstance(ctx.channel, discord.DMChannel) and not allowed:
            logger.warning(f"[RAG Admin Check] User {ctx.author.id} attempted RAG command in DM")
            return False

        return allowed

    return commands.check(predicate)


# Discord embed limits
DISCORD_EMBED_DESCRIPTION_LIMIT = 4096
DISCORD_EMBED_FIELD_VALUE_LIMIT = 1024
DISCORD_EMBED_TOTAL_LIMIT = 6000


def safe_embed_value(text: str, limit: int = DISCORD_EMBED_FIELD_VALUE_LIMIT) -> str:
    """Safely truncate text to fit within Discord embed limits."""
    if not text:
        return ""

    if len(text) <= limit:
        return text

    # Reserve space for ellipsis
    truncated = text[: limit - 3].rsplit(" ", 1)[0]
    return truncated + "..."


def chunk_text(text: str, chunk_size: int = DISCORD_EMBED_FIELD_VALUE_LIMIT) -> list[str]:
    """Split text into Discord-safe chunks."""
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])

    return chunks


class HybridSearchExtension:
    """Extension class to add hybrid_search method to BaseBot."""

    def __init__(self, bot) -> None:
        self.bot = bot
        bot.hybrid_search = bot.loop.create_task(self._init_hybrid_search())

    async def _init_hybrid_search(self):
        """Initialize hybrid search."""
        return await get_hybrid_search()


# Track invalidation state to prevent duplicate messages
_invalidation_in_progress = {}


class RAGCommands(commands.Cog):
    """Cog for RAG system management commands."""

    def __init__(self, bot) -> None:
        """Initialize the RAG cog.

        Args:
            bot: The bot instance

        """
        self.bot = bot
        self.hybrid_search = None
        self._init_task = None
        self.admin_ids = bot.admin_ids if hasattr(bot, "admin_ids") else []

        # Initialize hybrid search on startup
        self._init_task = asyncio.create_task(self._init_hybrid_search(), name="rag-init-hybrid-search")
        if self._init_task is not None:
            self._init_task.add_done_callback(lambda t: logger.error(f"hybrid search init failed: {t.exception()}", exc_info=t.exception()) if not t.cancelled() and t.exception() else None)

    async def _init_hybrid_search(self) -> None:
        """Initialize hybrid search."""
        try:
            self.hybrid_search = await get_hybrid_search()
            logger.info("[RAGCog] Hybrid search initialized")
        except Exception as e:
            logger.exception(f"[RAGCog] Failed to initialize hybrid search: {e}")

    async def cog_unload(self) -> None:
        """Cleanup on cog unload."""
        if self._init_task and not self._init_task.done():
            self._init_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._init_task
        if self.hybrid_search:
            await self.hybrid_search.close()

    @commands.group(name="rag", invoke_without_command=True)
    @is_admin_user()
    async def rag_group(self, ctx) -> None:
        """RAG system management commands."""
        embed = discord.Embed(
            title="🗂️  RAG System Management",
            description="Use `!rag <command>` to manage the RAG system.",
            color=discord.Color.blue(),
        )

        embed.add_field(
            name="Status Commands",
            value="`status` - Show RAG system status\n`test` - Run RAG system tests\n`reload` - Reload text index",
            inline=False,
        )

        embed.add_field(
            name="Management Commands",
            value="`clear` - Remove documents\n`wipe` - ⚠️ Wipe entire database\n`invalidate` - Invalidate collection",
            inline=False,
        )

        embed.add_field(
            name="Search Commands",
            value="`search <query>` - Search for documents\n`index <path>` - Index documents from directory",
            inline=False,
        )

        await ctx.send(embed=embed)

    @rag_group.command(name="status")
    @is_admin_user()
    async def rag_status(self, ctx) -> None:
        """Show the status of the RAG system."""
        try:
            # Get environment info
            env_info = get_rag_environment_info()

            # Create status embed
            embed = discord.Embed(title="🗂️  RAG System Status", color=discord.Color.blue())

            # Environment info
            env_text = f"**Mode:** {env_info.get('mode', 'unknown')}\n"
            if env_info.get("mode") == "dedicated":
                env_text += f"**Host:** {env_info.get('host', 'unknown')}\n"
            env_text += f"**Embeddings:** {env_info.get('embedding_model', 'unknown')}"

            embed.add_field(name="🔧 Environment", value=env_text, inline=False)

            # Collection stats
            if self.hybrid_search and self.hybrid_search.client:
                try:
                    stats = await self.hybrid_search.get_stats()
                    collection_stats = stats.get("collection_stats", {})

                    stats_text = (
                        f"**Documents:** {collection_stats.get('doc_count', 'N/A')}\n**Chunks:** {collection_stats.get('total_chunks', 'N/A')}\n**Avg chunks/doc:** {collection_stats.get('avg_chunks_per_doc', 'N/A'):.1f}"
                    )

                    embed.add_field(name="📊 Collection Stats", value=stats_text, inline=False)
                except Exception as e:
                    embed.add_field(
                        name="📊 Collection Stats",
                        value=f"Error getting stats: {e!s}",
                        inline=False,
                    )
            else:
                embed.add_field(
                    name="📊 Collection Stats",
                    value="Hybrid search not initialized",
                    inline=False,
                )

            # Health check
            health_emoji = "✅" if await validate_rag_environment() else "❌"
            embed.add_field(name="💚 Health Check", value=f"{health_emoji} Environment valid")

            await ctx.send(embed=embed)

        except Exception as e:
            logger.exception(f"[RAG Commands] Status command failed: {e}")
            await ctx.send(f"❌ **Error:** {safe_embed_value(str(e))}")

    @rag_group.command(name="test")
    @is_admin_user()
    async def rag_test(self, ctx) -> None:
        """Run comprehensive tests on the RAG system."""
        try:
            # Initial message
            message = await ctx.send("🧪 Running RAG system tests...")

            test_results = []

            # Test 1: Environment validation
            try:
                is_valid = await asyncio.to_thread(validate_rag_environment)
                test_results.append(
                    (
                        "✅" if is_valid else "❌",
                        "Environment",
                        "Valid" if is_valid else "Invalid configuration",
                    ),
                )
            except Exception as e:
                error_msg = safe_embed_value(str(e), 50)
                test_results.append(("❌", "Environment", f"Error: {error_msg}"))

            # Test 2: Hybrid search initialization
            try:
                if not self.hybrid_search:
                    self.hybrid_search = await get_hybrid_search()
                test_results.append(("✅", "Hybrid Search", "Successfully initialized"))
            except Exception as e:
                error_msg = safe_embed_value(str(e), 50)
                test_results.append(("❌", "Hybrid Search", f"Failed: {error_msg}"))

            # Test 3: Search functionality
            try:
                if self.hybrid_search:
                    results = await self.hybrid_search.search("test query", top_k=1)
                    result_count = len(results) if isinstance(results, list) else 0
                    test_results.append(("✅", "Search Functionality", f"Found {result_count} results"))
                else:
                    test_results.append(("⚠️", "Search Functionality", "Hybrid search not available"))
            except Exception as e:
                error_msg = safe_embed_value(str(e), 50)
                test_results.append(("❌", "Search Functionality", f"Failed: {error_msg}"))

            # Test 4: Collection stats
            try:
                search_engine_var = locals().get("search_engine")
                if search_engine_var is not None and hasattr(search_engine_var, "get_stats"):
                    stats = await search_engine_var.get_stats()
                    if "collection_stats" in stats:
                        chunks = stats["collection_stats"].get("total_chunks", 0)
                        test_results.append(("✅", "Collection Access", f"{chunks} chunks available"))
                    else:
                        test_results.append(("⚠️", "Collection Access", "No collection stats"))
                else:
                    test_results.append(("❌", "Collection Access", "Search engine not available"))
            except Exception as e:
                error_msg = safe_embed_value(str(e), 50)
                test_results.append(("❌", "Collection Access", f"Failed: {error_msg}"))

            # Update embed with test results
            embed = discord.Embed(title="🧪 RAG System Test Results", color=discord.Color.green())

            # Add test results to embed with safe truncation
            for icon, test_name, result in test_results:
                embed.add_field(
                    name=f"{icon} {safe_embed_value(test_name, 256)}",
                    value=safe_embed_value(result),
                    inline=False,
                )

            # Overall status
            passed_tests = sum(1 for icon, _, _ in test_results if icon == "✅")
            total_tests = len(test_results)

            summary_text = f"**Passed:** {passed_tests}/{total_tests} tests\n**Status:** {'✅ All systems operational' if passed_tests == total_tests else '⚠️ Some issues detected'}"

            embed.add_field(name="📊 Summary", value=safe_embed_value(summary_text), inline=False)

            await message.edit(embed=embed)

        except Exception as e:
            logger.exception(f"[RAG Commands] Test command failed: {e}")
            embed = discord.Embed(
                title="❌ Test Error",
                description=safe_embed_value(f"Test execution failed: {e!s}"),
                color=discord.Color.red(),
            )
            await ctx.send(embed=embed)

    @rag_group.command(name="wipe")
    @is_admin_user()
    async def rag_wipe(self, ctx) -> None:
        """Wipe the entire RAG database. ⚠️ This action is irreversible!"""
        try:
            # Check bot permissions first
            if ctx.guild:
                bot_member = ctx.guild.get_member(self.bot.user.id)
                if not bot_member:
                    await ctx.send("❌ **Error**: Bot member not found in guild.")
                    return

                required_perms = [
                    "send_messages",
                    "embed_links",
                    "add_reactions",
                    "read_message_history",
                    "use_external_emojis",
                ]

                missing_perms = []
                bot_perms = ctx.channel.permissions_for(bot_member)

                for perm in required_perms:
                    if not getattr(bot_perms, perm, False):
                        missing_perms.append(perm.replace("_", " ").title())

                if missing_perms:
                    perm_list = ", ".join(missing_perms)
                    error_msg = f"❌ **Missing Permissions**: The bot needs the following permissions: {perm_list}"
                    try:
                        await ctx.send(error_msg)
                    except discord.Forbidden:
                        logger.exception("[RAG Commands] Cannot send permission error message - missing Send Messages permission")
                    return

            # Verify hybrid search is initialized
            if not self.hybrid_search:
                self.hybrid_search = await get_hybrid_search()

            # Wipe the collection
            success = await self.hybrid_search.wipe_collection()

            if success:
                embed = discord.Embed(
                    title="🗑️ Database Wiped",
                    description="The RAG database has been successfully wiped.",
                    color=discord.Color.green(),
                )
            else:
                embed = discord.Embed(
                    title="❌ Wipe Failed",
                    description="Failed to wipe the RAG database.",
                    color=discord.Color.red(),
                )

            await ctx.send(embed=embed)

        except Exception as e:
            logger.exception(f"[RAG Commands] Wipe command failed: {e}")
            await ctx.send(f"❌ **Error:** {safe_embed_value(str(e))}")

    @rag_group.command(name="clear")
    @commands.cooldown(1, 120, type=commands.BucketType.user)
    @is_admin_user()
    async def rag_clear(self, ctx, *source_patterns: str) -> None:
        """Clear documents from the RAG database.

        Args:
            source_patterns: Optional source patterns to filter by

        """
        try:
            # Verify hybrid search is initialized
            if not self.hybrid_search:
                self.hybrid_search = await get_hybrid_search()

            # Create initial embed
            embed = discord.Embed(
                title="🗑️ Removing Documents",
                description=f"Removing documents matching: {', '.join(source_patterns) if source_patterns else 'all sources'}\n...",
                color=discord.Color.yellow(),
            )
            message = await ctx.send(embed=embed)

            # Clear documents
            if source_patterns:
                # Currently hybrid_search.clear() doesn't take source_patterns
                # But we should handle this case - for now, clear all
                await self.hybrid_search.clear()
                note = "Cleared all documents (source filtering not yet implemented)"
            else:
                await self.hybrid_search.clear()
                note = "Cleared all documents"

            embed = discord.Embed(
                title="✅ Documents Removed",
                description=f"Documents have been cleared.\n{note}",
                color=discord.Color.green(),
            )
            await message.edit(embed=embed)

        except Exception as e:
            logger.exception(f"[RAG Commands] Clear command failed: {e}")
            await ctx.send(f"❌ **Error:** {safe_embed_value(str(e))}")

    @rag_group.command(name="search")
    @is_admin_user()
    async def rag_search(self, ctx, *, query: str) -> None:
        """Search the RAG database.

        Args:
            query: The search query

        """
        try:
            # Verify hybrid search is initialized
            if not self.hybrid_search:
                self.hybrid_search = await get_hybrid_search()

            # Send searching message
            message = await ctx.send("🔍 Searching...")

            # Perform search
            results = await self.hybrid_search.search(query, top_k=10)

            if not results:
                embed = discord.Embed(
                    title="🔍 Search Results",
                    description="No results found.",
                    color=discord.Color.yellow(),
                )
                await message.edit(embed=embed)
                return

            # Create result embeds
            total_length = 0
            embeds = []

            # Create main results embed
            main_embed = discord.Embed(
                title=f"🔍 Results for '{safe_embed_value(query, 100)}'",
                color=discord.Color.blue(),
            )

            for i, result in enumerate(results, 1):
                content = safe_embed_value(result.get("content", ""), 500)
                meta = safe_embed_value(result.get("metadata", ""), 200)
                score = result.get("score", 0)

                field_name = f"Result {i} (Score: {score:.2f})"
                field_value = f"{content}\n\nMetadata: {meta}"

                # Check length limits
                field_length = len(field_name) + len(field_value)
                if total_length + field_length > DISCORD_EMBED_TOTAL_LIMIT - 1000:
                    break

                main_embed.add_field(
                    name=safe_embed_value(field_name, 256),
                    value=safe_embed_value(field_value),
                    inline=False,
                )
                total_length += field_length

            embeds.append(main_embed)

            await message.edit(embeds=embeds)

        except Exception as e:
            logger.exception(f"[RAG Commands] Search command failed: {e}")
            await ctx.send(f"❌ **Error:** {safe_embed_value(str(e))}")

    @commands.command(name="index")
    @commands.cooldown(2, 120, type=commands.BucketType.user)
    async def index_message_content(self, ctx, *, text: str | None = None) -> None:
        """Index the current message text, URLs, and supported attachments into RAG."""
        try:
            hybrid_search = self.hybrid_search or getattr(ctx.bot, "hybrid_search", None)
            if hybrid_search is None:
                hybrid_search = await get_hybrid_search()
            if hybrid_search is None:
                await ctx.send("❌ RAG indexing is unavailable right now.")
                return

            guild_id = getattr(getattr(ctx, "guild", None), "id", None)
            if guild_id is not None and not is_server_feature_enabled(guild_id, "rag"):
                await ctx.send("❌ RAG is disabled on this server.")
                return

            message = getattr(ctx, "message", None)
            attachments = list(getattr(message, "attachments", []) or [])
            raw_text = (text or "").strip()
            urls = re.findall(r"https?://\S+", raw_text)
            text_without_urls = re.sub(r"https?://\S+", " ", raw_text).strip()

            work_items = []
            if text_without_urls:
                work_items.append(("text", text_without_urls, None))
            for url in urls:
                work_items.append(("url", url, None))
            for attachment in attachments:
                work_items.append(("attachment", attachment, None))

            if not work_items:
                await ctx.send("Usage: `!index <text>` or attach files / paste a URL to index.")
                return

            guild_id = getattr(getattr(ctx, "guild", None), "id", None)
            user_id = getattr(getattr(ctx, "author", None), "id", None)
            message_id = getattr(message, "id", None)
            base_metadata = {
                "guild_id": guild_id,
                "channel_id": getattr(getattr(ctx, "channel", None), "id", None),
                "user_id": user_id,
                "message_id": message_id,
                "source": "discord",
            }

            queue_enabled = bool(getattr(getattr(hybrid_search, "_indexing_queue", None), "enabled", False))
            action_verb = "Queued" if queue_enabled else "Indexed"
            successes = 0
            failures = []

            for kind, item, _ in work_items:
                if kind == "text":
                    source_id = f"discord://guild/{guild_id}/message/{message_id}/text"
                    ok = await hybrid_search.add_document(
                        source_id=source_id,
                        text=item,
                        metadata={**base_metadata, "source_type": "text"},
                        file_type="text",
                    )
                    if ok:
                        successes += 1
                    else:
                        failures.append("text content")
                    continue

                if kind == "url":
                    try:
                        from bot.document_ingest import ingest_document_from_url

                        extracted = await ingest_document_from_url(item)
                        extracted_text = (extracted or {}).get("text") or ""
                        if not extracted_text:
                            failures.append(item)
                            continue
                        source_id = f"discord://guild/{guild_id}/message/{message_id}/url/{len(failures) + successes}"
                        ok = await hybrid_search.add_document(
                            source_id=source_id,
                            text=extracted_text,
                            metadata={
                                **base_metadata,
                                "source_type": "url",
                                "url": item,
                                "extraction_metadata": (extracted or {}).get("metadata", {}),
                            },
                            file_type="url",
                        )
                        if ok:
                            successes += 1
                        else:
                            failures.append(item)
                    except Exception as exc:
                        failures.append(f"{item} ({exc})")
                    continue

                try:
                    from bot.document_ingest import ingest_document_attachment

                    extracted = await ingest_document_attachment(item)
                    extracted_text = (extracted or {}).get("text") or ""
                    if not extracted_text:
                        failures.append(getattr(item, "filename", "attachment"))
                        continue
                    source_id = f"discord://guild/{guild_id}/message/{message_id}/attachment/{getattr(item, 'filename', 'attachment')}"
                    ok = await hybrid_search.add_document(
                        source_id=source_id,
                        text=extracted_text,
                        metadata={
                            **base_metadata,
                            "source_type": "attachment",
                            "filename": getattr(item, "filename", "attachment"),
                            "attachment_metadata": (extracted or {}).get("metadata", {}),
                        },
                        file_type=Path(getattr(item, "filename", "attachment")).suffix.lstrip(".") or "attachment",
                    )
                    if ok:
                        successes += 1
                    else:
                        failures.append(getattr(item, "filename", "attachment"))
                except Exception as exc:
                    failures.append(f"{getattr(item, 'filename', 'attachment')} ({exc})")

            if successes == 0 and failures:
                await ctx.send(f"❌ RAG indexing failed for: {safe_embed_value(', '.join(failures), 1800)}")
                return

            summary = f"{action_verb} {successes} item{'s' if successes != 1 else ''}."
            if failures:
                summary += f" Failed: {safe_embed_value(', '.join(failures), 1200)}"
            await ctx.send(summary)
        except Exception as e:
            logger.exception(f"[RAG Commands] Index-this command failed: {e}")
            await ctx.send(f"❌ **Error:** {safe_embed_value(str(e))}")

    @rag_group.command(name="index")
    @is_admin_user()
    async def rag_index(self, ctx, directory: str | None = None) -> None:
        """Index documents from a directory into the RAG database.

        Args:
            directory: Path to the directory to index (defaults to config paths)

        """
        try:
            from bot.rag.config import get_text_index_paths
            from bot.rag.document_processing import index_text_directory

            # Get paths from config or use provided directory
            paths = [Path(directory)] if directory else get_text_index_paths()

            if not paths:
                await ctx.send("❌ **Error**: No text index paths configured. Set TEXT_INDEX_PATHS in .env")
                return

            # Send initial status
            message = await ctx.send("📚 Starting document indexing...")

            total_indexed = 0
            errors = []

            for path in paths:
                if not path.exists():
                    errors.append(f"Path does not exist: {path}")
                    continue

                embed = discord.Embed(
                    title="📚 Indexing Documents",
                    description=f"Processing: `{path}`\n...",
                    color=discord.Color.yellow(),
                )
                await message.edit(embed=embed)

                try:
                    count = await index_text_directory(path)
                    total_indexed += count
                except Exception as e:
                    errors.append(f"Error indexing {path}: {e!s}")

            # Final status
            if errors:
                error_text = "\n".join(f"- {err}" for err in errors[:5])
                if len(errors) > 5:
                    error_text += f"\n... and {len(errors) - 5} more errors"

                embed = discord.Embed(
                    title="⚠️ Indexing Completed with Errors",
                    description=f"Indexed {total_indexed} documents\n\nErrors:\n{error_text}",
                    color=discord.Color.yellow(),
                )
            else:
                embed = discord.Embed(
                    title="✅ Indexing Complete",
                    description=f"Successfully indexed {total_indexed} documents",
                    color=discord.Color.green(),
                )

            await message.edit(embed=embed)

        except Exception as e:
            logger.exception(f"[RAG Commands] Index command failed: {e}")
            await ctx.send(f"❌ **Error:** {safe_embed_value(str(e))}")

    @rag_group.command(name="reload")
    @is_admin_user()
    async def rag_reload(self, ctx) -> None:
        """Reload the text index into the ChromaDB vector store."""
        try:
            from bot.rag.document_processing import reload_text_index

            message = await ctx.send("🔄 Reloading text index...")

            try:
                result = await reload_text_index()

                # Check result type - could be dict or int
                if isinstance(result, dict):
                    indexed = result.get("indexed", 0)
                    errors = result.get("errors", [])
                else:
                    indexed = result
                    errors = []

                if errors:
                    error_text = "\n".join(f"- {err}" for err in errors[:5])
                    if len(errors) > 5:
                        error_text += f"\n... and {len(errors) - 5} more"
                    description = f"Indexed {indexed} documents\n\n⚠️ Errors:\n{error_text}"
                    color = discord.Color.yellow()
                else:
                    description = f"Successfully indexed {indexed} documents"
                    color = discord.Color.green()

                embed = discord.Embed(
                    title="✅ Text Index Reloaded",
                    description=description,
                    color=color,
                )
            except Exception as e:
                embed = discord.Embed(
                    title="❌ Reload Failed",
                    description=f"Error: {safe_embed_value(str(e))}",
                    color=discord.Color.red(),
                )

            await message.edit(embed=embed)

        except Exception as e:
            logger.exception(f"[RAG Commands] Reload command failed: {e}")
            await ctx.send(f"❌ **Error:** {safe_embed_value(str(e))}")

    @rag_group.command(name="invalidate")
    @is_admin_user()
    async def rag_invalidate(self, ctx) -> None:
        """Invalidate and rebuild the ChromaDB collection."""
        try:
            global _invalidation_in_progress

            # Check if an invalidation is already in progress
            request_id = f"{ctx.guild.id if ctx.guild else 'DM'}:{ctx.channel.id}"
            if _invalidation_in_progress.get(request_id, False):
                await ctx.send("⚠️ **Invalidation already in progress**\nPlease wait for the current invalidation to complete.")
                return

            # Mark invalidation as in progress
            _invalidation_in_progress[request_id] = True

            try:
                from bot.rag.document_processing import invalidate_collection

                message = await ctx.send("🔄 Invalidating collection...")

                result = await invalidate_collection()

                if result.get("success"):
                    embed = discord.Embed(
                        title="✅ Collection Invalidated",
                        description=f"Successfully rebuilt collection with {result.get('indexed', 0)} documents",
                        color=discord.Color.green(),
                    )
                else:
                    embed = discord.Embed(
                        title="❌ Invalidation Failed",
                        description=f"Error: {result.get('error', 'Unknown error')}",
                        color=discord.Color.red(),
                    )

                await message.edit(embed=embed)

            finally:
                # Clear the invalidation flag
                _invalidation_in_progress[request_id] = False

        except Exception as e:
            logger.exception(f"[RAG Commands] Invalidate command failed: {e}")
            await ctx.send(f"❌ **Error:** {safe_embed_value(str(e))}")


async def setup(bot) -> None:
    """Add the RAG cog to the bot."""
    await bot.add_cog(RAGCommands(bot))
