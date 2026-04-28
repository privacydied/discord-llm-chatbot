"""
RAG system management commands for Discord bot.
"""

import asyncio
from pathlib import Path
from typing import List, Tuple
import discord
from discord.ext import commands

from ..rag.hybrid_search import get_hybrid_search
from ..rag.config import get_rag_environment_info, validate_rag_environment
from ..utils.logging import get_logger

logger = get_logger(__name__)


def is_admin_user():
    """Custom check that allows admin users in both guilds and DMs."""

    async def predicate(ctx):
        try:
            logger.info(
                f"[RAG Admin Check] User {ctx.author.id} ({ctx.author.name}) attempting RAG command"
            )
            logger.info(f"[RAG Admin Check] Channel type: {type(ctx.channel).__name__}")

            # In DMs, only allow the bot owner (not any guild admin) [SFT]
            if isinstance(ctx.channel, discord.DMChannel):
                logger.info("[RAG Admin Check] DM context - checking bot owner only")
                try:
                    app_info = await ctx.bot.application_info()
                    if ctx.author.id == app_info.owner.id:
                        logger.info(
                            f"[RAG Admin Check] Bot owner {ctx.author.id} allowed DM access"
                        )
                        return True
                except Exception as e:
                    logger.error(f"[RAG Admin Check] Failed to get bot owner info: {e}")

                logger.warning(
                    f"[RAG Admin Check] User {ctx.author.id} attempted RAG command in DM - only bot owner allowed"
                )
                return False

            # In guilds, use standard admin permission check
            if (
                hasattr(ctx.author, "guild_permissions")
                and ctx.author.guild_permissions
            ):
                is_admin = ctx.author.guild_permissions.administrator
                logger.info(
                    f"[RAG Admin Check] Guild context - User admin status: {is_admin}"
                )
                return is_admin
            else:
                logger.warning(
                    f"[RAG Admin Check] Could not check guild permissions for user {ctx.author.id}"
                )
                return False

        except Exception as e:
            logger.error(
                f"[RAG Admin Check] Error in permission check: {e}", exc_info=True
            )
            return False

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


def chunk_text(text: str, chunk_size: int = DISCORD_EMBED_FIELD_VALUE_LIMIT) -> List[str]:
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

    def __init__(self, bot):
        self.bot = bot
        bot.hybrid_search = bot.loop.create_task(self._init_hybrid_search())

    async def _init_hybrid_search(self):
        """Initialize hybrid search."""
        return await get_hybrid_search()


# Track invalidation state to prevent duplicate messages
_invalidation_in_progress = {}


class RAGCommands(commands.Cog):
    """Cog for RAG system management commands."""

    def __init__(self, bot):
        """Initialize the RAG cog.

        Args:
            bot: The bot instance
        """
        self.bot = bot
        self.hybrid_search = None
        self.admin_ids = bot.admin_ids if hasattr(bot, "admin_ids") else []

        # Initialize hybrid search on startup
        asyncio.create_task(self._init_hybrid_search())

    async def _init_hybrid_search(self):
        """Initialize hybrid search."""
        try:
            self.hybrid_search = await get_hybrid_search()
            logger.info("[RAGCog] Hybrid search initialized")
        except Exception as e:
            logger.error(f"[RAGCog] Failed to initialize hybrid search: {e}")

    async def cog_unload(self):
        """Cleanup on cog unload."""
        if self.hybrid_search:
            await self.hybrid_search.close()

    @commands.group(name="rag", invoke_without_command=True)
    @is_admin_user()
    async def rag_group(self, ctx):
        """RAG system management commands."""
        embed = discord.Embed(
            title="🗂️  RAG System Management",
            description="Use `!rag <command>` to manage the RAG system.",
            color=discord.Color.blue(),
        )

        embed.add_field(
            name="Status Commands",
            value="`status` - Show RAG system status\n"
            "`test` - Run RAG system tests\n"
            "`reload` - Reload text index",
            inline=False,
        )

        embed.add_field(
            name="Management Commands",
            value="`clear` - Remove documents\n"
            "`wipe` - ⚠️ Wipe entire database\n"
            "`invalidate` - Invalidate collection",
            inline=False,
        )

        embed.add_field(
            name="Search Commands",
            value="`search <query>` - Search for documents\n"
            "`index <path>` - Index documents from directory",
            inline=False,
        )

        await ctx.send(embed=embed)

    @rag_group.command(name="status")
    @is_admin_user()
    async def rag_status(self, ctx):
        """Show the status of the RAG system."""
        try:
            # Get environment info
            env_info = get_rag_environment_info()

            # Create status embed
            embed = discord.Embed(
                title="🗂️  RAG System Status", color=discord.Color.blue()
            )

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
                        f"**Documents:** {collection_stats.get('doc_count', 'N/A')}\n"
                        f"**Chunks:** {collection_stats.get('total_chunks', 'N/A')}\n"
                        f"**Avg chunks/doc:** {collection_stats.get('avg_chunks_per_doc', 'N/A'):.1f}"
                    )

                    embed.add_field(
                        name="📊 Collection Stats", value=stats_text, inline=False
                    )
                except Exception as e:
                    embed.add_field(
                        name="📊 Collection Stats",
                        value=f"Error getting stats: {str(e)}",
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
            embed.add_field(
                name="💚 Health Check", value=f"{health_emoji} Environment valid"
            )

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"[RAG Commands] Status command failed: {e}")
            await ctx.send(f"❌ **Error:** {safe_embed_value(str(e))}")

    @rag_group.command(name="test")
    @is_admin_user()
    async def rag_test(self, ctx):
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
                    )
                )
            except Exception as e:
                error_msg = safe_embed_value(str(e), 50)
                test_results.append(("❌", "Environment", f"Error: {error_msg}"))

            # Test 2: Hybrid search initialization
            try:
                if not self.hybrid_search:
                    self.hybrid_search = await get_hybrid_search()
                test_results.append(
                    ("✅", "Hybrid Search", "Successfully initialized")
                )
            except Exception as e:
                error_msg = safe_embed_value(str(e), 50)
                test_results.append(("❌", "Hybrid Search", f"Failed: {error_msg}"))

            # Test 3: Search functionality
            try:
                if self.hybrid_search:
                    results = await self.hybrid_search.search("test query", top_k=1)
                    result_count = len(results) if isinstance(results, list) else 0
                    test_results.append(
                        ("✅", "Search Functionality", f"Found {result_count} results")
                    )
                else:
                    test_results.append(
                        ("⚠️", "Search Functionality", "Hybrid search not available")
                    )
            except Exception as e:
                error_msg = safe_embed_value(str(e), 50)
                test_results.append(
                    ("❌", "Search Functionality", f"Failed: {error_msg}")
                )

            # Test 4: Collection stats
            try:
                if "search_engine" in locals():
                    stats = await search_engine.get_stats()
                    if "collection_stats" in stats:
                        chunks = stats["collection_stats"].get("total_chunks", 0)
                        test_results.append(
                            ("✅", "Collection Access", f"{chunks} chunks available")
                        )
                    else:
                        test_results.append(
                            ("⚠️", "Collection Access", "No collection stats")
                        )
                else:
                    test_results.append(
                        ("❌", "Collection Access", "Search engine not available")
                    )
            except Exception as e:
                error_msg = safe_embed_value(str(e), 50)
                test_results.append(("❌", "Collection Access", f"Failed: {error_msg}"))

            # Update embed with test results
            embed = discord.Embed(
                title="🧪 RAG System Test Results", color=discord.Color.green()
            )

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

            summary_text = (
                f"**Passed:** {passed_tests}/{total_tests} tests\n"
                f"**Status:** {'✅ All systems operational' if passed_tests == total_tests else '⚠️ Some issues detected'}"
            )

            embed.add_field(
                name="📊 Summary", value=safe_embed_value(summary_text), inline=False
            )

            await message.edit(embed=embed)

        except Exception as e:
            logger.error(f"[RAG Commands] Test command failed: {e}")
            embed = discord.Embed(
                title="❌ Test Error",
                description=safe_embed_value(f"Test execution failed: {str(e)}"),
                color=discord.Color.red(),
            )
            await ctx.send(embed=embed)

    @rag_group.command(name="wipe")
    @is_admin_user()
    async def rag_wipe(self, ctx):
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
                        logger.error(
                            "[RAG Commands] Cannot send permission error message - missing Send Messages permission"
                        )
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
            logger.error(f"[RAG Commands] Wipe command failed: {e}")
            await ctx.send(f"❌ **Error:** {safe_embed_value(str(e))}")

    @rag_group.command(name="clear")
    @is_admin_user()
    async def rag_clear(self, ctx, *source_patterns: str):
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
                note = f"Cleared all documents (source filtering not yet implemented)"
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
            logger.error(f"[RAG Commands] Clear command failed: {e}")
            await ctx.send(f"❌ **Error:** {safe_embed_value(str(e))}")

    @rag_group.command(name="search")
    @is_admin_user()
    async def rag_search(self, ctx, *, query: str):
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
            logger.error(f"[RAG Commands] Search command failed: {e}")
            await ctx.send(f"❌ **Error:** {safe_embed_value(str(e))}")

    @rag_group.command(name="index")
    @is_admin_user()
    async def rag_index(self, ctx, directory: str = None):
        """Index documents from a directory into the RAG database.

        Args:
            directory: Path to the directory to index (defaults to config paths)
        """
        try:
            from ..rag.config import get_text_index_paths
            from ..rag.document_processing import index_text_directory

            # Get paths from config or use provided directory
            if directory:
                paths = [Path(directory)]
            else:
                paths = get_text_index_paths()

            if not paths:
                await ctx.send(
                    "❌ **Error**: No text index paths configured. Set TEXT_INDEX_PATHS in .env"
                )
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
                    errors.append(f"Error indexing {path}: {str(e)}")

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
            logger.error(f"[RAG Commands] Index command failed: {e}")
            await ctx.send(f"❌ **Error:** {safe_embed_value(str(e))}")

    @rag_group.command(name="reload")
    @is_admin_user()
    async def rag_reload(self, ctx):
        """Reload the text index into the ChromaDB vector store."""
        try:
            from ..rag.document_processing import reload_text_index

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
            logger.error(f"[RAG Commands] Reload command failed: {e}")
            await ctx.send(f"❌ **Error:** {safe_embed_value(str(e))}")

    @rag_group.command(name="invalidate")
    @is_admin_user()
    async def rag_invalidate(self, ctx):
        """Invalidate and rebuild the ChromaDB collection."""
        try:
            global _invalidation_in_progress

            # Check if an invalidation is already in progress
            request_id = f"{ctx.guild.id if ctx.guild else 'DM'}:{ctx.channel.id}"
            if _invalidation_in_progress.get(request_id, False):
                await ctx.send(
                    "⚠️ **Invalidation already in progress**\n"
                    "Please wait for the current invalidation to complete."
                )
                return

            # Mark invalidation as in progress
            _invalidation_in_progress[request_id] = True

            try:
                from ..rag.document_processing import invalidate_collection

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
            logger.error(f"[RAG Commands] Invalidate command failed: {e}")
            await ctx.send(f"❌ **Error:** {safe_embed_value(str(e))}")


async def setup(bot):
    """Add the RAG cog to the bot."""
    await bot.add_cog(RAGCommands(bot))
