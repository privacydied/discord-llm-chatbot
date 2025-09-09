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

            # In DMs, check if user has admin permissions in any mutual guild
            if isinstance(ctx.channel, discord.DMChannel):
                logger.info("[RAG Admin Check] DM context - checking mutual guilds")
                # Check if user is admin in any mutual guild with the bot
                for guild in ctx.bot.guilds:
                    member = guild.get_member(ctx.author.id)
                    if member:
                        logger.debug(
                            f"[RAG Admin Check] Found user in guild {guild.name} ({guild.id})"
                        )
                        if member.guild_permissions.administrator:
                            logger.info(
                                f"[RAG Admin Check] ✅ User {ctx.author.id} has admin in guild {guild.id}, allowing DM access"
                            )
                            return True
                    else:
                        logger.debug(
                            f"[RAG Admin Check] User not found in guild {guild.name}"
                        )

                # Also allow bot owner in DMs
                try:
                    app_info = await ctx.bot.application_info()
                    if ctx.author.id == app_info.owner.id:
                        logger.info(
                            f"[RAG Admin Check] ✅ Bot owner {ctx.author.id} allowed DM access"
                        )
                        return True
                except Exception as e:
                    logger.error(f"[RAG Admin Check] Failed to get bot owner info: {e}")

                logger.warning(
                    f"[RAG Admin Check] ❌ User {ctx.author.id} attempted RAG command in DM without admin permissions"
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
                    f"[RAG Admin Check] ❌ Could not check guild permissions for user {ctx.author.id}"
                )
                return False

        except Exception as e:
            logger.error(
                f"[RAG Admin Check] ❌ Error in permission check: {e}", exc_info=True
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

    # Truncate and add indicator
    return text[: limit - 6] + "... ⚠️"


async def check_bot_permissions(ctx, required_perms=None) -> Tuple[bool, List[str]]:
    """Check if bot has required permissions in the current context.

    Args:
        ctx: Discord command context
        required_perms: List of permission names to check (defaults to basic set)

    Returns:
        Tuple of (has_all_permissions, list_of_missing_permissions)
    """
    if required_perms is None:
        required_perms = ["send_messages", "embed_links"]

    # In DMs, assume we have basic permissions
    if not ctx.guild:
        return True, []

    bot_member = ctx.guild.get_member(ctx.bot.user.id)
    if not bot_member:
        return False, ["Bot not found in guild"]

    missing_perms = []
    bot_perms = bot_member.permissions_in(ctx.channel)

    for perm in required_perms:
        if not getattr(bot_perms, perm, False):
            missing_perms.append(perm.replace("_", " ").title())

    return len(missing_perms) == 0, missing_perms


class RAGCommands(commands.Cog):
    """Commands for managing the RAG (Retrieval Augmented Generation) system."""

    def __init__(self, bot):
        self.bot = bot

    @commands.group(name="rag", invoke_without_command=True)
    @is_admin_user()
    async def rag_group(self, ctx):
        """RAG system management commands."""
        if ctx.invoked_subcommand is None:
            embed = discord.Embed(
                title="🧠 RAG System Commands",
                description="Retrieval Augmented Generation management",
                color=discord.Color.blue(),
            )

            embed.add_field(
                name="📊 Status & Info",
                value="`!rag status` - System status\n`!rag stats` - Search statistics\n`!rag config` - Configuration info",
                inline=False,
            )

            embed.add_field(
                name="🔄 Management",
                value="`!rag bootstrap` - Initialize knowledge base\n`!rag scan` - Scan for new documents only\n`!rag wipe` - Wipe entire database",
                inline=False,
            )

            embed.add_field(
                name="🔍 Testing",
                value="`!rag search <query>` - Test search\n`!rag test` - Run system tests",
                inline=False,
            )

            embed.add_field(
                name="⏹️ Control",
                value="`!rag stop` - Cancel running operations\n`!rag tasks` - Show active tasks",
                inline=False,
            )

            await ctx.send(embed=embed)

    @rag_group.command(name="status")
    @is_admin_user()
    async def rag_status(self, ctx):
        """Show RAG system status and health."""
        try:
            # Get hybrid search instance
            search_engine = await get_hybrid_search()
            stats = await search_engine.get_stats()

            # Validate environment
            env_valid, env_issues = validate_rag_environment()

            embed = discord.Embed(
                title="🧠 RAG System Status",
                color=discord.Color.green()
                if stats.get("rag_available")
                else discord.Color.orange(),
            )

            # System status
            status_icon = "✅" if stats.get("rag_available") else "⚠️"
            status_value = (
                f"**Enabled:** {stats.get('rag_enabled')}\n"
                f"**Initialized:** {stats.get('rag_initialized')}\n"
                f"**Available:** {stats.get('rag_available')}"
            )
            embed.add_field(
                name=f"{status_icon} System Status",
                value=safe_embed_value(status_value),
                inline=True,
            )

            # Collection info
            if "collection_stats" in stats:
                coll_stats = stats["collection_stats"]
                model_name = coll_stats.get("embedding_model", "N/A")
                if len(model_name) > 30:
                    model_name = model_name[:27] + "..."

                kb_value = (
                    f"**Total Chunks:** {coll_stats.get('total_chunks', 0)}\n"
                    f"**Collection:** {coll_stats.get('collection_name', 'N/A')}\n"
                    f"**Embedding Model:** {model_name}"
                )
                embed.add_field(
                    name="📚 Knowledge Base",
                    value=safe_embed_value(kb_value),
                    inline=True,
                )

            # Environment validation
            env_icon = "✅" if env_valid else "❌"
            env_status = "Valid" if env_valid else f"{len(env_issues)} issues"
            embed.add_field(
                name=f"{env_icon} Environment",
                value=safe_embed_value(f"**Status:** {env_status}"),
                inline=True,
            )

            # Show issues if any
            if env_issues:
                # Limit number of issues shown
                max_issues = 5
                issues_list = env_issues[:max_issues]
                issues_text = "\n".join(f"• {issue}" for issue in issues_list)

                if len(env_issues) > max_issues:
                    issues_text += f"\n• ... and {len(env_issues) - max_issues} more"

                embed.add_field(
                    name="⚠️ Configuration Issues",
                    value=safe_embed_value(f"```{issues_text}```"),
                    inline=False,
                )

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"[RAG Commands] Status command failed: {e}")
            embed = discord.Embed(
                title="❌ RAG Status Error",
                description=safe_embed_value(f"Failed to get system status: {str(e)}"),
                color=discord.Color.red(),
            )
            await ctx.send(embed=embed)

    @rag_group.command(name="stats")
    @is_admin_user()
    async def rag_stats(self, ctx):
        """Show RAG search statistics."""
        try:
            search_engine = await get_hybrid_search()
            stats = await search_engine.get_stats()

            embed = discord.Embed(
                title="📊 RAG Search Statistics", color=discord.Color.blue()
            )

            search_stats = stats.get("search_stats", {})
            total_searches = search_stats.get("total_searches", 0)

            if total_searches > 0:
                embed.add_field(
                    name="🔍 Search Breakdown",
                    value=f"**Total:** {total_searches}\n"
                    f"**Vector:** {search_stats.get('vector_searches', 0)}\n"
                    f"**Keyword:** {search_stats.get('keyword_searches', 0)}\n"
                    f"**Hybrid:** {search_stats.get('hybrid_searches', 0)}\n"
                    f"**Fallback:** {search_stats.get('fallback_searches', 0)}",
                    inline=True,
                )

                # Calculate percentages
                vector_pct = (
                    search_stats.get("vector_searches", 0) / total_searches
                ) * 100
                hybrid_pct = (
                    search_stats.get("hybrid_searches", 0) / total_searches
                ) * 100
                fallback_pct = (
                    search_stats.get("fallback_searches", 0) / total_searches
                ) * 100

                embed.add_field(
                    name="📈 Usage Patterns",
                    value=f"**Vector Usage:** {vector_pct:.1f}%\n"
                    f"**Hybrid Usage:** {hybrid_pct:.1f}%\n"
                    f"**Fallback Rate:** {fallback_pct:.1f}%",
                    inline=True,
                )
            else:
                embed.add_field(
                    name="📊 Statistics",
                    value="No searches performed yet",
                    inline=False,
                )

            # Collection stats
            if "collection_stats" in stats:
                coll_stats = stats["collection_stats"]
                embed.add_field(
                    name="📚 Collection Info",
                    value=f"**Chunks:** {coll_stats.get('total_chunks', 0)}\n"
                    f"**Dimensions:** {coll_stats.get('embedding_dimension', 'N/A')}",
                    inline=True,
                )

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"[RAG Commands] Stats command failed: {e}")
            await ctx.send(f"❌ Failed to get statistics: {str(e)}")

    @rag_group.command(name="config")
    @is_admin_user()
    async def rag_config(self, ctx):
        """Show RAG configuration information."""
        try:
            env_info = get_rag_environment_info()

            embed = discord.Embed(
                title="⚙️ RAG Configuration", color=discord.Color.blue()
            )

            # Core settings
            embed.add_field(
                name="🔧 Core Settings",
                value=f"**Enabled:** {env_info.get('ENABLE_RAG')}\n"
                f"**DB Path:** {env_info.get('RAG_DB_PATH')}\n"
                f"**KB Path:** {env_info.get('RAG_KB_PATH')}",
                inline=False,
            )

            # Embedding model
            embed.add_field(
                name="🤖 Embedding Model",
                value=f"**Type:** {env_info.get('RAG_EMBEDDING_MODEL_TYPE')}\n"
                f"**Model:** {env_info.get('RAG_EMBEDDING_MODEL_NAME')[:40]}...",
                inline=False,
            )

            # Search parameters
            embed.add_field(
                name="🔍 Search Parameters",
                value=f"**Vector Weight:** {env_info.get('RAG_VECTOR_WEIGHT')}\n"
                f"**Keyword Weight:** {env_info.get('RAG_KEYWORD_WEIGHT')}\n"
                f"**Confidence Threshold:** {env_info.get('RAG_VECTOR_CONFIDENCE_THRESHOLD')}\n"
                f"**Max Vector Results:** {env_info.get('RAG_MAX_VECTOR_RESULTS')}",
                inline=True,
            )

            # Chunking settings
            embed.add_field(
                name="📄 Chunking Settings",
                value=f"**Chunk Size:** {env_info.get('RAG_CHUNK_SIZE')}\n"
                f"**Overlap:** {env_info.get('RAG_CHUNK_OVERLAP')}\n"
                f"**Min Size:** {env_info.get('RAG_MIN_CHUNK_SIZE')}",
                inline=True,
            )

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"[RAG Commands] Config command failed: {e}")
            await ctx.send(f"❌ Failed to get configuration: {str(e)}")

    @rag_group.command(name="bootstrap")
    @is_admin_user()
    async def rag_bootstrap(self, ctx, force: bool = False):
        """Bootstrap the RAG knowledge base from files."""
        try:
            embed = discord.Embed(
                title="🚀 RAG Bootstrap Starting",
                description="Initializing knowledge base from files...",
                color=discord.Color.orange(),
            )
            message = await ctx.send(embed=embed)

            # Get search engine and bootstrap
            search_engine = await get_hybrid_search()

            if not search_engine.bootstrap:
                raise Exception("Bootstrap utility not available")

            # Run bootstrap
            result = await search_engine.bootstrap.bootstrap_knowledge_base(
                force_refresh=force
            )

            # Update embed with results
            if "error" in result:
                embed = discord.Embed(
                    title="❌ Bootstrap Failed",
                    description=safe_embed_value(
                        result["error"], DISCORD_EMBED_DESCRIPTION_LIMIT
                    ),
                    color=discord.Color.red(),
                )
            else:
                embed = discord.Embed(
                    title="✅ Bootstrap Completed", color=discord.Color.green()
                )

                # Results summary
                stats_value = (
                    f"**Files Processed:** {result.get('files_processed', 0)}\n"
                    f"**Files Skipped:** {result.get('files_skipped', 0)}\n"
                    f"**Total Chunks:** {result.get('total_chunks', 0)}"
                )
                embed.add_field(
                    name="📊 Results", value=safe_embed_value(stats_value), inline=True
                )

                # Errors (if any)
                if result.get("errors"):
                    # Limit number of errors shown
                    max_errors = 3
                    error_list = result["errors"][:max_errors]
                    error_text = "\n".join(error_list)

                    if len(result["errors"]) > max_errors:
                        error_text += (
                            f"\n... and {len(result['errors']) - max_errors} more"
                        )

                    embed.add_field(
                        name="⚠️ Errors",
                        value=safe_embed_value(f"```{error_text}```"),
                        inline=False,
                    )

                # Processed files (if available and not too many)
                if (
                    result.get("processed_files")
                    and len(result["processed_files"]) <= 10
                ):
                    files_text = "\n".join(
                        f"• {f['file']} ({f['chunks']} chunks)"
                        for f in result["processed_files"][:10]
                    )

                    if len(result["processed_files"]) > 10:
                        files_text += f"\n... and {len(result['processed_files']) - 10} more files"

                    embed.add_field(
                        name="📄 Processed Files",
                        value=safe_embed_value(files_text),
                        inline=False,
                    )

            await message.edit(embed=embed)

        except Exception as e:
            logger.error(f"[RAG Commands] Bootstrap command failed: {e}")
            embed = discord.Embed(
                title="❌ Bootstrap Error",
                description=safe_embed_value(f"Bootstrap failed: {str(e)}"),
                color=discord.Color.red(),
            )
            await ctx.send(embed=embed)

    @rag_group.command(name="search")
    @is_admin_user()
    async def rag_search(self, ctx, *, query: str):
        """Test RAG search with a query."""
        try:
            embed = discord.Embed(
                title="🔍 RAG Search",
                description=safe_embed_value(
                    f"Query: {query}", DISCORD_EMBED_DESCRIPTION_LIMIT
                ),
                color=discord.Color.blue(),
            )
            message = await ctx.send(embed=embed)

            # Get search engine
            search_engine = await get_hybrid_search()

            # Perform search
            results = await search_engine.search(
                query=query,
                user_id=str(ctx.author.id),
                guild_id=str(ctx.guild.id) if ctx.guild else None,
                max_results=5,
            )

            # Update embed with results
            embed = discord.Embed(
                title="🔍 RAG Search Results",
                description=safe_embed_value(
                    f"Query: {query}", DISCORD_EMBED_DESCRIPTION_LIMIT
                ),
                color=discord.Color.blue(),
            )

            if not results:
                embed.add_field(
                    name="❌ No Results",
                    value="No matching documents found.",
                    inline=False,
                )
            else:
                # Limit number of results to show
                max_results = min(5, len(results))
                for i, result in enumerate(results[:max_results]):
                    # Format result with truncation
                    result_value = (
                        f"**Source:** {result.source}\n"
                        f"**Score:** {result.score:.2f}\n"
                        f"**Type:** {result.search_type}\n\n"
                        f"{result.snippet}"
                    )

                    embed.add_field(
                        name=f"{i + 1}. {safe_embed_value(result.title, 256)}",
                        value=safe_embed_value(result_value),
                        inline=False,
                    )

            await message.edit(embed=embed)

        except Exception as e:
            logger.error(f"[RAG Commands] Search command failed: {e}")
            embed = discord.Embed(
                title="❌ Search Error",
                description=safe_embed_value(f"Search failed: {str(e)}"),
                color=discord.Color.red(),
            )
            await ctx.send(embed=embed)

    @rag_group.command(name="test")
    @is_admin_user()
    async def rag_test(self, ctx):
        """Run RAG system tests."""
        try:
            embed = discord.Embed(
                title="🧪 RAG System Tests",
                description="Running comprehensive system tests...",
                color=discord.Color.orange(),
            )
            message = await ctx.send(embed=embed)

            test_results = []

            # Test 1: System initialization
            try:
                search_engine = await get_hybrid_search()
                test_results.append(("✅", "System Initialization", "Passed"))
            except Exception as e:
                error_msg = safe_embed_value(str(e), 50)
                test_results.append(
                    ("❌", "System Initialization", f"Failed: {error_msg}")
                )

            # Test 2: Environment validation
            env_valid, env_issues = validate_rag_environment()
            if env_valid:
                test_results.append(("✅", "Environment Validation", "Passed"))
            else:
                test_results.append(
                    ("⚠️", "Environment Validation", f"{len(env_issues)} issues")
                )

            # Test 3: Search functionality
            try:
                if "search_engine" in locals():
                    results = await search_engine.search("test query", max_results=1)
                    test_results.append(
                        (
                            "✅",
                            "Search Functionality",
                            f"Returned {len(results)} results",
                        )
                    )
                else:
                    test_results.append(
                        ("❌", "Search Functionality", "Search engine not available")
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

            # Send confirmation prompt
            embed = discord.Embed(
                title="⚠️ Confirm Database Wipe",
                description=(
                    "**This will permanently delete ALL documents from the RAG database!**\n\n"
                    "This action cannot be undone. All indexed documents, embeddings, "
                    "and metadata will be lost.\n\n"
                    "React with ✅ to confirm or ❌ to cancel."
                ),
                color=discord.Color.red(),
            )

            message = await ctx.send(embed=embed)
            await message.add_reaction("✅")
            await message.add_reaction("❌")

            def check(reaction, user):
                return (
                    user == ctx.author
                    and str(reaction.emoji) in ["✅", "❌"]
                    and reaction.message.id == message.id
                )

            try:
                reaction, user = await self.bot.wait_for(
                    "reaction_add", timeout=30.0, check=check
                )

                if str(reaction.emoji) == "❌":
                    embed = discord.Embed(
                        title="❌ Database Wipe Cancelled",
                        description="Database wipe operation was cancelled.",
                        color=discord.Color.blue(),
                    )
                    await message.edit(embed=embed)
                    return

                # User confirmed, proceed with wipe
                embed = discord.Embed(
                    title="🗑️ Wiping Database...",
                    description="Deleting all documents and embeddings. Please wait...",
                    color=discord.Color.orange(),
                )
                await message.edit(embed=embed)

                # Run wipe operation asynchronously to prevent Discord timeout
                try:
                    # Get hybrid search instance
                    search_engine = await get_hybrid_search()

                    # Create a task for the wipe operation
                    async def wipe_with_progress():
                        # Update progress message
                        progress_embed = discord.Embed(
                            title="🗑️ Wiping Database...",
                            description="🔄 Initializing wipe operation...",
                            color=discord.Color.orange(),
                        )
                        await message.edit(embed=progress_embed)

                        # Perform the actual wipe
                        result = await search_engine.wipe_collection()
                        return result

                    # Run the wipe operation with a reasonable timeout
                    wipe_successful = await asyncio.wait_for(
                        wipe_with_progress(), timeout=120.0
                    )

                    if not wipe_successful:
                        embed = discord.Embed(
                            title="❌ Database Wipe Failed",
                            description="Failed to wipe the database. Check logs for details.",
                            color=discord.Color.red(),
                        )
                        await message.edit(embed=embed)
                        return

                    # Also clear version tracking
                    version_file = Path("kb/.rag_versions.json")
                    if version_file.exists():
                        version_file.unlink()

                    embed = discord.Embed(
                        title="✅ Database Wiped Successfully",
                        description=(
                            "All documents and embeddings have been deleted.\n\n"
                            "You can now run `!rag bootstrap` to re-index your documents."
                        ),
                        color=discord.Color.green(),
                    )
                    await message.edit(embed=embed)

                except asyncio.TimeoutError:
                    embed = discord.Embed(
                        title="⏰ Database Wipe Timeout",
                        description=(
                            "The wipe operation is taking longer than expected.\n\n"
                            "The operation may still be running in the background. "
                            "Please check the logs and try again later if needed."
                        ),
                        color=discord.Color.orange(),
                    )
                    await message.edit(embed=embed)
                    logger.warning(
                        "[RAG Commands] Wipe operation timed out after 120 seconds"
                    )
                    return

            except asyncio.TimeoutError:
                embed = discord.Embed(
                    title="⏰ Confirmation Timeout",
                    description="Database wipe cancelled due to timeout.",
                    color=discord.Color.blue(),
                )
                await message.edit(embed=embed)

        except discord.Forbidden as e:
            logger.error(
                f"[RAG Commands] Discord permission error in wipe command: {e}"
            )
            # Try to send a simple text message as fallback
            try:
                await ctx.send(
                    f"❌ **Permission Error**: The bot lacks necessary Discord permissions.\n\nError: {e}\n\nPlease ensure the bot has these permissions:\n• Send Messages\n• Add Reactions\n• Embed Links\n• Read Message History\n• Use External Emojis"
                )
            except discord.Forbidden:
                # If we can't even send messages, log it and give up
                logger.critical(
                    f"[RAG Commands] Cannot send any messages - bot lacks Send Messages permission in channel {ctx.channel.id}"
                )
        except Exception as e:
            logger.error(f"[RAG Commands] Wipe command failed: {e}", exc_info=True)
            # Try embed first, fallback to plain text
            try:
                embed = discord.Embed(
                    title="❌ Wipe Error",
                    description=safe_embed_value(f"Database wipe failed: {str(e)}"),
                    color=discord.Color.red(),
                )
                await ctx.send(embed=embed)
            except discord.Forbidden:
                # Fallback to plain text if embed permissions are missing
                try:
                    await ctx.send(f"❌ Database wipe failed: {str(e)}")
                except discord.Forbidden:
                    logger.critical(
                        f"[RAG Commands] Cannot send error message - bot lacks Send Messages permission in channel {ctx.channel.id}"
                    )

    @rag_group.command(name="scan")
    @is_admin_user()
    async def rag_scan(self, ctx):
        """Scan for new documents only (incremental update)."""
        try:
            embed = discord.Embed(
                title="🔍 Scanning for New Documents...",
                description="Checking for new or modified files in the knowledge base.",
                color=discord.Color.blue(),
            )
            message = await ctx.send(embed=embed)

            # Get hybrid search instance
            search_engine = await get_hybrid_search()

            # Run incremental scan (force_refresh=False)
            result = await search_engine.bootstrap.bootstrap_knowledge_base(
                force_refresh=False
            )

            # Parse results
            files_processed = result.get("files_processed", 0)
            files_skipped = result.get("files_skipped", 0)
            total_chunks = result.get("total_chunks", 0)
            errors = result.get("errors", [])

            # Determine embed color based on results
            if errors:
                color = discord.Color.orange()
                status_icon = "⚠️"
                status_text = "Completed with warnings"
            elif files_processed > 0:
                color = discord.Color.green()
                status_icon = "✅"
                status_text = "Scan completed successfully"
            else:
                color = discord.Color.blue()
                status_icon = "ℹ️"
                status_text = "No new documents found"

            embed = discord.Embed(
                title=f"{status_icon} Incremental Scan {status_text}", color=color
            )

            # Add statistics
            stats_value = (
                f"**New Files Processed:** {files_processed}\n"
                f"**Files Skipped:** {files_skipped}\n"
                f"**New Chunks Created:** {total_chunks}"
            )
            embed.add_field(
                name="📊 Scan Results",
                value=safe_embed_value(stats_value),
                inline=False,
            )

            # Add processed files if any
            if "processed_files" in result and result["processed_files"]:
                processed_list = []
                for file_info in result["processed_files"][:5]:  # Show max 5 files
                    filename = file_info.get("filename", "Unknown")
                    chunks = file_info.get("chunks", 0)
                    processed_list.append(f"• {filename} ({chunks} chunks)")

                if len(result["processed_files"]) > 5:
                    processed_list.append(
                        f"... and {len(result['processed_files']) - 5} more"
                    )

                embed.add_field(
                    name="📄 New Files Processed",
                    value=safe_embed_value("\n".join(processed_list)),
                    inline=False,
                )

            # Add errors if any
            if errors:
                error_list = []
                for error in errors[:3]:  # Show max 3 errors
                    error_list.append(f"• {safe_embed_value(str(error), 100)}")

                if len(errors) > 3:
                    error_list.append(f"... and {len(errors) - 3} more errors")

                embed.add_field(
                    name="❌ Errors",
                    value=safe_embed_value("\n".join(error_list)),
                    inline=False,
                )

            # Add helpful tip
            if files_processed == 0 and not errors:
                embed.add_field(
                    name="💡 Tip",
                    value="All files are up to date. Use `!rag bootstrap force:True` to force re-process all files.",
                    inline=False,
                )

            await message.edit(embed=embed)

        except Exception as e:
            logger.error(f"[RAG Commands] Scan command failed: {e}")
            embed = discord.Embed(
                title="❌ Scan Error",
                description=safe_embed_value(f"Incremental scan failed: {str(e)}"),
                color=discord.Color.red(),
            )
            await ctx.send(embed=embed)

    @rag_group.command(name="tasks")
    @is_admin_user()
    async def rag_tasks(self, ctx):
        """Show active long-running RAG tasks."""
        try:
            # Get active tasks for this user
            user_tasks = self.bot.get_active_tasks_for_user(ctx.author.id)

            embed = discord.Embed(
                title="🔄 Active RAG Tasks", color=discord.Color.blue()
            )

            if not user_tasks:
                embed.description = "No active RAG tasks running."
                embed.add_field(
                    name="💡 Tip",
                    value="Long-running commands like `!rag bootstrap` will appear here while running.",
                    inline=False,
                )
            else:
                task_list = []
                for task_id, metadata in user_tasks:
                    command = metadata.get("command", "unknown")
                    started_at = metadata.get("started_at", 0)
                    current_time = asyncio.get_event_loop().time()
                    duration = int(current_time - started_at)

                    task_list.append(f"• `{command}` - Running for {duration}s")
                    task_list.append(f"  Task ID: `{task_id}`")

                embed.description = "\n".join(task_list)
                embed.add_field(
                    name="⏹️ Stop Tasks",
                    value="Use `!rag stop <task_id>` or `!rag stop all` to cancel tasks.",
                    inline=False,
                )

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"[RAG Commands] Tasks command failed: {e}")
            embed = discord.Embed(
                title="❌ Tasks Error",
                description=f"Failed to get active tasks: {str(e)}",
                color=discord.Color.red(),
            )
            await ctx.send(embed=embed)

    @rag_group.command(name="stop")
    @is_admin_user()
    async def rag_stop(self, ctx, task_id: str = None):
        """Stop/cancel active RAG tasks.

        Usage:
        !rag stop <task_id>  - Stop specific task
        !rag stop all        - Stop all your active tasks
        !rag stop            - Show active tasks (same as !rag tasks)
        """
        try:
            # If no task_id provided, show active tasks
            if not task_id:
                await self.rag_tasks(ctx)
                return

            user_tasks = self.bot.get_active_tasks_for_user(ctx.author.id)

            if not user_tasks:
                embed = discord.Embed(
                    title="ℹ️ No Active Tasks",
                    description="You have no active RAG tasks to stop.",
                    color=discord.Color.blue(),
                )
                await ctx.send(embed=embed)
                return

            # Handle "all" to stop all user tasks
            if task_id.lower() == "all":
                stopped_count = 0
                for tid, metadata in user_tasks:
                    if await self.bot.cancel_task(tid):
                        stopped_count += 1

                embed = discord.Embed(
                    title="⏹️ Tasks Stopped",
                    description=f"Successfully stopped {stopped_count} task(s).",
                    color=discord.Color.green(),
                )
                await ctx.send(embed=embed)
                return

            # Stop specific task
            task_found = False
            for tid, metadata in user_tasks:
                if tid == task_id or tid.endswith(f"_{task_id}"):
                    task_found = True
                    if await self.bot.cancel_task(tid):
                        command = metadata.get("command", "unknown")
                        embed = discord.Embed(
                            title="⏹️ Task Stopped",
                            description=f"Successfully stopped task: `{command}`",
                            color=discord.Color.green(),
                        )
                        embed.add_field(name="Task ID", value=f"`{tid}`", inline=False)
                    else:
                        embed = discord.Embed(
                            title="❌ Stop Failed",
                            description=f"Failed to stop task: `{task_id}`",
                            color=discord.Color.red(),
                        )
                    break

            if not task_found:
                embed = discord.Embed(
                    title="❌ Task Not Found",
                    description=f"Task `{task_id}` not found or not owned by you.",
                    color=discord.Color.red(),
                )
                embed.add_field(
                    name="💡 Tip",
                    value="Use `!rag tasks` to see your active tasks.",
                    inline=False,
                )

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"[RAG Commands] Stop command failed: {e}")
            embed = discord.Embed(
                title="❌ Stop Error",
                description=f"Failed to stop task: {str(e)}",
                color=discord.Color.red(),
            )
            await ctx.send(embed=embed)


async def setup(bot):
    """Set up RAG commands."""
    try:
        logger.info("[RAG Setup] Starting RAG commands cog initialization...")

        # Check if cog is already loaded
        existing_cog = bot.get_cog("RAGCommands")
        if existing_cog:
            logger.warning("[RAG Setup] RAGCommands cog already loaded, removing first")
            await bot.remove_cog("RAGCommands")

        # Create and add the cog
        rag_cog = RAGCommands(bot)
        await bot.add_cog(rag_cog)

        # Verify the cog was added
        loaded_cog = bot.get_cog("RAGCommands")
        if loaded_cog:
            logger.info("✅ RAG commands cog loaded successfully")

            # List the commands that were registered
            rag_commands = [cmd.name for cmd in loaded_cog.get_commands()]
            logger.info(f"[RAG Setup] Registered commands: {rag_commands}")
        else:
            logger.error("❌ RAG commands cog failed to load - not found after adding")

    except Exception as e:
        logger.error(f"❌ Failed to load RAG commands cog: {e}", exc_info=True)
        raise
