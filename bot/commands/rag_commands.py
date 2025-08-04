"""
RAG system management commands for Discord bot.
"""
import asyncio
from typing import Optional, Dict, Any, List, Tuple
import discord
from discord.ext import commands

from ..rag.hybrid_search import get_hybrid_search
from ..rag.config import get_rag_environment_info, validate_rag_environment
from ..util.logging import get_logger

logger = get_logger(__name__)

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
    return text[:limit-6] + "... ⚠️"


class RAGCommands(commands.Cog):
    """Commands for managing the RAG (Retrieval Augmented Generation) system."""
    
    def __init__(self, bot):
        self.bot = bot
    
    @commands.group(name='rag', invoke_without_command=True)
    @commands.has_permissions(administrator=True)
    async def rag_group(self, ctx):
        """RAG system management commands."""
        if ctx.invoked_subcommand is None:
            embed = discord.Embed(
                title="🧠 RAG System Commands",
                description="Retrieval Augmented Generation management",
                color=discord.Color.blue()
            )
            
            embed.add_field(
                name="📊 Status & Info",
                value="`!rag status` - System status\n`!rag stats` - Search statistics\n`!rag config` - Configuration info",
                inline=False
            )
            
            embed.add_field(
                name="🔄 Management",
                value="`!rag bootstrap` - Initialize knowledge base\n`!rag refresh` - Refresh all documents\n`!rag update` - Incremental update",
                inline=False
            )
            
            embed.add_field(
                name="🔍 Testing",
                value="`!rag search <query>` - Test search\n`!rag test` - Run system tests",
                inline=False
            )
            
            await ctx.send(embed=embed)
    
    @rag_group.command(name='status')
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
                color=discord.Color.green() if stats.get('rag_available') else discord.Color.orange()
            )
            
            # System status
            status_icon = "✅" if stats.get('rag_available') else "⚠️"
            status_value = (
                f"**Enabled:** {stats.get('rag_enabled')}\n"
                f"**Initialized:** {stats.get('rag_initialized')}\n"
                f"**Available:** {stats.get('rag_available')}"
            )
            embed.add_field(
                name=f"{status_icon} System Status",
                value=safe_embed_value(status_value),
                inline=True
            )
            
            # Collection info
            if 'collection_stats' in stats:
                coll_stats = stats['collection_stats']
                model_name = coll_stats.get('embedding_model', 'N/A')
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
                    inline=True
                )
            
            # Environment validation
            env_icon = "✅" if env_valid else "❌"
            env_status = "Valid" if env_valid else f"{len(env_issues)} issues"
            embed.add_field(
                name=f"{env_icon} Environment",
                value=safe_embed_value(f"**Status:** {env_status}"),
                inline=True
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
                    inline=False
                )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"[RAG Commands] Status command failed: {e}")
            embed = discord.Embed(
                title="❌ RAG Status Error",
                description=safe_embed_value(f"Failed to get system status: {str(e)}"),
                color=discord.Color.red()
            )
            await ctx.send(embed=embed)
    
    @rag_group.command(name='stats')
    async def rag_stats(self, ctx):
        """Show RAG search statistics."""
        try:
            search_engine = await get_hybrid_search()
            stats = await search_engine.get_stats()
            
            embed = discord.Embed(
                title="📊 RAG Search Statistics",
                color=discord.Color.blue()
            )
            
            search_stats = stats.get('search_stats', {})
            total_searches = search_stats.get('total_searches', 0)
            
            if total_searches > 0:
                embed.add_field(
                    name="🔍 Search Breakdown",
                    value=f"**Total:** {total_searches}\n"
                          f"**Vector:** {search_stats.get('vector_searches', 0)}\n"
                          f"**Keyword:** {search_stats.get('keyword_searches', 0)}\n"
                          f"**Hybrid:** {search_stats.get('hybrid_searches', 0)}\n"
                          f"**Fallback:** {search_stats.get('fallback_searches', 0)}",
                    inline=True
                )
                
                # Calculate percentages
                vector_pct = (search_stats.get('vector_searches', 0) / total_searches) * 100
                hybrid_pct = (search_stats.get('hybrid_searches', 0) / total_searches) * 100
                fallback_pct = (search_stats.get('fallback_searches', 0) / total_searches) * 100
                
                embed.add_field(
                    name="📈 Usage Patterns",
                    value=f"**Vector Usage:** {vector_pct:.1f}%\n"
                          f"**Hybrid Usage:** {hybrid_pct:.1f}%\n"
                          f"**Fallback Rate:** {fallback_pct:.1f}%",
                    inline=True
                )
            else:
                embed.add_field(
                    name="📊 Statistics",
                    value="No searches performed yet",
                    inline=False
                )
            
            # Collection stats
            if 'collection_stats' in stats:
                coll_stats = stats['collection_stats']
                embed.add_field(
                    name="📚 Collection Info",
                    value=f"**Chunks:** {coll_stats.get('total_chunks', 0)}\n"
                          f"**Dimensions:** {coll_stats.get('embedding_dimension', 'N/A')}",
                    inline=True
                )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"[RAG Commands] Stats command failed: {e}")
            await ctx.send(f"❌ Failed to get statistics: {str(e)}")
    
    @rag_group.command(name='config')
    async def rag_config(self, ctx):
        """Show RAG configuration information."""
        try:
            env_info = get_rag_environment_info()
            
            embed = discord.Embed(
                title="⚙️ RAG Configuration",
                color=discord.Color.blue()
            )
            
            # Core settings
            embed.add_field(
                name="🔧 Core Settings",
                value=f"**Enabled:** {env_info.get('ENABLE_RAG')}\n"
                      f"**DB Path:** {env_info.get('RAG_DB_PATH')}\n"
                      f"**KB Path:** {env_info.get('RAG_KB_PATH')}",
                inline=False
            )
            
            # Embedding model
            embed.add_field(
                name="🤖 Embedding Model",
                value=f"**Type:** {env_info.get('RAG_EMBEDDING_MODEL_TYPE')}\n"
                      f"**Model:** {env_info.get('RAG_EMBEDDING_MODEL_NAME')[:40]}...",
                inline=False
            )
            
            # Search parameters
            embed.add_field(
                name="🔍 Search Parameters",
                value=f"**Vector Weight:** {env_info.get('RAG_VECTOR_WEIGHT')}\n"
                      f"**Keyword Weight:** {env_info.get('RAG_KEYWORD_WEIGHT')}\n"
                      f"**Confidence Threshold:** {env_info.get('RAG_VECTOR_CONFIDENCE_THRESHOLD')}\n"
                      f"**Max Vector Results:** {env_info.get('RAG_MAX_VECTOR_RESULTS')}",
                inline=True
            )
            
            # Chunking settings
            embed.add_field(
                name="📄 Chunking Settings",
                value=f"**Chunk Size:** {env_info.get('RAG_CHUNK_SIZE')}\n"
                      f"**Overlap:** {env_info.get('RAG_CHUNK_OVERLAP')}\n"
                      f"**Min Size:** {env_info.get('RAG_MIN_CHUNK_SIZE')}",
                inline=True
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"[RAG Commands] Config command failed: {e}")
            await ctx.send(f"❌ Failed to get configuration: {str(e)}")
    
    @rag_group.command(name='bootstrap')
    async def rag_bootstrap(self, ctx, force: bool = False):
        """Bootstrap the RAG knowledge base from files."""
        try:
            embed = discord.Embed(
                title="🚀 RAG Bootstrap Starting",
                description="Initializing knowledge base from files...",
                color=discord.Color.orange()
            )
            message = await ctx.send(embed=embed)
            
            # Get search engine and bootstrap
            search_engine = await get_hybrid_search()
            
            if not search_engine.bootstrap:
                raise Exception("Bootstrap utility not available")
            
            # Run bootstrap
            result = await search_engine.bootstrap.bootstrap_knowledge_base(force_refresh=force)
            
            # Update embed with results
            if 'error' in result:
                embed = discord.Embed(
                    title="❌ Bootstrap Failed",
                    description=safe_embed_value(result['error'], DISCORD_EMBED_DESCRIPTION_LIMIT),
                    color=discord.Color.red()
                )
            else:
                embed = discord.Embed(
                    title="✅ Bootstrap Completed",
                    color=discord.Color.green()
                )
                
                # Results summary
                stats_value = (
                    f"**Files Processed:** {result.get('files_processed', 0)}\n"
                    f"**Files Skipped:** {result.get('files_skipped', 0)}\n"
                    f"**Total Chunks:** {result.get('total_chunks', 0)}"
                )
                embed.add_field(
                    name="📊 Results",
                    value=safe_embed_value(stats_value),
                    inline=True
                )
                
                # Errors (if any)
                if result.get('errors'):
                    # Limit number of errors shown
                    max_errors = 3
                    error_list = result['errors'][:max_errors]
                    error_text = "\n".join(error_list)
                    
                    if len(result['errors']) > max_errors:
                        error_text += f"\n... and {len(result['errors']) - max_errors} more"
                    
                    embed.add_field(
                        name="⚠️ Errors",
                        value=safe_embed_value(f"```{error_text}```"),
                        inline=False
                    )
                
                # Processed files (if available and not too many)
                if result.get('processed_files') and len(result['processed_files']) <= 10:
                    files_text = "\n".join(
                        f"• {f['file']} ({f['chunks']} chunks)" 
                        for f in result['processed_files'][:10]
                    )
                    
                    if len(result['processed_files']) > 10:
                        files_text += f"\n... and {len(result['processed_files']) - 10} more files"
                    
                    embed.add_field(
                        name="📄 Processed Files",
                        value=safe_embed_value(files_text),
                        inline=False
                    )
            
            await message.edit(embed=embed)
            
        except Exception as e:
            logger.error(f"[RAG Commands] Bootstrap command failed: {e}")
            embed = discord.Embed(
                title="❌ Bootstrap Error",
                description=safe_embed_value(f"Bootstrap failed: {str(e)}"),
                color=discord.Color.red()
            )
            await ctx.send(embed=embed)
    
    @rag_group.command(name='search')
    async def rag_search(self, ctx, *, query: str):
        """Test RAG search with a query."""
        try:
            embed = discord.Embed(
                title="🔍 RAG Search",
                description=safe_embed_value(f"Query: {query}", DISCORD_EMBED_DESCRIPTION_LIMIT),
                color=discord.Color.blue()
            )
            message = await ctx.send(embed=embed)
            
            # Get search engine
            search_engine = await get_hybrid_search()
            
            # Perform search
            results = await search_engine.search(
                query=query,
                user_id=str(ctx.author.id),
                guild_id=str(ctx.guild.id) if ctx.guild else None,
                max_results=5
            )
            
            # Update embed with results
            embed = discord.Embed(
                title="🔍 RAG Search Results",
                description=safe_embed_value(f"Query: {query}", DISCORD_EMBED_DESCRIPTION_LIMIT),
                color=discord.Color.blue()
            )
            
            if not results:
                embed.add_field(
                    name="❌ No Results",
                    value="No matching documents found.",
                    inline=False
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
                        name=f"{i+1}. {safe_embed_value(result.title, 256)}",
                        value=safe_embed_value(result_value),
                        inline=False
                    )
            
            await message.edit(embed=embed)
            
        except Exception as e:
            logger.error(f"[RAG Commands] Search command failed: {e}")
            embed = discord.Embed(
                title="❌ Search Error",
                description=safe_embed_value(f"Search failed: {str(e)}"),
                color=discord.Color.red()
            )
            await ctx.send(embed=embed)
    
    @rag_group.command(name='test')
    async def rag_test(self, ctx):
        """Run RAG system tests."""
        try:
            embed = discord.Embed(
                title="🧪 RAG System Tests",
                description="Running comprehensive system tests...",
                color=discord.Color.orange()
            )
            message = await ctx.send(embed=embed)
            
            test_results = []
            
            # Test 1: System initialization
            try:
                search_engine = await get_hybrid_search()
                test_results.append(("✅", "System Initialization", "Passed"))
            except Exception as e:
                error_msg = safe_embed_value(str(e), 50)
                test_results.append(("❌", "System Initialization", f"Failed: {error_msg}"))
            
            # Test 2: Environment validation
            env_valid, env_issues = validate_rag_environment()
            if env_valid:
                test_results.append(("✅", "Environment Validation", "Passed"))
            else:
                test_results.append(("⚠️", "Environment Validation", f"{len(env_issues)} issues"))
            
            # Test 3: Search functionality
            try:
                if 'search_engine' in locals():
                    results = await search_engine.search("test query", max_results=1)
                    test_results.append(("✅", "Search Functionality", f"Returned {len(results)} results"))
                else:
                    test_results.append(("❌", "Search Functionality", "Search engine not available"))
            except Exception as e:
                error_msg = safe_embed_value(str(e), 50)
                test_results.append(("❌", "Search Functionality", f"Failed: {error_msg}"))
            
            # Test 4: Collection stats
            try:
                if 'search_engine' in locals():
                    stats = await search_engine.get_stats()
                    if 'collection_stats' in stats:
                        chunks = stats['collection_stats'].get('total_chunks', 0)
                        test_results.append(("✅", "Collection Access", f"{chunks} chunks available"))
                    else:
                        test_results.append(("⚠️", "Collection Access", "No collection stats"))
                else:
                    test_results.append(("❌", "Collection Access", "Search engine not available"))
            except Exception as e:
                error_msg = safe_embed_value(str(e), 50)
                test_results.append(("❌", "Collection Access", f"Failed: {error_msg}"))
            
            # Update embed with test results
            embed = discord.Embed(
                title="🧪 RAG System Test Results",
                color=discord.Color.green()
            )
            
            # Add test results to embed with safe truncation
            for icon, test_name, result in test_results:
                embed.add_field(
                    name=f"{icon} {safe_embed_value(test_name, 256)}",
                    value=safe_embed_value(result),
                    inline=False
                )
            
            # Overall status
            passed_tests = sum(1 for icon, _, _ in test_results if icon == "✅")
            total_tests = len(test_results)
            
            summary_text = (
                f"**Passed:** {passed_tests}/{total_tests} tests\n"
                f"**Status:** {'✅ All systems operational' if passed_tests == total_tests else '⚠️ Some issues detected'}"
            )
            
            embed.add_field(
                name="📊 Summary",
                value=safe_embed_value(summary_text),
                inline=False
            )
            
            await message.edit(embed=embed)
            
        except Exception as e:
            logger.error(f"[RAG Commands] Test command failed: {e}")
            embed = discord.Embed(
                title="❌ Test Error",
                description=safe_embed_value(f"Test execution failed: {str(e)}"),
                color=discord.Color.red()
            )
            await ctx.send(embed=embed)


async def setup(bot):
    """Set up RAG commands."""
    await bot.add_cog(RAGCommands(bot))
    logger.info("✔ RAG commands cog loaded")
