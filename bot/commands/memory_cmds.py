"""
Memory-related commands for the Discord bot.

This module provides commands to manage user and server memories.
"""
import logging

import discord
import asyncio
from discord.ext import commands

# Import bot modules
from ..memory import get_profile, save_profile, get_server_profile, save_server_profile
import logging
from bot.logger import log_command

logger = logging.getLogger(__name__)
from ..config import load_config
from ..router import get_router

# Load configuration
config = load_config()

class MemoryCommands(commands.Cog):
    """Commands for managing user and server memories."""
    
    def __init__(self, bot):
        self.bot = bot
        self.config = load_config()
        self.router = bot.router
        self.prefix = self.config.get('COMMAND_PREFIX', '!')
    
    @commands.group(name="memory", invoke_without_command=True)
    async def memory_group(self, ctx):
        """Memory management commands.
        
        Usage:
        !memory add <content> - Add a new memory
        !memory list [limit] - List your recent memories (default: 5)
        !memory clear - Clear all your memories
        """
        if ctx.invoked_subcommand is None:
            await ctx.send_help(ctx.command)

    @memory_group.command(name="add")
    async def add_memory_cmd(self, ctx, *, content: str):
        """Add a memory to your profile.
        
        Example:
        !memory add I prefer to be called by my nickname, not my full name
        """
        try:
            # Get user's profile
            profile = get_profile(str(ctx.author.id), str(ctx.author))
            
            # Add the memory
            memory = {
                'content': content,
                'timestamp': discord.utils.utcnow().isoformat(),
                'context': f"Added via command in {ctx.channel.name if hasattr(ctx.channel, 'name') else 'DM'}"
            }
            
            if 'memories' not in profile:
                profile['memories'] = []
            
            profile['memories'].append(memory)
            
            # Enforce memory limit
            if len(profile['memories']) > self.config["MAX_MEMORIES"]:
                profile['memories'] = profile['memories'][-self.config["MAX_MEMORIES"]:]
            
            # Save the profile
            if save_profile(profile):
                await ctx.send(f"✅ Memory added! You now have {len(profile['memories'])} memories.")
                log_command(ctx, f"Added memory: {content[:50]}...")
            else:
                await ctx.send("❌ Failed to save memory. Please try again.")
                logging.error(f"Failed to save memory for user {ctx.author.id}")
                
        except Exception as e:
            logging.error(f"Error in add_memory_cmd: {str(e)}", exc_info=True)
            await ctx.send("❌ An error occurred while adding the memory.")
            log_command(ctx, "memory_add_error", {"error": str(e)}, success=False)

    @memory_group.command(name="list")
    async def list_memories_cmd(self, ctx, limit: int = 5):
        """List your recent memories.
        
        Args:
            limit: Number of memories to show (default: 5, max: 20)
            
        Example:
        !memory list 3 - Show your 3 most recent memories
        """
        try:
            # Enforce a reasonable limit
            limit = min(max(1, limit), 20)
            
            profile = get_profile(str(ctx.author.id))
            
            if not profile or 'memories' not in profile or not profile['memories']:
                await ctx.send("You don't have any memories yet. Use `!memory add <content>` to add one!")
                return
                
            # Limit the number of memories to show
            memories = profile['memories'][-limit:]
            
            if not memories:
                await ctx.send("No memories found.")
                return
                
            # Create an embed to display memories
            embed = discord.Embed(
                title=f"Your Recent Memories (Last {len(memories)} of {len(profile['memories'])})",
                color=discord.Color.blue()
            )
            
            for i, memory in enumerate(reversed(memories), 1):
                timestamp = memory.get('timestamp', 'Unknown')
                context = memory.get('context', 'No context')
                embed.add_field(
                    name=f"Memory #{len(profile['memories']) - len(memories) + i}",
                    value=f"{memory['content']}\n*{context} - {timestamp}*",
                    inline=False
                )
                
            await ctx.send(embed=embed)
            log_command(ctx, "Listed memories")
            
        except Exception as e:
            logging.error(f"Error in list_memories_cmd: {str(e)}", exc_info=True)
            await ctx.send("❌ An error occurred while retrieving memories.")
            log_command(ctx, "memory_list_error", {"error": str(e)}, success=False)

    @memory_group.command(name="clear")
    @commands.cooldown(1, 30, commands.BucketType.user)
    async def clear_memories_cmd(self, ctx):
        """Clear all your memories after confirmation."""
        try:
            # Ask for confirmation
            confirm_msg = await ctx.send("⚠️ Are you sure you want to delete ALL your memories? This cannot be undone. Type `yes` to confirm.")
            
            def check(m):
                return m.author == ctx.author and m.channel == ctx.channel and m.content.lower() == 'yes'
            
            try:
                await ctx.bot.wait_for('message', check=check, timeout=30.0)
            except asyncio.TimeoutError:
                await confirm_msg.edit(content="Memory clear cancelled due to timeout.")
                return
                
            # Get and clear memories
            profile = get_profile(str(ctx.author.id))
            if not profile:
                await ctx.send("No profile found to clear.")
                return
                
            if 'memories' in profile and profile['memories']:
                memory_count = len(profile['memories'])
                profile['memories'] = []
                
                if save_profile(profile):
                    await ctx.send(f"✅ Successfully cleared {memory_count} memories.")
                    log_command(ctx, f"Cleared {memory_count} memories")
                else:
                    await ctx.send("❌ Failed to clear memories. Please try again.")
            else:
                await ctx.send("No memories found to clear.")
                
        except Exception as e:
            logging.error(f"Error in clear_memories_cmd: {str(e)}", exc_info=True)
            await ctx.send("❌ An error occurred while clearing memories.")
        finally:
            # Reset cooldown if command failed
            self.clear_memories_cmd.reset_cooldown(ctx)

    @commands.group(name="server-memory", description="Manage server memories (Admin only)", invoke_without_command=True)
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def server_memory_group(self, ctx):
        """Manage server memories (Admin only).
        
        Subcommands:
        add <content> - Add a server memory
        list - List all server memories
        clear - Clear all server memories
        """
        if ctx.invoked_subcommand is None:
            await ctx.send_help(ctx.command)
    
    @server_memory_group.command(name="add")
    @commands.has_permissions(administrator=True)
    async def server_memory_add(self, ctx, *, content: str):
        """Add a memory to the server's profile."""
        try:
            # Get server profile
            server_id = str(ctx.guild.id)
            profile = get_server_profile(server_id, ctx.guild.name)
            
            # Add the memory
            memory = {
                'content': content,
                'timestamp': discord.utils.utcnow().isoformat(),
                'added_by': str(ctx.author),
                'context': f"Added in #{ctx.channel.name if hasattr(ctx.channel, 'name') else 'unknown'}"
            }
            
            if 'memories' not in profile:
                profile['memories'] = []
                
            profile['memories'].append(memory)
            
            # Enforce memory limit
            if len(profile['memories']) > self.config["MAX_SERVER_MEMORIES"]:
                profile['memories'] = profile['memories'][-self.config["MAX_SERVER_MEMORIES"]:]
            
            # Save the profile
            if save_server_profile(server_id, profile):
                await ctx.send(f"✅ Server memory added! There are now {len(profile['memories'])} server memories.")
                log_command(ctx, f"Added server memory: {content[:50]}...")
            else:
                await ctx.send("❌ Failed to save server memory. Please try again.")
                
        except Exception as e:
            logging.error(f"Error adding server memory: {str(e)}", exc_info=True)
            await ctx.send("❌ An error occurred while adding the server memory.")
    
    @server_memory_group.command(name="list")
    @commands.has_permissions(administrator=True)
    async def server_memory_list(self, ctx):
        """List all server memories."""
        try:
            profile = get_server_profile(str(ctx.guild.id))
            
            if not profile or 'memories' not in profile or not profile['memories']:
                await ctx.send("No server memories found. Use `!server-memory add <content>` to add one!")
                return
                
            # Create an embed to display memories
            embed = discord.Embed(
                title=f"Server Memories ({len(profile['memories'])} total)",
                color=discord.Color.green()
            )
            
            for i, memory in enumerate(reversed(profile['memories']), 1):
                added_by = memory.get('added_by', 'Unknown')
                timestamp = memory.get('timestamp', 'Unknown')
                context = memory.get('context', 'No context')
                
                embed.add_field(
                    name=f"Memory #{i}",
                    value=f"{memory['content']}\n*Added by {added_by} - {context} - {timestamp}*",
                    inline=False
                )
                
                # Discord has a limit of 25 fields per embed
                if i >= 25:
                    embed.set_footer(text=f"Showing 25 most recent of {len(profile['memories'])} memories.")
                    break
                    
            await ctx.send(embed=embed)
            log_command(ctx, "Listed server memories")
            
        except Exception as e:
            logging.error(f"Error listing server memories: {str(e)}", exc_info=True)
            await ctx.send("❌ An error occurred while retrieving server memories.")
    
    @server_memory_group.command(name="clear")
    @commands.has_permissions(administrator=True)
    @commands.cooldown(1, 60, commands.BucketType.guild)
    async def server_memory_clear(self, ctx):
        """Clear all server memories after confirmation."""
        try:
            # Ask for confirmation
            confirm_msg = await ctx.send("⚠️ Are you sure you want to delete ALL server memories? This cannot be undone. Type `yes` to confirm.")
            
            def check(m):
                return m.author == ctx.author and m.channel == ctx.channel and m.content.lower() == 'yes'
            
            try:
                await ctx.bot.wait_for('message', check=check, timeout=30.0)
            except asyncio.TimeoutError:
                await confirm_msg.edit(content="Server memory clear cancelled due to timeout.")
                return
                
            # Clear server memories
            profile = get_server_profile(str(ctx.guild.id))
            if not profile or 'memories' not in profile or not profile['memories']:
                await ctx.send("No server memories found to clear.")
                return
                
            memory_count = len(profile['memories'])
            profile['memories'] = []
            
            if save_server_profile(str(ctx.guild.id), profile):
                await ctx.send(f"✅ Successfully cleared {memory_count} server memories.")
                log_command(ctx, f"Cleared {memory_count} server memories")
            else:
                await ctx.send("❌ Failed to clear server memories. Please try again.")
                
        except Exception as e:
            logging.error(f"Error clearing server memories: {str(e)}", exc_info=True)
            await ctx.send("❌ An error occurred while clearing server memories.")
        finally:
            # Reset cooldown if command failed
            self.server_memory_clear.reset_cooldown(ctx)

async def setup(bot):
    """Add memory commands to the bot."""
    if not bot.get_cog('MemoryCommands'):
        await bot.add_cog(MemoryCommands(bot))
    else:
        logger.warning("'MemoryCommands' cog already loaded, skipping setup.")
