"""
Memory-related commands for the Discord bot.
"""
import logging
from typing import Optional
import discord
from discord.ext import commands

# Import bot modules
from ..memory import add_memory, get_profile, save_profile, get_server_profile, save_server_profile
from ..context import get_conversation_history, reset_context
from ..logs import log_command
from ..config import load_config

# Load configuration
config = load_config()

# Command group for memory-related commands
group = commands.Group(
    name="memory",
    description="Manage your memories and server memories",
    invoke_without_command=True
)

@group.command(name="add")
async def add_memory_cmd(ctx, *, content: str):
    """Add a memory to your profile."""
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
        if len(profile['memories']) > config["MAX_MEMORIES"]:
            profile['memories'] = profile['memories'][-config["MAX_MEMORIES"]:]
        
        # Save the profile
        if save_profile(profile):
            await ctx.send(f"✅ Memory added! You now have {len(profile['memories'])} memories.")
            log_command(ctx, "memory_add", {"memory_length": len(content), "total_memories": len(profile['memories'])})
        else:
            await ctx.send("❌ Failed to save memory. Please try again later.")
            log_command(ctx, "memory_add_error", {"error": "Failed to save profile"}, success=False)
    
    except Exception as e:
        logging.error(f"Error in add_memory_cmd: {e}", exc_info=True)
        await ctx.send("❌ An error occurred while adding your memory.")
        log_command(ctx, "memory_add_error", {"error": str(e)}, success=False)

@group.command(name="list")
async def list_memories_cmd(ctx, limit: int = 5):
    """List your recent memories."""
    try:
        # Get user's profile
        profile = get_profile(str(ctx.author.id), str(ctx.author))
        
        if not profile.get('memories'):
            await ctx.send("You don't have any memories yet!")
            return
        
        # Limit the number of memories to show
        memories = profile['memories'][-min(limit, len(profile['memories'])):]
        
        # Format the response
        response = [f"**Your recent memories (showing {len(memories)} of {len(profile['memories'])}):**\n"]
        
        for i, memory in enumerate(reversed(memories), 1):
            content = memory.get('content', 'No content')
            timestamp = memory.get('timestamp', 'Unknown time')
            context = memory.get('context', 'No context')
            
            response.append(f"**{i}.** {content[:100]}{'...' if len(content) > 100 else ''}")
            response.append(f"   *{timestamp} - {context}*\n")
        
        # Send the response in chunks to avoid Discord's message length limit
        await ctx.send(''.join(response)[:1900])
        log_command(ctx, "memory_list", {"limit": limit})
    
    except Exception as e:
        logging.error(f"Error in list_memories_cmd: {e}", exc_info=True)
        await ctx.send("❌ An error occurred while listing your memories.")
        log_command(ctx, "memory_list_error", {"error": str(e)}, success=False)

@group.command(name="clear")
async def clear_memories_cmd(ctx):
    """Clear all your memories."""
    try:
        # Get user's profile
        profile = get_profile(str(ctx.author.id), str(ctx.author))
        
        if not profile.get('memories'):
            await ctx.send("You don't have any memories to clear!")
            return
        
        # Confirm before clearing
        confirm_msg = await ctx.send(
            "⚠️ **Are you sure you want to clear all your memories?** This cannot be undone. "
            "Type `confirm` to proceed or `cancel` to cancel."
        )
        
        def check(m):
            return m.author == ctx.author and m.channel == ctx.channel and m.content.lower() in ['confirm', 'cancel']
        
        try:
            msg = await ctx.bot.wait_for('message', check=check, timeout=30.0)
            
            if msg.content.lower() == 'confirm':
                # Clear memories
                memory_count = len(profile.get('memories', []))
                profile['memories'] = []
                
                if save_profile(profile):
                    await ctx.send(f"✅ Cleared {memory_count} memories!")
                    log_command(ctx, "memory_clear", {"cleared_count": memory_count})
                else:
                    await ctx.send("❌ Failed to clear memories. Please try again later.")
                    log_command(ctx, "memory_clear_error", {"error": "Failed to save profile"}, success=False)
            else:
                await ctx.send("✅ Memory clear cancelled.")
                log_command(ctx, "memory_clear_cancelled", {})
        
        except asyncio.TimeoutError:
            await ctx.send("⏱️ Memory clear timed out. Please try again if you want to clear your memories.")
            log_command(ctx, "memory_clear_timeout", {}, success=False)
    
    except Exception as e:
        logging.error(f"Error in clear_memories_cmd: {e}", exc_info=True)
        await ctx.send("❌ An error occurred while clearing your memories.")
        log_command(ctx, "memory_clear_error", {"error": str(e)}, success=False)

@group.command(name="server")
@commands.has_permissions(manage_guild=True)
async def server_memory_cmd(ctx, action: str, *, content: Optional[str] = None):
    """Manage server memories (Admin only)."""
    if not ctx.guild:
        await ctx.send("This command can only be used in a server.")
        return
    
    try:
        action = action.lower()
        
        if action == 'add' and content:
            # Add a server memory
            server_profile = get_server_profile(str(ctx.guild.id))
            
            memory = {
                'content': content,
                'added_by': str(ctx.author),
                'timestamp': discord.utils.utcnow().isoformat(),
                'context': f"Added in #{ctx.channel.name} by {ctx.author}"
            }
            
            if 'memories' not in server_profile:
                server_profile['memories'] = []
            
            server_profile['memories'].append(memory)
            
            # Enforce server memory limit
            if len(server_profile['memories']) > config["MAX_SERVER_MEMORY"]:
                server_profile['memories'] = server_profile['memories'][-config["MAX_SERVER_MEMORY"]:]
            
            # Save the server profile
            if save_server_profile(str(ctx.guild.id)):
                await ctx.send(f"✅ Server memory added! The server now has {len(server_profile['memories'])} memories.")
                log_command(ctx, "server_memory_add", 
                          {"memory_length": len(content), 
                           "total_memories": len(server_profile['memories'])})
            else:
                await ctx.send("❌ Failed to save server memory. Please try again later.")
                log_command(ctx, "server_memory_add_error", 
                          {"error": "Failed to save server profile"}, success=False)
        
        elif action == 'list':
            # List server memories
            server_profile = get_server_profile(str(ctx.guild.id))
            
            if not server_profile.get('memories'):
                await ctx.send("This server doesn't have any memories yet!")
                return
            
            # Format the response
            response = [f"**Server memories (showing all {len(server_profile['memories'])}):**\n"]
            
            for i, memory in enumerate(reversed(server_profile['memories']), 1):
                content = memory.get('content', 'No content')
                added_by = memory.get('added_by', 'Unknown user')
                timestamp = memory.get('timestamp', 'Unknown time')
                
                response.append(f"**{i}.** {content[:100]}{'...' if len(content) > 100 else ''}")
                response.append(f"   *Added by {added_by} on {timestamp}*\n")
            
            # Send the response in chunks to avoid Discord's message length limit
            await ctx.send(''.join(response)[:1900])
            log_command(ctx, "server_memory_list", {})
        
        elif action == 'clear':
            # Clear server memories (with confirmation)
            server_profile = get_server_profile(str(ctx.guild.id))
            
            if not server_profile.get('memories'):
                await ctx.send("This server doesn't have any memories to clear!")
                return
            
            # Confirm before clearing
            confirm_msg = await ctx.send(
                "⚠️ **Are you sure you want to clear all server memories?** This cannot be undone. "
                "Type `confirm` to proceed or `cancel` to cancel."
            )
            
            def check(m):
                return m.author == ctx.author and m.channel == ctx.channel and m.content.lower() in ['confirm', 'cancel']
            
            try:
                msg = await ctx.bot.wait_for('message', check=check, timeout=30.0)
                
                if msg.content.lower() == 'confirm':
                    # Clear server memories
                    memory_count = len(server_profile.get('memories', []))
                    server_profile['memories'] = []
                    
                    if save_server_profile(str(ctx.guild.id)):
                        await ctx.send(f"✅ Cleared {memory_count} server memories!")
                        log_command(ctx, "server_memory_clear", 
                                  {"cleared_count": memory_count})
                    else:
                        await ctx.send("❌ Failed to clear server memories. Please try again later.")
                        log_command(ctx, "server_memory_clear_error", 
                                  {"error": "Failed to save server profile"}, success=False)
                else:
                    await ctx.send("✅ Server memory clear cancelled.")
                    log_command(ctx, "server_memory_clear_cancelled", {})
            
            except asyncio.TimeoutError:
                await ctx.send("⏱️ Server memory clear timed out. Please try again if you want to clear server memories.")
                log_command(ctx, "server_memory_clear_timeout", {}, success=False)
        
        else:
            await ctx.send("❌ Invalid action. Use `add`, `list`, or `clear`.")
            log_command(ctx, "server_memory_invalid_action", 
                       {"action": action}, success=False)
    
    except Exception as e:
        logging.error(f"Error in server_memory_cmd: {e}", exc_info=True)
        await ctx.send("❌ An error occurred while processing your request.")
        log_command(ctx, "server_memory_error", 
                   {"error": str(e), "action": action}, success=False)

# Register the command group
def setup(bot):
    bot.add_command(group)
