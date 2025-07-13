"""
TTS (Text-to-Speech) commands for the Discord bot.

This module provides commands to control TTS settings and behavior.
"""
import logging
from typing import Optional

import discord
from discord.ext import commands

from bot.tts_state import tts_state
from bot.tts_manager import tts_manager
from bot.router import get_router

class TTSCommands(commands.Cog):
    """Commands for controlling TTS functionality."""
    
    def __init__(self, bot):
        self.bot = bot
        self.router = get_router()
        self.prefix = '!'
    
    @commands.group(name="tts", invoke_without_command=True)
    async def tts_group(self, ctx, *, content: Optional[str] = None):
        """Text-to-speech commands.
        
        Usage:
        !tts on/off - Enable/disable TTS for yourself
        !tts status - Check your TTS status
        !tts <text> --mode=tts - Speak the given text
        """
        if ctx.invoked_subcommand is not None or not content:
            if not ctx.invoked_subcommand:
                await ctx.send_help(ctx.command)
            return
            
        # If we get here, user did something like !tts hello
        await self.router.handle(ctx.message, content)
    
    @tts_group.command(name="on")
    async def tts_on(self, ctx):
        """Enable TTS for your messages."""
        if not tts_manager.is_available():
            await ctx.send("❌ TTS is not available. The Kokoro-ONNX TTS package is not installed or failed to initialize.")
            return
            
        tts_state.set_user_tts(ctx.author.id, True)
        await ctx.send("🔊 TTS enabled for you.")
    
    @tts_group.command(name="off")
    async def tts_off(self, ctx):
        """Disable TTS for your messages."""
        tts_state.set_user_tts(ctx.author.id, False)
        await ctx.send("🔊 TTS disabled for you.")
    
    @tts_group.command(name="status")
    async def tts_status(self, ctx):
        """Check your current TTS status."""
        status = "enabled" if tts_state.is_user_tts_enabled(ctx.author.id) else "disabled"
        global_status = "enabled" if tts_state.global_enabled else "disabled"
        await ctx.send(f"Your TTS is **{status}** (Global TTS is **{global_status}**).")
    
    @commands.command(name="tts-all", description="[Admin] Enable/disable TTS for all users")
    @commands.has_permissions(administrator=True)
    async def tts_all(self, ctx, state: Optional[str] = None):
        """Admin command to toggle TTS for everyone."""
        try:
            # Validate input
            if not state:
                await ctx.send("❌ Missing argument. Usage: `!tts-all [on|off]`")
                return
                
            state = state.lower()
            if state not in ['on', 'off']:
                await ctx.send("❌ Invalid argument. Use 'on' or 'off'.")
                return
                
            # Check TTS availability
            if not tts_manager.is_available():
                await ctx.send("⚠️ TTS is not available. The Kokoro-ONNX TTS package is not installed or failed to initialize. You can still enable it, but audio will not work until fixed.")

            # Update global TTS state
            enabled = (state == 'on')
            tts_state.global_enabled = enabled
            
            # Notify channel
            status = "enabled" if enabled else "disabled"
            await ctx.send(f"🔊 TTS has been **{status}** for all users.")
            
        except Exception as e:
            logging.error(f"Error in tts_all: {str(e)}", exc_info=True)
            await ctx.send("⚠️ An error occurred while processing your request. Please try again.")

async def setup(bot):
    """Add TTS commands to the bot."""
    await bot.add_cog(TTSCommands(bot))
