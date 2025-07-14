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
    
    @commands.group(name='tts', invoke_without_command=True)
    async def tts_group(self, ctx: commands.Context, *, text: Optional[str] = None):
        """Base command for TTS functionality."""
        if text:
            # If there's text after !tts, treat it as a one-off TTS request
            await self.speak(ctx, text)
        else:
            # Otherwise, show help
            await ctx.send("Please specify 'on', 'off', or text to speak.")
    
    @tts_group.command(name='on')
    async def tts_on(self, ctx: commands.Context):
        """Enable TTS for your messages."""
        tts_state.set_user_tts(ctx.author.id, True)
        await ctx.send("✅ TTS responses enabled for you.")
    
    @tts_group.command(name='off')
    async def tts_off(self, ctx: commands.Context):
        """Disable TTS for your messages."""
        tts_state.set_user_tts(ctx.author.id, False)
        await ctx.send("✅ TTS responses disabled for you.")
    
    @tts_group.command(name='all')
    @commands.guild_only()  # Block usage in DMs
    @commands.has_permissions(administrator=True)
    async def tts_all(self, ctx: commands.Context, setting: str):
        """Admin-only: Enable/disable TTS globally. Only works in servers, not DMs."""
        setting = setting.lower()
        if setting == 'on':
            tts_state.set_global_tts(True)
            await ctx.send("✅ TTS responses enabled globally.")
        elif setting == 'off':
            tts_state.set_global_tts(False)
            await ctx.send("✅ TTS responses disabled globally.")
        else:
            await ctx.send("❌ Invalid setting. Use 'on' or 'off'.")
    
    @commands.command(name='speak')
    async def speak(self, ctx: commands.Context, *, text: Optional[str] = None):
        """Make the next response TTS or speak the given text."""
        tts_state.set_one_time_tts(ctx.author.id)
        
        if text:
            # If text is provided, process it immediately
            await self.router.handle(ctx.message, text)
        else:
            # Otherwise, just set the flag for the next response
            await ctx.send("🗣️ The next response will be spoken.")
    
    @commands.command(name='say')
    async def say(self, ctx: commands.Context, *, text: str):
        """Make the bot say exactly what you type."""
        # Use TTS manager to synthesize and send the text
        if not tts_manager.is_available():
            await ctx.send("❌ TTS is not available at the moment.")
            return
        
        try:
            # Use the correct method name generate_tts instead of synthesize
            audio_path = await tts_manager.generate_tts(text, tts_manager.voice)
            await ctx.send(file=discord.File(audio_path))
            logging.debug(f"🔊 Direct TTS response sent for '{text[:30]}...'")
        except Exception as e:
            logging.error(f"Error in say command: {e}", exc_info=True)
            await ctx.send(f"❌ An error occurred while generating TTS: {str(e)}")
    
    # Note: The standalone tts-all command is removed to avoid duplication
    # The functionality is now handled by the @tts_group.command(name='all') subcommand

async def setup(bot):
    """Add TTS commands to the bot."""
    await bot.add_cog(TTSCommands(bot))
