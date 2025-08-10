"""
TTS (Text-to-Speech) commands for the Discord bot.

This module provides commands to control TTS settings and behavior.
"""
import logging
import io
from typing import Optional

import discord
import logging
from discord.ext import commands

logger = logging.getLogger(__name__)

from bot.tts.state import tts_state

from bot.router import get_router

class TTSCommands(commands.Cog):
    """Commands for controlling TTS functionality."""
    
    def __init__(self, bot):
        self.bot = bot
        self.router = bot.router
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
    async def speak(self, ctx: commands.Context, *, text: Optional[str] = None, pcm16: bool = False):
        """Make the next response TTS or speak the given text."""
        # Don't set one_time_tts when providing text directly - use voice_only instead
        # This prevents duplicate responses by avoiding double TTS triggering
        
        user_id = str(ctx.author.id)
        channel_id = str(ctx.channel.id)
        
        if text:
            # If text is provided, delegate to the 'say' command's logic for direct synthesis.
            logging.debug(f"🔊 Delegating !speak with text to !say handler: '{text[:30]}...'" ,
                         extra={'subsys': 'tts_cmds', 'event': 'speak.with_text', 
                                'user_id': user_id, 'channel_id': channel_id})
            await self.say(ctx, text=text)
        else:
            # Only set the flag when no text is provided (for next response)
            logging.debug(f"🔊 Setting one-time TTS flag for user {user_id}",
                         extra={'subsys': 'tts_cmds', 'event': 'speak.one_time_flag', 
                                'user_id': user_id, 'channel_id': channel_id})
            tts_state.set_one_time_tts(ctx.author.id)
            await ctx.send("🗯️ The next response will be spoken.")
    
    @commands.command(name='say')
    async def say(self, ctx: commands.Context, *, text: Optional[str] = None):
        """Make the bot say exactly what you type without generating AI response."""
        # Check for image attachments first
        if ctx.message.attachments:
            # Has attachments, process through VL/document flow with voice_only=True
            # The router will process the attachment, and we'll set a one-time TTS flag
            # to ensure the response is spoken, fulfilling the command's intent.
            logging.debug("📷 !say command with attachments, setting one-time TTS and dispatching to router.")
            tts_state.set_one_time_tts(ctx.author.id)
            await self.router.dispatch_message(ctx.message)
            return
        
        # No attachments, direct TTS with provided text
        if not self.bot.tts_manager.is_available():
            await ctx.send("❌ TTS is not available at the moment.")
            return
            
        # Handle empty text input by falling back to previous messages
        user_id = str(ctx.author.id)
        channel_id = str(ctx.channel.id)
        
        if text is None or text.strip() == "":
            logging.debug("🔍 !say command with empty text, attempting to find previous message",
                         extra={'subsys': 'tts_cmds', 'event': 'say.empty_text', 
                                'user_id': user_id, 'channel_id': channel_id})
            # Try to get previous message from the channel
            previous_messages = [msg async for msg in ctx.channel.history(limit=5) 
                               if msg.id != ctx.message.id and msg.author.id == ctx.author.id]
            
            if previous_messages:
                # Use the most recent message from the user
                text = previous_messages[0].content
                msg_id = str(previous_messages[0].id)
                logging.debug(f"✅ Found previous message to use for TTS: '{text[:30]}...'",
                             extra={'subsys': 'tts_cmds', 'event': 'say.fallback_found', 
                                    'user_id': user_id, 'channel_id': channel_id, 
                                    'msg_id': msg_id, 'content_length': len(text)})
            else:
                # No previous message found
                logging.warning("⚠️ No previous message found for !say with empty text",
                              extra={'subsys': 'tts_cmds', 'event': 'say.fallback_not_found', 
                                     'user_id': user_id, 'channel_id': channel_id})
                await ctx.send("❌ Please provide text to speak or send a message before using !say")
                return
        
        try:
            # Direct TTS synthesis (bypassing router for simple text)
            logging.debug(f"🔊 Direct TTS synthesis for !say command: '{text[:30]}...'")
            try:
                if hasattr(self.bot.tts_manager, 'generate_tts'):
                    # Prefer file-path generating API when available (compat with tests)
                    audio_path = await self.bot.tts_manager.generate_tts(text)
                    await ctx.send(file=discord.File(audio_path))
                else:
                    # Fallback to bytes-based API
                    audio_bytes = await self.bot.tts_manager.synthesize(text)
                    audio_stream = io.BytesIO(audio_bytes)
                    audio_stream.seek(0)
                    await ctx.send(file=discord.File(audio_stream, filename="tts_audio.wav"))
                logging.debug("✅ Direct TTS response sent successfully")
            except Exception as e:
                logging.error(f"Error in say command: {e}", exc_info=True)
                await ctx.send(f"❌ An error occurred while generating TTS: {str(e)}")
        except Exception as e:
            logging.error(f"Error in say command: {e}", exc_info=True)
            await ctx.send(f"❌ An error occurred while generating TTS: {str(e)}")
            return
    
    # Note: The standalone tts-all command is removed to avoid duplication
    # The functionality is now handled by the @tts_group.command(name='all') subcommand

async def setup(bot):
    """Add the TTS commands to the bot."""
    if not bot.get_cog('TTSCommands'):
        await bot.add_cog(TTSCommands(bot))
    else:
        logger.warning("'TTSCommands' cog already loaded, skipping setup.")
