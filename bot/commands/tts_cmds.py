"""
TTS (Text-to-Speech) commands for the Discord bot.
"""
import logging
from typing import Optional
import discord
from discord.ext import commands

# Import bot modules
from ..tts import (
    is_tts_enabled, set_tts_enabled, set_global_tts,
    get_available_voices, set_voice, set_speaking_rate,
    send_tts_reply, DIA_AVAILABLE
)
from ..logs import log_command
from ..memory import get_profile, save_profile, get_server_profile, save_server_profile

# Command group for TTS commands
group = commands.Group(
    name="tts",
    description="Control text-to-speech settings",
    invoke_without_command=True
)

@group.command(name="on")
async def tts_on(ctx):
    """Enable TTS for your messages."""
    try:
        if not DIA_AVAILABLE:
            await ctx.send("❌ TTS is not available. The DIA TTS package is not installed.")
            log_command(ctx, "tts_on_error", {"error": "DIA TTS not available"}, success=False)
            return
        
        # Update user's TTS preference
        profile = get_profile(str(ctx.author.id), str(ctx.author))
        if 'preferences' not in profile:
            profile['preferences'] = {}
        
        profile['preferences']['tts_enabled'] = True
        
        if save_profile(profile):
            await ctx.send("✅ TTS has been enabled for your messages!")
            log_command(ctx, "tts_on", {})
        else:
            await ctx.send("❌ Failed to update your TTS preferences. Please try again later.")
            log_command(ctx, "tts_on_error", {"error": "Failed to save profile"}, success=False)
    
    except Exception as e:
        logging.error(f"Error in tts_on: {e}", exc_info=True)
        await ctx.send("❌ An error occurred while updating your TTS preferences.")
        log_command(ctx, "tts_on_error", {"error": str(e)}, success=False)

@group.command(name="off")
async def tts_off(ctx):
    """Disable TTS for your messages."""
    try:
        # Update user's TTS preference
        profile = get_profile(str(ctx.author.id), str(ctx.author))
        if 'preferences' not in profile:
            profile['preferences'] = {}
        
        profile['preferences']['tts_enabled'] = False
        
        if save_profile(profile):
            await ctx.send("🔇 TTS has been disabled for your messages.")
            log_command(ctx, "tts_off", {})
        else:
            await ctx.send("❌ Failed to update your TTS preferences. Please try again later.")
            log_command(ctx, "tts_off_error", {"error": "Failed to save profile"}, success=False)
    
    except Exception as e:
        logging.error(f"Error in tts_off: {e}", exc_info=True)
        await ctx.send("❌ An error occurred while updating your TTS preferences.")
        log_command(ctx, "tts_off_error", {"error": str(e)}, success=False)

@group.command(name="status")
async def tts_status(ctx):
    """Check your current TTS settings."""
    try:
        profile = get_profile(str(ctx.author.id), str(ctx.author))
        tts_enabled = profile.get('preferences', {}).get('tts_enabled', False)
        
        status = "enabled" if tts_enabled else "disabled"
        status_emoji = "✅" if tts_enabled else "🔇"
        
        response = [
            f"{status_emoji} **TTS Status**",
            f"• **Your TTS:** {status}",
            f"• **Global TTS:** {'enabled' if await is_tts_enabled(ctx.author.id) else 'disabled'}"
        ]
        
        if DIA_AVAILABLE:
            response.append("\nUse `!tts on` to enable or `!tts off` to disable TTS for your messages.")
        else:
            response.append("\n⚠️ TTS functionality is currently unavailable. The DIA TTS package is not installed.")
        
        await ctx.send("\n".join(response))
        log_command(ctx, "tts_status", {"status": status})
    
    except Exception as e:
        logging.error(f"Error in tts_status: {e}", exc_info=True)
        await ctx.send("❌ An error occurred while checking your TTS status.")
        log_command(ctx, "tts_status_error", {"error": str(e)}, success=False)

@group.command(name="say")
async def tts_say(ctx, *, text: str):
    """Make the bot say something with TTS."""
    if not DIA_AVAILABLE:
        await ctx.send("❌ TTS is not available. The DIA TTS package is not installed.")
        log_command(ctx, "tts_say_error", {"error": "DIA TTS not available"}, success=False)
        return
    
    if not text.strip():
        await ctx.send("Please provide some text for me to say!")
        return
    
    try:
        # Generate and send TTS
        success = await send_tts_reply(ctx.message, text)
        
        if success:
            log_command(ctx, "tts_say", {"text_length": len(text)})
        else:
            await ctx.send("❌ Failed to generate TTS. Please try again later.")
            log_command(ctx, "tts_say_error", {"error": "Failed to generate TTS"}, success=False)
    
    except Exception as e:
        logging.error(f"Error in tts_say: {e}", exc_info=True)
        await ctx.send("❌ An error occurred while generating TTS.")
        log_command(ctx, "tts_say_error", {"error": str(e)}, success=False)

@group.command(name="voices")
async def list_voices(ctx):
    """List available TTS voices."""
    if not DIA_AVAILABLE:
        await ctx.send("❌ TTS is not available. The DIA TTS package is not installed.")
        log_command(ctx, "tts_voices_error", {"error": "DIA TTS not available"}, success=False)
        return
    
    try:
        voices = get_available_voices()
        
        if not voices:
            await ctx.send("❌ No TTS voices available.")
            log_command(ctx, "tts_voices_error", {"error": "No voices available"}, success=False)
            return
        
        response = ["**Available TTS Voices:**"]
        
        for voice_id, voice_name in voices.items():
            response.append(f"• `{voice_id}` - {voice_name}")
        
        response.append("\nUse `!tts set_voice <voice_id>` to change your voice.")
        
        # Send in chunks to avoid message length limits
        await ctx.send("\n".join(response)[:2000])
        log_command(ctx, "tts_voices", {"voice_count": len(voices)})
    
    except Exception as e:
        logging.error(f"Error in list_voices: {e}", exc_info=True)
        await ctx.send("❌ An error occurred while listing TTS voices.")
        log_command(ctx, "tts_voices_error", {"error": str(e)}, success=False)

@group.command(name="set_voice")
async def set_voice_cmd(ctx, voice_id: str):
    """Set your preferred TTS voice."""
    if not DIA_AVAILABLE:
        await ctx.send("❌ TTS is not available. The DIA TTS package is not installed.")
        log_command(ctx, "tts_set_voice_error", {"error": "DIA TTS not available"}, success=False)
        return
    
    try:
        voices = get_available_voices()
        
        if voice_id not in voices:
            await ctx.send("❌ Invalid voice ID. Use `!tts voices` to see available voices.")
            log_command(ctx, "tts_set_voice_error", 
                       {"error": f"Invalid voice ID: {voice_id}"}, success=False)
            return
        
        # Update user's voice preference
        profile = get_profile(str(ctx.author.id), str(ctx.author))
        if 'preferences' not in profile:
            profile['preferences'] = {}
        
        profile['preferences']['tts_voice'] = voice_id
        
        if save_profile(profile):
            await ctx.send(f"✅ Your TTS voice has been set to: **{voices[voice_id]}**")
            log_command(ctx, "tts_set_voice", {"voice_id": voice_id})
        else:
            await ctx.send("❌ Failed to update your voice preference. Please try again later.")
            log_command(ctx, "tts_set_voice_error", 
                       {"error": "Failed to save profile"}, success=False)
    
    except Exception as e:
        logging.error(f"Error in set_voice_cmd: {e}", exc_info=True)
        await ctx.send("❌ An error occurred while updating your voice preference.")
        log_command(ctx, "tts_set_voice_error", {"error": str(e)}, success=False)

@group.command(name="speed")
async def set_speed(ctx, speed: float):
    """Set the TTS speaking rate (0.5 to 2.0)."""
    try:
        # Validate speed
        speed = float(speed)
        if speed < 0.5 or speed > 2.0:
            await ctx.send("❌ Speed must be between 0.5 and 2.0.")
            log_command(ctx, "tts_set_speed_error", 
                       {"error": f"Speed out of range: {speed}"}, success=False)
            return
        
        # Update user's speed preference
        profile = get_profile(str(ctx.author.id), str(ctx.author))
        if 'preferences' not in profile:
            profile['preferences'] = {}
        
        profile['preferences']['tts_speed'] = speed
        
        if save_profile(profile):
            await ctx.send(f"✅ Your TTS speed has been set to: **{speed:.1f}x**")
            log_command(ctx, "tts_set_speed", {"speed": speed})
        else:
            await ctx.send("❌ Failed to update your speed preference. Please try again later.")
            log_command(ctx, "tts_set_speed_error", 
                       {"error": "Failed to save profile"}, success=False)
    
    except ValueError:
        await ctx.send("❌ Invalid speed. Please provide a number between 0.5 and 2.0.")
        log_command(ctx, "tts_set_speed_error", 
                   {"error": f"Invalid speed: {speed}"}, success=False)
    except Exception as e:
        logging.error(f"Error in set_speed: {e}", exc_info=True)
        await ctx.send("❌ An error occurred while updating your speed preference.")
        log_command(ctx, "tts_set_speed_error", {"error": str(e)}, success=False)

@group.command(name="all")
@commands.has_permissions(manage_guild=True)
async def tts_all(ctx, status: str):
    """Enable or disable TTS for all users in this server (Admin only)."""
    if not ctx.guild:
        await ctx.send("This command can only be used in a server.")
        return
    
    try:
        status = status.lower()
        
        if status not in ['on', 'off']:
            await ctx.send("❌ Invalid status. Use `on` or `off`.")
            log_command(ctx, "tts_all_error", 
                       {"error": f"Invalid status: {status}"}, success=False)
            return
        
        # Update server's TTS preference
        server_profile = get_server_profile(str(ctx.guild.id))
        if 'preferences' not in server_profile:
            server_profile['preferences'] = {}
        
        server_profile['preferences']['tts_global_enabled'] = (status == 'on')
        
        if save_server_profile(str(ctx.guild.id)):
            status_text = "enabled" if status == 'on' else "disabled"
            await ctx.send(f"✅ TTS has been {status_text} for all users in this server.")
            log_command(ctx, f"tts_all_{status}", {})
        else:
            await ctx.send("❌ Failed to update server TTS settings. Please try again later.")
            log_command(ctx, "tts_all_error", 
                       {"error": "Failed to save server profile"}, success=False)
    
    except Exception as e:
        logging.error(f"Error in tts_all: {e}", exc_info=True)
        await ctx.send("❌ An error occurred while updating server TTS settings.")
        log_command(ctx, "tts_all_error", {"error": str(e)}, success=False)

# Register the command group
def setup(bot):
    bot.add_command(group)
