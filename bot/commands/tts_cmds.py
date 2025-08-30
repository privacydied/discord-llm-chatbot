"""
TTS (Text-to-Speech) commands for the Discord bot.

This module provides commands to control TTS settings and behavior.
"""
import logging
import inspect
import io
from typing import Optional
from pathlib import Path
from types import SimpleNamespace

import discord
import logging
from discord.ext import commands

logger = logging.getLogger(__name__)

from bot.tts.state import tts_state

from bot.router import get_router
try:
    from bot.voice.publisher import VoiceMessagePublisher  # type: ignore
except Exception:
    # Keep tests import-light if optional deps like 'utils' aren't on path
    VoiceMessagePublisher = None  # type: ignore
from bot.action import BotAction

class TTSCommands(commands.Cog):
    """Commands for controlling TTS functionality."""
    
    def __init__(self, bot):
        self.bot = bot
        self.router = bot.router
        self.prefix = '!'
        self.voice_publisher = VoiceMessagePublisher(logger=logger) if VoiceMessagePublisher else None
    
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
            # Support both unit-test monkeypatched function (AsyncMock) and Command object
            target = getattr(self, 'say', None)
            if target is None:
                return
            if isinstance(target, commands.Command):
                await target.callback(self, ctx, text=text)
            else:
                await target(ctx, text=text)
        else:
            # Only set the flag when no text is provided (for next response)
            logging.debug(f"🔊 Setting one-time TTS flag for user {user_id}",
                         extra={'subsys': 'tts_cmds', 'event': 'speak.one_time_flag', 
                                'user_id': user_id, 'channel_id': channel_id})
            tts_state.set_one_time_tts(ctx.author.id)
            await ctx.send("🗯️ The next response will be spoken.")
    
    @commands.command(name='say')
    async def say(
        self,
        ctx: commands.Context,
        *,
        text: Optional[str] = None,
        timeout_s: Optional[float] = None,
        cold: Optional[bool] = None,
        timeout_cold_s: Optional[float] = None,
        timeout_warm_s: Optional[float] = None,
    ):
        """Make the bot say exactly what you type without generating AI response."""
        async def maybe_call(func, *args, **kwargs):
            """Call a possibly-async function and await only if needed."""
            try:
                res = func(*args, **kwargs)
                if inspect.isawaitable(res):
                    return await res
                return res
            except TypeError as e:
                # Some MagicMocks raise when awaited; re-call without awaiting
                try:
                    return func(*args, **kwargs)
                except Exception:
                    raise e
        # 1) Attachment fast-path → delegate to router with one-time TTS
        has_attachments = False
        try:
            atts = getattr(ctx.message, "attachments", None)
            has_attachments = isinstance(atts, list) and len(atts) > 0
        except Exception:
            has_attachments = False
        if has_attachments:
            logging.debug(
                "📷 !say command with attachments, setting one-time TTS and dispatching to router.",
                extra={'subsys': 'tts_cmds', 'event': 'say.attachments'}
            )
            tts_state.set_one_time_tts(ctx.author.id)
            await self.router.dispatch_message(ctx.message)
            return

        # 2) Ensure TTS is available
        if not self.bot.tts_manager.is_available():
            await maybe_call(ctx.send, "❌ TTS is not available at the moment.")
            return

        # 3) Resolve text (fallback to user's previous message if empty)
        user_id = str(ctx.author.id)
        channel_id = str(ctx.channel.id)
        if text is None or text.strip() == "":
            logging.debug(
                "🔍 !say command with empty text, attempting to find previous message",
                extra={'subsys': 'tts_cmds', 'event': 'say.empty_text', 'user_id': user_id, 'channel_id': channel_id},
            )
            previous_messages = [
                msg async for msg in ctx.channel.history(limit=5)
                if msg.id != ctx.message.id and msg.author.id == ctx.author.id
            ]
            if previous_messages:
                text = previous_messages[0].content
                msg_id = str(previous_messages[0].id)
                logging.debug(
                    f"✅ Found previous message to use for TTS: '{text[:30]}...'",
                    extra={
                        'subsys': 'tts_cmds', 'event': 'say.fallback_found',
                        'user_id': user_id, 'channel_id': channel_id,
                        'msg_id': msg_id, 'content_length': len(text),
                    },
                )
            else:
                logging.warning(
                    "⚠️ No previous message found for !say with empty text",
                    extra={'subsys': 'tts_cmds', 'event': 'say.fallback_not_found', 'user_id': user_id, 'channel_id': channel_id},
                )
                await maybe_call(ctx.send, "❌ Please provide text to speak or send a message before using !say")
                return

        try:
            # 4) Permission check (guilds only)
            guild_id = str(ctx.guild.id) if ctx.guild else None
            channel_id = str(ctx.channel.id)
            user_id = str(ctx.author.id)

            can_send = True
            can_attach = True
            if ctx.guild:
                me = ctx.guild.me or ctx.guild.get_member(self.bot.user.id)
                if me is not None:
                    perms = ctx.channel.permissions_for(me)
                    can_send = bool(getattr(perms, 'send_messages', False))
                    can_attach = bool(getattr(perms, 'attach_files', False))
                logging.debug(
                    "🔐 Permission check",
                    extra={
                        'subsys': 'tts_cmds', 'event': 'say.permission_check',
                        'guild_id': guild_id, 'channel_id': channel_id, 'user_id': user_id,
                        'detail': {'can_send': can_send, 'can_attach': can_attach},
                    },
                )

            # 5) Synthesize audio using TTSManager.process with dynamic timeout meta
            meta: dict = {}
            try:
                if timeout_s is not None:
                    meta['tts_timeout_s'] = float(timeout_s)
            except Exception:
                pass
            if cold is not None:
                meta['tts_cold'] = bool(cold)
            try:
                if timeout_cold_s is not None:
                    meta['tts_timeout_cold_s'] = float(timeout_cold_s)
            except Exception:
                pass
            try:
                if timeout_warm_s is not None:
                    meta['tts_timeout_warm_s'] = float(timeout_warm_s)
            except Exception:
                pass

            audio_path = None
            audio_bytes = None
            mime_type = "audio/ogg"  # Default to OGG format
            try:
                action = BotAction(content=text, meta=meta)
                res_action = await self.bot.tts_manager.process(action)
                audio_path = res_action.audio_path
                if audio_path:
                    audio_path = str(audio_path)
                    # Determine MIME type from file extension
                    if audio_path.endswith('.ogg'):
                        mime_type = "audio/ogg"
                    elif audio_path.endswith('.wav'):
                        mime_type = "audio/wav"
            except Exception:
                # Fallback to legacy direct calls if process not available or failed
                try:
                    audio_path, mime_type = await self.bot.tts_manager.generate_tts(text, output_format="ogg")
                except Exception:
                    audio_bytes = await self.bot.tts_manager.synthesize(text)
                    mime_type = "audio/wav"

            # 6) Try native voice message first (guild only)
            guild_obj = getattr(ctx, 'guild', None)
            in_guild = isinstance(guild_obj, discord.Guild)
            if self.voice_publisher and audio_path and in_guild and can_send:
                try:
                    pub_res = await self.voice_publisher.publish(
                        message=ctx.message,
                        wav_path=audio_path,
                        include_transcript=False,
                        transcript_text=None,
                    )
                    if getattr(pub_res, 'ok', False):
                        logging.debug(
                            "✅ Native voice message published",
                            extra={'subsys': 'tts_cmds', 'event': 'say.native_voice_ok', 'guild_id': guild_id, 'channel_id': channel_id, 'user_id': user_id},
                        )
                        return
                    else:
                        logging.warning(
                            "⚠️ Native voice message publish reported not ok; falling back",
                            extra={'subsys': 'tts_cmds', 'event': 'say.native_voice_not_ok', 'guild_id': guild_id, 'channel_id': channel_id, 'user_id': user_id},
                        )
                except Exception as e:
                    logging.warning(
                        f"Native voice message publish failed: {e}",
                        exc_info=True,
                        extra={'subsys': 'tts_cmds', 'event': 'say.native_voice_error', 'guild_id': guild_id, 'channel_id': channel_id, 'user_id': user_id},
                    )

            # Helper to build a discord.File with correct filename and MIME handling
            def build_file_for_send() -> discord.File:
                nonlocal audio_bytes, audio_path, mime_type
                if audio_path:
                    # Determine filename based on MIME type  
                    path_obj = Path(audio_path)
                    ext = ".ogg" if mime_type == "audio/ogg" else ".wav"
                    filename = path_obj.name if path_obj.suffix == ext else f"{path_obj.stem}{ext}"
                    return discord.File(audio_path, filename=filename)
                else:
                    stream = io.BytesIO(audio_bytes or b"")
                    stream.seek(0)
                    filename = "tts_audio.ogg" if mime_type == "audio/ogg" else "tts_audio.wav"
                    return discord.File(stream, filename=filename)

            async def send_in_channel() -> bool:
                try:
                    await maybe_call(ctx.send, file=build_file_for_send())
                    logging.debug(
                        "✅ TTS response sent in channel",
                        extra={'subsys': 'tts_cmds', 'event': 'say.sent_channel', 'guild_id': guild_id, 'channel_id': channel_id, 'user_id': user_id},
                    )
                    return True
                except discord.Forbidden as e:
                    logging.warning(
                        f"Cannot send TTS in channel (permissions?): {e}",
                        extra={'subsys': 'tts_cmds', 'event': 'say.forbidden_channel', 'guild_id': guild_id, 'channel_id': channel_id, 'user_id': user_id},
                    )
                    return False
                except Exception as e:
                    logging.error(
                        f"Unexpected error sending in channel: {e}",
                        exc_info=True,
                        extra={'subsys': 'tts_cmds', 'event': 'say.error_channel', 'guild_id': guild_id, 'channel_id': channel_id, 'user_id': user_id},
                    )
                    return False

            async def send_in_dm() -> bool:
                try:
                    await maybe_call(ctx.author.send, file=build_file_for_send())
                    logging.debug(
                        "✅ TTS response sent via DM",
                        extra={'subsys': 'tts_cmds', 'event': 'say.sent_dm', 'guild_id': guild_id, 'channel_id': channel_id, 'user_id': user_id},
                    )
                    return True
                except discord.Forbidden as e:
                    logging.warning(
                        f"Cannot DM user (privacy settings?): {e}",
                        extra={'subsys': 'tts_cmds', 'event': 'say.forbidden_dm', 'guild_id': guild_id, 'channel_id': channel_id, 'user_id': user_id},
                    )
                    return False
                except Exception as e:
                    logging.error(
                        f"Unexpected error sending DM: {e}",
                        exc_info=True,
                        extra={'subsys': 'tts_cmds', 'event': 'say.error_dm', 'guild_id': guild_id, 'channel_id': channel_id, 'user_id': user_id},
                    )
                    return False

            sent = False
            if can_send and can_attach:
                sent = await send_in_channel()
                if not sent:
                    sent = await send_in_dm()
            else:
                sent = await send_in_dm()

            if not sent:
                # Last-resort notification; in tests, send_in_channel() succeeds so this won't fire
                msg = "❌ I couldn't deliver the audio. Please enable DMs or ask a moderator to grant me 'Attach Files' permission."
                try:
                    await maybe_call(ctx.send, msg)
                except Exception:
                    logging.error(
                        "Failed to notify user about undeliverable audio",
                        extra={'subsys': 'tts_cmds', 'event': 'say.notify_failed', 'guild_id': guild_id, 'channel_id': channel_id, 'user_id': user_id},
                    )

        except Exception as e:
            logging.error(f"Error in say command: {e}", exc_info=True)
            try:
                await maybe_call(ctx.send, f"❌ An error occurred while generating TTS: {str(e)}")
            except Exception:
                pass
            return

    
    
    # Note: The standalone tts-all command is removed to avoid duplication
    # The functionality is now handled by the @tts_group.command(name='all') subcommand

async def setup(bot):
    """Add the TTS commands to the bot."""
    if not bot.get_cog('TTSCommands'):
        await bot.add_cog(TTSCommands(bot))
    else:
        logger.warning("'TTSCommands' cog already loaded, skipping setup.")
