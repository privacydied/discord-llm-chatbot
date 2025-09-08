"""
!img prefix command implementation

Provides traditional prefix command interface for image generation that works
with or without bot mention in guild channels. Delegates to the existing
vision orchestrator system for actual processing.

Enhancements:
- Allow attachment-fed prompts when no inline text is provided (text/* or .txt/.md/.rtf/.json, ≤ 64 KB)
- Provide a single, stylish help embed card (no job submission on help path)
- Add a debug log indicating the prompt source (inline|attachment)
"""

import os
import re
import tempfile
from pathlib import Path
import discord
from discord.ext import commands
from typing import Optional

from bot.util.logging import get_logger
from bot.config import load_config
from bot.utils.file_utils import download_file

logger = get_logger(__name__)


class ImgCommands(commands.Cog):
    """Traditional !img prefix command cog"""
    
    def __init__(self, bot):
        self.bot = bot
        self.config = load_config()
        self.logger = logger
        
        self.logger.info("!img prefix commands cog initialized")
    
    @commands.command(name="img", aliases=["image"], help="Generate images from text prompts")
    async def img_command(self, ctx, *, prompt: Optional[str] = None):
        """
        Handle !img prefix command - delegates to vision generation system
        
        Usage:
        !img a kitten playing with yarn
        @Bot !img a sunset over mountains
        """
        # Help trigger (explicit)
        if prompt and prompt.strip().lower() == "help":
            await ctx.send(embed=self._build_img_help_embed())
            return

        final_prompt: Optional[str] = None
        inline = (prompt or "").strip()
        if inline:
            # Inline prompt takes precedence
            final_prompt = inline
            try:
                self.logger.debug(
                    f"IMG.prompt_source=inline file=none size={len(inline)} msg_id={ctx.message.id}"
                )
            except Exception:
                pass
        else:
            # No inline prompt: try to use first eligible attachment as prompt
            attachments = getattr(ctx.message, 'attachments', []) or []
            eligible_exts = {".txt", ".md", ".rtf", ".json"}
            max_bytes = 0
            try:
                max_bytes = int(os.getenv("IMG_ATTACHMENT_MAX_BYTES", "65536"))
            except Exception:
                max_bytes = 65536
            chosen = None
            for att in attachments:
                try:
                    ct = (att.content_type or "").lower()
                except Exception:
                    ct = ""
                try:
                    ext = Path(att.filename or "").suffix.lower()
                except Exception:
                    ext = ""
                size_ok = True
                try:
                    size_ok = int(getattr(att, 'size', 0) or 0) <= max_bytes
                except Exception:
                    size_ok = True
                if (ct.startswith("text/") or ext in eligible_exts) and size_ok:
                    chosen = att
                    break

            if chosen is not None:
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False) as tf:
                        tmp_path = Path(tf.name)
                    ok = await download_file(chosen.url, tmp_path)
                    if not ok:
                        await ctx.send(embed=self._build_img_help_embed())
                        return
                    # Guard read size and decode
                    raw = b""
                    with open(tmp_path, "rb") as f:
                        raw = f.read(max_bytes + 1)
                    try:
                        text = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        text = raw.decode("latin-1", errors="ignore")
                    # Sanitize: strip NULs, collapse whitespace
                    text = text.replace("\x00", "")
                    text = re.sub(r"\s+", " ", text).strip()
                    if text:
                        final_prompt = text
                        try:
                            self.logger.debug(
                                f"IMG.prompt_source=attachment file={chosen.filename} size={getattr(chosen, 'size', 0)} msg_id={ctx.message.id}"
                            )
                        except Exception:
                            pass
                    else:
                        await ctx.send(embed=self._build_img_help_embed())
                        return
                except Exception as e:
                    self.logger.warning(f"Attachment prompt extraction failed: {e}")
                    await ctx.send(embed=self._build_img_help_embed())
                    return
                finally:
                    try:
                        if tmp_path and tmp_path.exists():
                            tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            else:
                # No eligible attachment
                await ctx.send(embed=self._build_img_help_embed())
                return

        # Log command detection
        self.logger.info(f"Found command 'IMG', delegating to vision system (msg_id: {ctx.message.id})")
        
        # Check if Vision is enabled
        if not self.config.get("VISION_ENABLED", False):
            await ctx.send("🚫 Vision generation is currently disabled.", ephemeral=True)
            return
        
        # Delegate to router's vision generation handler
        # Import here to avoid circular imports
        from bot.vision.intent_router import VisionIntentResult, VisionIntentParams
        from bot.vision.types import VisionTask
        
        # Create a mock intent result that matches what the vision system expects
        class MockIntentParams:
            def __init__(self, prompt: str):
                self.task = VisionTask.TEXT_TO_IMAGE.value
                self.prompt = prompt
                self.negative_prompt = ""
                self.width = 1024
                self.height = 1024
                self.steps = 30
                self.guidance_scale = 7.0
                self.seed = None
                self.preferred_provider = None
        
        class MockIntentResult:
            def __init__(self, prompt: str):
                self.extracted_params = MockIntentParams(prompt)
        
        # Get router from bot and delegate
        if hasattr(self.bot, 'router') and self.bot.router:
            try:
                mock_intent = MockIntentResult(final_prompt)
                action = await self.bot.router._handle_vision_generation(
                    mock_intent, 
                    ctx.message, 
                    ""  # context_str
                )
                
                # The vision handler manages its own response, so we don't need to do anything more
                self.logger.info(f"Successfully delegated !img to vision system (msg_id: {ctx.message.id})")
                
            except Exception as e:
                self.logger.error(f"Failed to delegate !img to vision system: {e}", exc_info=True)
                await ctx.send("❌ Failed to process image generation request. Please try again.")
        else:
            await ctx.send("🚫 Vision system is not available right now. Please try again later.")

    def _build_img_help_embed(self) -> discord.Embed:
        """Build the single, stylish help embed card for !img usage."""
        # Brand purple (amethyst) or adjust to your brand color constant
        BRAND_PURPLE = 0x9B59B6
        embed = discord.Embed(
            title="🎨 Image Generation Help",
            description="Generate images from text descriptions.",
            color=BRAND_PURPLE,
        )
        embed.add_field(name="Usage", value="!img <description>", inline=False)
        embed.add_field(name="Example", value="!img a kitten playing with yarn", inline=False)
        embed.add_field(
            name="Use a .txt file:",
            value=(
                "Attach message.txt with your prompt and send !img (no text needed).\n"
                "Supported: .txt, .md, .rtf, .json (≤ 64 KB)"
            ),
            inline=False,
        )
        embed.add_field(
            name="Tips:",
            value="Keep it clear and SFW. Style words help (e.g., ‘cinematic, soft lighting’).",
            inline=False,
        )
        embed.set_footer(text="Works in DMs and guilds. One image per request.")
        return embed


async def setup(bot):
    """Setup function for Discord cog loading"""
    await bot.add_cog(ImgCommands(bot))
