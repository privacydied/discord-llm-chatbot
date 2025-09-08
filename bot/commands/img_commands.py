"""
!img prefix command implementation

Provides traditional prefix command interface for image generation that works
with or without bot mention in guild channels. Delegates to the existing
vision orchestrator system for actual processing.

Enhancements:
- Allow attachment-fed prompts when no inline text is provided (text/* or .txt/.md/.rtf/.json/.yaml/.yml, ≤ 256 KB)
- Provide a single, stylish help embed card (no job submission on help path)
- Add a debug log indicating the prompt source (inline|attachment)
"""

import os
import re
import json
from pathlib import Path
import discord
from discord.ext import commands
from typing import Optional, Tuple, Dict, Any

from bot.util.logging import get_logger
from bot.config import load_config
 

logger = get_logger(__name__)


class ImgCommands(commands.Cog):
    """Traditional !img prefix command cog"""
    
    def __init__(self, bot):
        self.bot = bot
        self.config = load_config()
        self.logger = logger
        
        self.logger.info("!img prefix commands cog initialized")

    # === Helpers for attachment-fed prompts ===
    def _eligible_text_attachment(self, att: discord.Attachment) -> bool:
        """Return True if the attachment is likely text based on extension or content_type.
        Accepts .txt/.md/.rtf/.json/.yaml/.yml, and tolerates missing or octet-stream content types.
        """
        try:
            filename = (att.filename or "").lower()
            ext = Path(filename).suffix
            ctype = (att.content_type or "").lower()
        except Exception:
            filename, ext, ctype = "", "", ""

        allowed_exts = {".txt", ".md", ".rtf", ".json", ".yaml", ".yml"}
        if ext in allowed_exts:
            return True

        # Discord often leaves content_type None or application/octet-stream for .txt
        if not ctype or ctype == "application/octet-stream":
            return True

        if ctype.startswith("text/"):
            return True
        if "json" in ctype or "yaml" in ctype:
            return True
        return False

    async def _read_attachment_text(self, att: discord.Attachment, limit_bytes: int) -> Optional[str]:
        """Read and decode attachment text using Discord's API with size checks and sanitization."""
        # Enforce size cap before reading
        try:
            size = int(getattr(att, 'size', 0) or 0)
            if size and size > limit_bytes:
                return None
        except Exception:
            pass

        try:
            # Read up to limit+1 to detect oversize
            data = await att.read(limit_bytes + 1)
            if not isinstance(data, (bytes, bytearray)):
                return None
            if len(data) > limit_bytes:
                return None
            try:
                text = data.decode("utf-8")
            except Exception:
                text = data.decode("latin-1", errors="ignore")
            text = text.replace("\x00", "")
            text = re.sub(r"\s+", " ", text).strip()
            return text or None
        except Exception:
            return None

    def _parse_prompt_blob(self, blob: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Parse a blob into (prompt, params). Supports optional fenced JSON with keys.
        Keys: prompt, negative_prompt, width, height, steps, guidance_scale, seed, model
        """
        params: Dict[str, Any] = {}
        if not blob:
            return None, params

        s = blob.strip()
        # Strip triple backtick code fences if present
        # e.g., ```json { ... } ``` or ``` { ... } ```
        fence_match = re.match(r"^```[a-zA-Z0-9_-]*\s*(.*?)\s*```$", s, flags=re.S)
        if fence_match:
            s = fence_match.group(1).strip()

        # If JSON-like, extract known fields
        candidate = s.lstrip()
        if candidate.startswith("{"):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    # Prompt
                    p = obj.get("prompt")
                    if isinstance(p, str):
                        params_prompt = p.strip()
                    else:
                        params_prompt = None
                    # Optional params
                    if "negative_prompt" in obj and isinstance(obj.get("negative_prompt"), str):
                        params["negative_prompt"] = obj.get("negative_prompt").strip()
                    for k in ("width", "height", "steps", "seed"):
                        if k in obj:
                            try:
                                params[k] = int(obj[k])
                            except Exception:
                                pass
                    if "guidance_scale" in obj:
                        try:
                            params["guidance_scale"] = float(obj["guidance_scale"])
                        except Exception:
                            pass
                    if "model" in obj and isinstance(obj.get("model"), str):
                        params["model"] = obj.get("model").strip()

                    # Enforce max length for prompt
                    if params_prompt:
                        return params_prompt[:2000], params
                    # If no prompt key, fall back to plain text using the original blob
                    s_plain = s.strip()
                    return (s_plain[:2000] if s_plain else None), params
            except Exception:
                # Fall through to treat as plain text
                pass

        # Plain text prompt
        s = s.strip()
        return (s[:2000] if s else None), params
    
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
        parsed_params: Dict[str, Any] = {}
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
            # No inline prompt: collect attachments from current and referenced message
            attachments = list(getattr(ctx.message, 'attachments', []) or [])
            # Include attachments from replied-to message if present
            try:
                ref = getattr(ctx.message, 'reference', None)
                ref_msg = None
                if ref:
                    if getattr(ref, 'resolved', None) and isinstance(ref.resolved, discord.Message):
                        ref_msg = ref.resolved
                    elif getattr(ref, 'message_id', None):
                        ref_msg = await ctx.channel.fetch_message(ref.message_id)
                if ref_msg and getattr(ref_msg, 'attachments', None):
                    attachments.extend(ref_msg.attachments)
            except Exception:
                pass
            # Log attachments count breadcrumb
            try:
                self.logger.debug(f"IMG.attachments count={len(attachments)} msg_id={ctx.message.id}")
            except Exception:
                pass

            try:
                max_bytes = int(os.getenv("IMG_ATTACHMENT_MAX_BYTES", "262144"))
            except Exception:
                max_bytes = 262144

            # Prefer text-like extensions but do not require them
            allowed_exts = (".txt", ".md", ".rtf", ".json", ".yaml", ".yml")
            preferred: list[discord.Attachment] = []
            others: list[discord.Attachment] = []
            for att in attachments:
                name = (getattr(att, 'filename', '') or '').lower()
                if not name:
                    continue
                try:
                    size_ok = int(getattr(att, 'size', 0) or 0) <= max_bytes
                except Exception:
                    size_ok = True
                if not size_ok:
                    continue
                if name.endswith(allowed_exts):
                    preferred.append(att)
                else:
                    others.append(att)

            candidates = preferred + others

            found = False
            for cand in candidates:
                try:
                    blob = await self._read_attachment_text(cand, max_bytes)
                    if not blob:
                        continue
                    p, params = self._parse_prompt_blob(blob)
                    if not p:
                        # Fall back to using plain text if JSON lacked 'prompt'
                        p = blob.strip()[:2000]
                    if p:
                        final_prompt = p
                        parsed_params = params or {}
                        try:
                            self.logger.debug(
                                f"IMG.prompt_source=attachment file={cand.filename} size={getattr(cand, 'size', 0)} msg_id={ctx.message.id}"
                            )
                        except Exception:
                            pass
                        found = True
                        break
                except Exception:
                    # Continue to next candidate on read/parse failure
                    continue

            if not found:
                # No usable attachment-derived prompt
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
                self.model = None
        
        class MockIntentResult:
            def __init__(self, prompt: str):
                self.extracted_params = MockIntentParams(prompt)
        
        # Get router from bot and delegate
        if hasattr(self.bot, 'router') and self.bot.router:
            try:
                mock_intent = MockIntentResult(final_prompt)
                # Gently apply parsed params (only keys present)
                if parsed_params:
                    ep = mock_intent.extracted_params
                    for k in ("negative_prompt", "width", "height", "steps", "guidance_scale", "seed"):
                        if k in parsed_params and parsed_params[k] is not None:
                            try:
                                setattr(ep, k, parsed_params[k])
                            except Exception:
                                pass
                    if "model" in parsed_params and isinstance(parsed_params["model"], str):
                        try:
                            ep.model = parsed_params["model"]
                        except Exception:
                            pass
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
        embed.add_field(name="Examples", value="!img a kitten playing with yarn\n(or) attach message.txt with your prompt and send !img", inline=False)
        embed.add_field(
            name="Use a file:",
            value=(
                "You can also attach a small .txt or .json file and send !img with no text.\n"
                "Max attachment size: 256 KB. JSON example: { \"prompt\": \"a red fox in snow\", \"width\": 1024, \"height\": 1024 }"
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
