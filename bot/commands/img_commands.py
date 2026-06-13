"""!img prefix command implementation.

Provides traditional prefix command interface for image generation that works
with or without bot mention in guild channels. Delegates to the existing
vision orchestrator system for actual processing.

Enhancements:
- Allow attachment-fed prompts when no inline text is provided (text/* or .txt/.md/.rtf/.json/.yaml/.yml, ≤ 256 KB)
- Provide a single, stylish help embed card (no job submission on help path)
- Add a debug log indicating the prompt source (inline|attachment)
"""

import contextlib
import json
import os
import re
from pathlib import Path
from typing import Any

import discord
from discord.ext import commands

from bot.config import load_config
from bot.utils.logging import get_logger

logger = get_logger(__name__)

# === Attachment prompt feature toggles (env-driven) ===
# Defaults: enabled, max 256 KiB, preferred extensions order informational only
IMG_ATTACHMENT_MAX_BYTES = int(os.getenv("IMG_ATTACHMENT_MAX_BYTES", "262144") or "262144")
IMG_ATTACHMENT_ENABLE = os.getenv("IMG_ATTACHMENT_ENABLE", "true").strip().lower() not in {"0", "false", "no", "off"}
IMG_ATTACHMENT_EXTS = [".txt", ".md", ".json", ".rtf", ".yaml", ".yml"]


class ImgCommands(commands.Cog):
    """Traditional !img prefix command cog."""

    def __init__(self, bot) -> None:
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
        return bool("json" in ctype or "yaml" in ctype)

    async def _read_attachment_text(self, att: discord.Attachment, limit_bytes: int) -> str | None:
        """Read and decode attachment text using Discord's API with size checks and sanitization."""
        try:
            # Read the full attachment, enforce limit after
            data = await att.read()
            if not isinstance(data, (bytes, bytearray)):
                return None
            if len(data) > limit_bytes:
                return None

            # Try multiple encodings to handle different file types
            text = None
            for encoding in ["utf-8", "utf-16", "latin-1"]:
                try:
                    text = data.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if not text:
                return None

            # Clean up the text
            text = text.replace("\x00", "")
            text = re.sub(r"\s+", " ", text).strip()
            return text or None
        except Exception as e:
            self.logger.debug(f"Failed to read attachment {att.filename}: {e}")
            return None

    def _parse_prompt_blob(self, blob: str) -> tuple[str | None, dict[str, Any]]:
        """Parse a blob into (prompt, params). Supports optional fenced JSON with keys.
        Keys: prompt, negative_prompt, width, height, steps, guidance_scale, seed, model.
        """
        params: dict[str, Any] = {}
        if not blob:
            return None, params

        s = blob.strip()
        # Strip triple backtick code fences if present
        # e.g., ```json { ... } ``` or ``` { ... } ```
        fence_match = re.match(r"^```[a-zA-Z0-9_-]*\s*(.*?)\s*```$", s, flags=re.DOTALL)
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
                    params_prompt = p.strip() if isinstance(p, str) else None
                    # Optional params
                    if "negative_prompt" in obj and isinstance(obj.get("negative_prompt"), str):
                        params["negative_prompt"] = obj.get("negative_prompt").strip()
                    for k in ("width", "height", "steps", "seed"):
                        if k in obj:
                            with contextlib.suppress(Exception):
                                params[k] = int(obj[k])
                    if "guidance_scale" in obj:
                        try:
                            params["guidance_scale"] = float(obj["guidance_scale"])
                        except (ValueError, TypeError) as e:
                            self.logger.debug(f"Failed to parse guidance_scale: {e}")
                    if "model" in obj and isinstance(obj.get("model"), str):
                        params["model"] = obj.get("model").strip()

                    # Enforce max length for prompt
                    if params_prompt:
                        return params_prompt[:2000], params
                    # If no prompt key, fall back to plain text using the original blob
                    s_plain = s.strip()
                    return (s_plain[:2000] if s_plain else None), params
            except json.JSONDecodeError as e:
                # Fall through to treat as plain text
                self.logger.debug(f"JSON decode failed, treating as plain text: {e}")

        # Plain text prompt
        s = s.strip()
        return (s[:2000] if s else None), params

    @commands.command(name="img", aliases=["image"], help="Generate images from text prompts")
    async def img_command(self, ctx: commands.Context, *, prompt: str | None = None) -> None:
        """Handle !img prefix command - delegates to vision generation system.

        Usage:
        !img a kitten playing with yarn
        @Bot !img a sunset over mountains
        """
        # Help trigger (explicit)
        if prompt and prompt.strip().lower() == "help":
            await ctx.send(embed=self._build_img_help_embed())
            return

        final_prompt: str | None = None
        parsed_params: dict[str, Any] = {}
        inline = (prompt or "").strip()

        self.logger.info(f"IMG: Raw prompt param: '{prompt}' (type: {type(prompt)}), stripped inline: '{inline}' (len: {len(inline)})")

        # Only use inline if it's actually meaningful content (not just whitespace or empty)
        if inline and len(inline) > 0:
            # Inline prompt takes precedence
            final_prompt = inline
            try:
                self.logger.info(f"IMG.prompt_source=inline len={len(inline)} content='{inline[:50]}...' msg_id={ctx.message.id}")
            except Exception as e:
                self.logger.debug(f"Failed to log inline prompt: {e}")
        else:
            # No inline prompt: collect attachments from current and referenced message
            attachments = list(getattr(ctx.message, "attachments", []) or [])
            current_count = len(attachments)

            self.logger.info(f"IMG: Processing attachments - found {current_count} attachments, enable={IMG_ATTACHMENT_ENABLE}")

            # Respect feature flag: if disabled and no inline, show help immediately
            if not IMG_ATTACHMENT_ENABLE:
                self.logger.info("IMG: Attachment processing disabled, showing help")
                await ctx.send(embed=self._build_img_help_embed())
                return
            # Include attachments from replied-to message if present
            try:
                ref = getattr(ctx.message, "reference", None)
                ref_msg = None
                if ref:
                    if getattr(ref, "resolved", None) and isinstance(ref.resolved, discord.Message):
                        ref_msg = ref.resolved
                    elif getattr(ref, "message_id", None):
                        ref_msg = await ctx.channel.fetch_message(ref.message_id)
                if ref_msg and getattr(ref_msg, "attachments", None):
                    attachments.extend(ref_msg.attachments)
            except Exception as e:
                self.logger.debug(f"IMG: Error processing reply attachments: {e}")
            # Log attachments count breadcrumb
            try:
                reply_has = bool(ref_msg and getattr(ref_msg, "attachments", None))
                total = len(attachments)
                self.logger.info(f"IMG.attachments current={current_count} reply_has={reply_has} total={total} msg_id={ctx.message.id}")
                for i, att in enumerate(attachments):
                    self.logger.info(f"IMG: Attachment {i}: {att.filename} ({att.size} bytes, type: {att.content_type})")
            except Exception as e:
                self.logger.debug(f"Failed to log attachments: {e}")

            max_bytes = IMG_ATTACHMENT_MAX_BYTES

            # Try all attachments regardless of extension - simple approach
            candidates = []
            for att in attachments:
                try:
                    size_ok = int(getattr(att, "size", 0) or 0) <= max_bytes
                except (ValueError, TypeError) as e:
                    self.logger.debug(f"Failed to parse attachment size: {e}")
                    size_ok = True
                if size_ok:
                    candidates.append(att)

            found = False
            for cand in candidates:
                try:
                    self.logger.info(f"IMG: Trying to read attachment: {cand.filename}")
                    blob = await self._read_attachment_text(cand, max_bytes)
                    self.logger.info(f"IMG: Read blob from {cand.filename}: '{blob[:100] if blob else None}...'")
                    if not blob:
                        self.logger.info(f"IMG: No text content from {cand.filename}")
                        continue
                    p, params = self._parse_prompt_blob(blob)
                    if not p:
                        # Fall back to using plain text if JSON lacked 'prompt'
                        p = blob.strip()[:2000]
                        self.logger.info(f"IMG: Using fallback plain text: '{p[:50]}...'")
                    if p:
                        final_prompt = p
                        parsed_params = params or {}
                        try:
                            self.logger.info(f"IMG.prompt_source=attachment file={cand.filename} size={getattr(cand, 'size', 0)} prompt='{p[:50]}...' msg_id={ctx.message.id}")
                        except Exception as e:
                            self.logger.debug(f"Failed to log attachment prompt source: {e}")
                        found = True
                        break
                except Exception as e:
                    # Continue to next candidate on read/parse failure
                    self.logger.exception(f"IMG: Exception reading {cand.filename}: {e}")
                    continue

            if not found:
                # No usable attachment-derived prompt
                self.logger.info("IMG: No usable attachment found, showing help")
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
        from bot.vision.types import VisionTask

        # Create a mock intent result that matches what the vision system expects
        class MockIntentParams:
            def __init__(self, prompt: str) -> None:
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
            def __init__(self, prompt: str) -> None:
                self.extracted_params = MockIntentParams(prompt)

        # Get router from bot and delegate
        if hasattr(self.bot, "router") and self.bot.router:
            try:
                mock_intent = MockIntentResult(final_prompt)
                # Gently apply parsed params (only keys present)
                if parsed_params:
                    ep = mock_intent.extracted_params
                    for k in (
                        "negative_prompt",
                        "width",
                        "height",
                        "steps",
                        "guidance_scale",
                        "seed",
                    ):
                        if k in parsed_params and parsed_params[k] is not None:
                            with contextlib.suppress(Exception):
                                setattr(ep, k, parsed_params[k])
                    if "model" in parsed_params and isinstance(parsed_params["model"], str):
                        with contextlib.suppress(Exception):
                            ep.model = parsed_params["model"]
                await self.bot.router._handle_vision_generation(
                    mock_intent,
                    ctx.message,
                    "",  # context_str
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
        # Discord brand primary color
        BRAND_PRIMARY = 0x5865F2
        embed = discord.Embed(
            title="🎨 Image Generation Help",
            description=("Generate images from text prompts.\nSend text inline, or attach a small .txt/.json file and send !img."),
            color=BRAND_PRIMARY,
        )
        embed.add_field(name="Usage", value="!img <description>", inline=False)
        embed.add_field(
            name="Examples",
            value="• !img a kitten playing with yarn\n• !img (attach prompt.txt or prompt.json)",
            inline=False,
        )
        embed.add_field(
            name="Attachments:",
            value=('Attach a small .txt or .json (≤256 KB) and send !img with no text. JSON may include options: {"prompt":"a foggy forest","width":1024,"height":1024}.'),
            inline=False,
        )
        embed.set_footer(text="Works in DMs and guild channels. You can also reply to a message with a file.")
        return embed


async def setup(bot) -> None:
    """Setup function for Discord cog loading."""
    await bot.add_cog(ImgCommands(bot))
