"""Tool: generate an image from a text prompt.
[CA][REH][SFT].

Image generation only triggered on narrow command patterns ("img: ...",
"!img ..."), so ordinary requests like "draw yourself" reached the text model,
which then said it can't make images. Exposing generation as a tool lets the
model recognise any phrasing and write a proper visual prompt itself.

The job runs through the existing vision orchestrator (policy, budgets,
per-user limits, progress card, upload); this tool only starts it.
"""

from __future__ import annotations

from typing import Any

from bot.utils.logging import get_logger

from ..types import ToolContext, ToolResult, ToolSpec

logger = get_logger(__name__)

MIN_PROMPT_CHARS = 3
MAX_PROMPT_CHARS = 1000
SIZES = {"square": (1024, 1024), "portrait": (832, 1216), "landscape": (1216, 832)}

DESCRIPTION = (
    "Create a picture and post it in the chat. Use whenever someone asks you to draw, generate, make, "
    "create or show an image, picture, pic, drawing, meme or artwork -- you CAN make images with this. "
    "Write the prompt as a detailed visual description in English (subject, setting, style, lighting). "
    "For a picture of yourself, describe your own appearance from your persona."
)

PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string", "description": "Detailed visual description of the image to create."},
        "shape": {"type": "string", "enum": sorted(SIZES), "default": "square"},
    },
    "required": ["prompt"],
}


async def generate_image(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    """Start a text-to-image job; the image is posted when ready. Never raises."""
    prompt = " ".join(str(arguments.get("prompt") or "").split())[:MAX_PROMPT_CHARS]
    if len(prompt) < MIN_PROMPT_CHARS:
        return ToolResult.failure("prompt is required")
    router = getattr(ctx.bot, "router", None)
    message = ctx.message
    if router is None or message is None or not hasattr(router, "start_image_generation"):
        return ToolResult.failure("image generation isn't available here")
    width, height = SIZES.get(str(arguments.get("shape") or "square"), SIZES["square"])
    started, detail = await router.start_image_generation(prompt, message, width=width, height=height)
    logger.info("tool.generate_image started=%s detail=%s prompt=%s", started, detail, prompt[:80])
    if not started:
        return ToolResult.failure(f"couldn't start the image: {detail}")
    return ToolResult.success(
        "Image generation started; the picture will be posted in this channel by itself in about a minute. Reply with one short line saying it's coming. Don't describe it as finished and don't paste the prompt."
    )


SPEC = ToolSpec(
    name="generate_image",
    description=DESCRIPTION,
    parameters=PARAMETERS,
    handler=generate_image,
    timeout_s=30.0,
)
