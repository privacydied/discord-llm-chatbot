"""Tool: check whether a post's images are AI-generated (Sightengine). [CA][SFT]

"@bot is this AI?" on a tweet / instagram link (or a reply to one, or to an
image) -> pulls the media through the bot's own fetchers and scores each image.
The model gets per-image probabilities plus a summary and must present them as
a probability, never as proof.
"""

from __future__ import annotations

import asyncio
from typing import Any

import discord

from bot import ai_detect
from bot.utils.logging import get_logger

from ..types import ToolContext, ToolResult, ToolSpec

logger = get_logger(__name__)

REFERENCE_TIMEOUT_S = 8.0

DESCRIPTION = (
    "Check whether the images in a post are AI-generated, with a dedicated AI-image detector. Always use "
    "this (not view_image) when someone asks if a picture, pic, tweet, instagram post or image is AI, fake, "
    "generated, real, edited or photoshopped -- eyeballing it is not a check. Works on X/Twitter and "
    "Instagram links (including vxtwitter/fixupx/oginstagram) and on images attached to the message "
    "or to the message being replied to. Leave url empty to use the link or image in this message or "
    "the one it replies to."
)

PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"url": {"type": "string", "description": "Tweet or Instagram link to check. Optional."}},
    "required": [],
}

GUIDANCE = (
    "Report this as a probability from an AI-image detector, not proof: say what it found per image in "
    "plain words (e.g. 'detector says ~92% likely AI'), give your read in one line, and mention once that "
    "detectors can be wrong, especially on screenshots, heavy compression or edited photos. Video is "
    "judged from a single frame. Don't invent details the scores don't give."
)


async def _replied_to(message: Any) -> Any | None:
    ref = getattr(message, "reference", None)
    if ref is None:
        return None
    resolved = getattr(ref, "resolved", None)
    if isinstance(resolved, discord.Message):
        return resolved
    channel = getattr(message, "channel", None)
    message_id = getattr(ref, "message_id", None)
    if channel is None or message_id is None or not hasattr(channel, "fetch_message"):
        return None
    try:
        return await asyncio.wait_for(channel.fetch_message(message_id), timeout=REFERENCE_TIMEOUT_S)
    except (TimeoutError, discord.HTTPException) as exc:
        logger.debug(f"tool.check_ai.reference_unavailable | {exc}")
        return None


def _summary(scores: list[ai_detect.Score], problems: list[str]) -> str:
    lines = []
    for score in scores:
        if score.ai_probability is None:
            lines.append(f"- {score.label}: not scored ({score.error})")
        else:
            pct = round(score.ai_probability * 100)
            lines.append(f"- {score.label}: {pct}% AI probability -> {ai_detect.verdict(score.ai_probability)}")
    lines.extend(f"- note: {p}" for p in problems)
    return "AI-image detector results (Sightengine):\n" + "\n".join(lines) + "\n\n" + GUIDANCE


async def check_ai_generated(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    """Collect the post's images and score them. Never raises."""
    creds = ai_detect.credentials(ctx.config)
    if creds is None:
        return ToolResult.failure("the AI-image detector isn't set up yet (SIGHTENGINE_API_USER / SIGHTENGINE_API_SECRET missing)")
    message = ctx.message
    replied = await _replied_to(message) if message is not None else None
    urls = ai_detect.find_post_urls(str(arguments.get("url") or ""))
    for msg in (message, replied):
        if not urls and msg is not None:
            urls = ai_detect.find_post_urls(getattr(msg, "content", ""))
    messages = [m for m in (message, replied) if m is not None]
    sources, problems = await ai_detect.collect_media(urls, messages)
    if not sources:
        why = "; ".join(problems) or "no tweet/instagram link or image found in this message or the one it replies to"
        return ToolResult.failure(f"nothing to check: {why}")
    scores = await ai_detect.score_all(sources, creds)
    logger.info(
        "tool.check_ai_generated urls=%s images=%d scores=%s",
        urls[:2],
        len(sources),
        [None if s.ai_probability is None else round(s.ai_probability, 3) for s in scores],
    )
    if all(s.ai_probability is None for s in scores):
        return ToolResult.failure("the detector couldn't score any image: " + "; ".join(sorted({s.error or "" for s in scores})))
    return ToolResult.success(_summary(scores, problems))


SPEC = ToolSpec(
    name="check_ai_generated",
    description=DESCRIPTION,
    parameters=PARAMETERS,
    handler=check_ai_generated,
    timeout_s=75.0,
)
