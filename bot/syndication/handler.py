"""
Syndication handler for Twitter/X content to VL flow integration.
"""

from typing import Dict, Any, Optional
from .extract import extract_text_and_images_from_syndication
import logging
import os

log = logging.getLogger(__name__)

# Guidelines injected into VL prompts to ensure consistent, compact, factual image descriptions
# that ignore non-content UI chrome and engagement metrics.
VL_PROMPT_GUIDELINES = (
    "describe only the actual picture. ignore ui chrome, cookie/login banners, overlays, "
    "timestamps, watermarks, likes/retweets/views and other page furniture. be factual and "
    "concise. include any visible text in quotes exactly as seen; skip illegible text. don't "
    "speculate—if unclear, say 'unclear'."
)

# Bounded concurrency for VL image processing [PA][RM]
# Configurable via env `VL_CONCURRENCY_LIMIT` (optional; default 4)
try:
    VL_CONCURRENCY_LIMIT: int = int(os.getenv("VL_CONCURRENCY_LIMIT", "4"))
except Exception:
    VL_CONCURRENCY_LIMIT = 4


async def handle_twitter_syndication_to_vl(
    tweet_json: Dict[str, Any],
    url: str,
    unified_vl_pipeline_func,
    prompt_guidelines: Optional[str] = None,
    timeout_s: Optional[float] = None,
    reply_style: str = "ack+thoughts",
) -> str:
    """
    Handle Twitter/X link to VL flow using syndication data and unified 1-hop pipeline.

    Args:
        tweet_json: Syndication JSON data from Twitter/X
        url: Original tweet URL for context
        unified_vl_pipeline_func: Function that handles unified VL→Text pipeline (no midstream sends)
        prompt_guidelines: Optional override for VL prompt guidelines; if None, uses default VL_PROMPT_GUIDELINES.
        timeout_s: Optional per-image timeout (seconds). If None, uses env `VL_IMAGE_TIMEOUT_S` or defaults to 12s.
        reply_style: Reply formatting style - "ack+thoughts", "summarize", or "verbatim+thoughts"

    Returns:
        Single final response from unified pipeline (no preview sends)
    """
    data = extract_text_and_images_from_syndication(tweet_json)

    text = data.get("text", "").strip()
    image_urls = data.get("image_urls", [])
    source = data.get("source", "unknown")
    had_card = bool(data.get("had_card", False))
    # Optional debug instrumentation
    try:
        debug_pick = os.getenv("SYND_DEBUG_MEDIA_PICK", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    except Exception:
        debug_pick = False
    if debug_pick:
        try:
            preview = ", ".join((image_urls[:2])) if image_urls else "(none)"
            log.info(
                "SYND_MEDIA_PICK | source=%s count=%d preview=%s card_present=%s",
                source,
                len(image_urls),
                preview[:360],
                str(had_card).lower(),
            )
            if had_card and source != "card":
                log.info("SYND_MEDIA_PICK | ignored_card_due_to_native=true")
        except Exception:
            pass

    if not image_urls:
        # No images - return text-only or fallback
        log.info("No images found via syndication; returning text only")
        return text if text else "No content or images available."

    # Use unified VL→Text pipeline (no midstream sends, enforces 1 out)
    try:
        # Download images to temp files
        import tempfile
        import aiohttp

        temp_paths = []

        max_images = int(os.getenv("VL_MAX_IMAGES", "4"))
        limited_urls = image_urls[:max_images]

        async with aiohttp.ClientSession() as session:
            for i, image_url in enumerate(limited_urls):
                try:
                    async with session.get(image_url) as response:
                        if response.status == 200:
                            # Create temp file with appropriate extension
                            suffix = ".jpg"
                            if "png" in image_url.lower():
                                suffix = ".png"
                            elif "webp" in image_url.lower():
                                suffix = ".webp"

                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=suffix
                            ) as tmp_file:
                                data = await response.read()
                                tmp_file.write(data)
                                temp_paths.append(tmp_file.name)
                                log.debug(
                                    f"Downloaded image {i + 1} to {tmp_file.name}"
                                )
                        else:
                            log.warning(
                                f"Failed to download image {i + 1}: HTTP {response.status}"
                            )
                except Exception as e:
                    log.warning(f"Error downloading image {i + 1}: {e}")

        if not temp_paths:
            return "Got the tweet but couldn't download any images."

        # Call unified pipeline with tweet text as caption
        try:
            result = await unified_vl_pipeline_func(temp_paths, text, "Tweet analysis")

            # Extract content from BotAction if that's what's returned
            if hasattr(result, "content"):
                return result.content
            else:
                return str(result)

        finally:
            # Clean up temp files
            for path in temp_paths:
                try:
                    os.unlink(path)
                except Exception:
                    pass

    except Exception as e:
        log.error(f"Unified VL pipeline failed for syndication: {e}")
        return f"Got the tweet but image analysis failed: {str(e)[:100]}"
