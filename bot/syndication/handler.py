"""
Syndication handler for Twitter/X content to VL flow integration.
"""
from typing import Dict, Any, List, Optional
from .extract import extract_text_and_images_from_syndication
import logging
import asyncio
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
    vl_handler_func,
    prompt_guidelines: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> str:
    """
    Handle Twitter/X link to VL flow using syndication data and full-resolution images.
    
    Args:
        tweet_json: Syndication JSON data from Twitter/X
        url: Original tweet URL for context
        vl_handler_func: Function to handle individual image VL analysis (e.g., _vl_describe_image_from_url)
        prompt_guidelines: Optional override for VL prompt guidelines; if None, uses default VL_PROMPT_GUIDELINES.
        timeout_s: Optional per-image timeout (seconds). If None, uses env `VL_IMAGE_TIMEOUT_S` or defaults to 12s.
        
    Returns:
        Formatted string with tweet text and all image descriptions
    """
    data = extract_text_and_images_from_syndication(tweet_json)
    
    text = data["text"].strip()
    image_urls = data["image_urls"]
    
    if not image_urls:
        # Keep existing behavior: return text-only or handle as needed
        log.info("No images found via syndication; returning text only")
        return text if text else "No content or images available."
    
    # Process all high-res images through VL
    descriptions: List[str] = []
    metrics = None

    # Metrics are optional and non-breaking
    try:
        from bot.metrics import METRICS  # type: ignore

        metrics = METRICS
    except Exception:
        metrics = None
    
    # Concurrency with order preservation
    total = len(image_urls)
    # Determine effective timeout
    if timeout_s is None:
        try:
            eff_timeout = float(os.getenv("VL_IMAGE_TIMEOUT_S", "12"))
        except Exception:
            eff_timeout = 12.0
    else:
        eff_timeout = float(timeout_s)
    guidelines = (
        prompt_guidelines.strip()
        if isinstance(prompt_guidelines, str) and prompt_guidelines.strip()
        else VL_PROMPT_GUIDELINES
    )

    sem = asyncio.Semaphore(VL_CONCURRENCY_LIMIT)

    async def _process(idx: int, image_url: str) -> Dict[str, Any]:
        log.debug(f"Processing image {idx}/{total}: {image_url}")
        if metrics:
            try:
                metrics.counter("x.syndication.vl_attempt").inc(1)
            except Exception:
                pass
        # Provide tweet text as context to VL adapter; keep concise/factual guidelines
        caption_ctx = f"Tweet caption: {text}" if text else f"Tweet URL: {url}"
        prompt = (
            f"{caption_ctx}\n"
            f"{guidelines}\n"
            f"Describe photo {idx} of {total} factually and concisely."
        )
        try:
            async with sem:
                try:
                    desc = await asyncio.wait_for(
                        vl_handler_func(image_url, prompt=prompt),
                        timeout=eff_timeout,
                    )
                except asyncio.TimeoutError:
                    log.warning(f"VL analysis timed out for image {idx} after {eff_timeout}s")
                    if metrics:
                        try:
                            metrics.counter("x.syndication.vl_timeout").inc(1)
                        except Exception:
                            pass
                    return {
                        "idx": idx,
                        "text": f"📷 Photo {idx}/{total} — analysis failed (timeout)",
                        "ok": False,
                    }
            if desc:
                if metrics:
                    try:
                        metrics.counter("x.syndication.vl_success").inc(1)
                    except Exception:
                        pass
                return {"idx": idx, "text": f"📷 Photo {idx}/{total}\n{desc}", "ok": True}
            else:
                if metrics:
                    try:
                        metrics.counter("x.syndication.vl_failure").inc(1)
                    except Exception:
                        pass
                return {"idx": idx, "text": f"📷 Photo {idx}/{total} — analysis unavailable", "ok": False}
        except Exception as e:
            log.warning(f"VL analysis failed for image {idx}: {e}")
            if metrics:
                try:
                    metrics.counter("x.syndication.vl_failure").inc(1)
                except Exception:
                    pass
            return {"idx": idx, "text": f"📷 Photo {idx}/{total} — analysis failed", "ok": False}

    tasks = [_process(idx, image_url) for idx, image_url in enumerate(image_urls, start=1)]
    results: List[Dict[str, Any]] = await asyncio.gather(*tasks)
    # Preserve order by idx
    results.sort(key=lambda r: r.get("idx", 0))
    descriptions = [r["text"] for r in results]
    successes = sum(1 for r in results if r.get("ok"))
    timeouts = sum(1 for r in results if isinstance(r, dict) and ("timeout" in r.get("text", "")))
    
    # Format response with tweet text and all image analyses
    base_text = text if text else f"Tweet from {url}"
    header = f"{base_text}\nPhotos analyzed: {successes}/{len(image_urls)}"
    try:
        log.info(
            "Syndication VL summary",
            extra={
                "detail": {
                    "url": url,
                    "found": len(image_urls),
                    "analyzed": successes,
                    "timeouts": timeouts,
                    "concurrency": VL_CONCURRENCY_LIMIT,
                    "timeout_s": eff_timeout,
                }
            },
        )
    except Exception:
        pass
    
    if descriptions:
        return f"{header}\n\n" + "\n\n".join(descriptions)
    else:
        return f"{base_text}\nNo images could be processed."
