"""
Syndication handler for Twitter/X content to VL flow integration.
"""
from typing import Dict, Any, List, Optional
from .extract import extract_text_and_images_from_syndication
import logging

log = logging.getLogger(__name__)

# Guidelines injected into VL prompts to ensure consistent, compact, factual image descriptions
# that ignore non-content UI chrome and engagement metrics.
VL_PROMPT_GUIDELINES = (
    "describe only the actual picture. ignore ui chrome, cookie/login banners, overlays, "
    "timestamps, watermarks, likes/retweets/views and other page furniture. be factual and "
    "concise. include any visible text in quotes exactly as seen; skip illegible text. don't "
    "speculate—if unclear, say 'unclear'."
)


async def handle_twitter_syndication_to_vl(
    tweet_json: Dict[str, Any],
    url: str,
    vl_handler_func,
    prompt_guidelines: Optional[str] = None,
) -> str:
    """
    Handle Twitter/X link to VL flow using syndication data and full-resolution images.
    
    Args:
        tweet_json: Syndication JSON data from Twitter/X
        url: Original tweet URL for context
        vl_handler_func: Function to handle individual image VL analysis (e.g., _vl_describe_image_from_url)
        prompt_guidelines: Optional override for VL prompt guidelines; if None, uses default VL_PROMPT_GUIDELINES.
        
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
    successes = 0
    failures = 0
    metrics = None

    # Metrics are optional and non-breaking
    try:
        from bot.metrics import METRICS  # type: ignore

        metrics = METRICS
    except Exception:
        metrics = None
    
    for idx, image_url in enumerate(image_urls, start=1):
        log.debug(f"Processing image {idx}/{len(image_urls)}: {image_url}")

        guidelines = (prompt_guidelines.strip() if isinstance(prompt_guidelines, str) and prompt_guidelines.strip() else VL_PROMPT_GUIDELINES)
        prompt = (
            f"photo {idx} of {len(image_urls)} from a tweet: {url}. "
            f"{guidelines}"
        )

        
        try:
            if metrics:
                try:
                    metrics.counter("x.syndication.vl_attempt").inc(1)
                except Exception:
                    pass
            desc = await vl_handler_func(image_url, prompt=prompt)
            if desc:
                successes += 1
                if metrics:
                    try:
                        metrics.counter("x.syndication.vl_success").inc(1)
                    except Exception:
                        pass
                descriptions.append(f"📷 Photo {idx}/{len(image_urls)}\n{desc}")
            else:
                failures += 1
                if metrics:
                    try:
                        metrics.counter("x.syndication.vl_failure").inc(1)
                    except Exception:
                        pass
                descriptions.append(f"📷 Photo {idx}/{len(image_urls)} — analysis unavailable")
        except Exception as e:
            log.warning(f"VL analysis failed for image {idx}: {e}")
            failures += 1
            if metrics:
                try:
                    metrics.counter("x.syndication.vl_failure").inc(1)
                except Exception:
                    pass
            descriptions.append(f"📷 Photo {idx}/{len(image_urls)} — analysis failed")
    
    # Format response with tweet text and all image analyses
    base_text = text if text else f"Tweet from {url}"
    header = f"{base_text}\nPhotos analyzed: {successes}/{len(image_urls)}"
    
    if descriptions:
        return f"{header}\n\n" + "\n\n".join(descriptions)
    else:
        return f"{base_text}\nNo images could be processed."
