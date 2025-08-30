"""
Syndication handler for Twitter/X content to VL flow integration.
"""
from typing import Dict, Any, List, Optional
from .extract import extract_text_and_images_from_syndication
import logging

log = logging.getLogger(__name__)


async def handle_twitter_syndication_to_vl(tweet_json: Dict[str, Any], url: str, vl_handler_func) -> str:
    """
    Handle Twitter/X link to VL flow using syndication data and full-resolution images.
    
    Args:
        tweet_json: Syndication JSON data from Twitter/X
        url: Original tweet URL for context
        vl_handler_func: Function to handle individual image VL analysis (e.g., _vl_describe_image_from_url)
        
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
    
    for idx, image_url in enumerate(image_urls, start=1):
        log.debug(f"Processing image {idx}/{len(image_urls)}: {image_url}")
        
        prompt = (
            f"This is photo {idx} of {len(image_urls)} from a tweet: {url}. "
            f"Describe it clearly and succinctly, including any visible text."
        )
        
        try:
            desc = await vl_handler_func(image_url, prompt=prompt)
            if desc:
                successes += 1
                descriptions.append(f"📷 Photo {idx}/{len(image_urls)}\n{desc}")
            else:
                failures += 1
                descriptions.append(f"📷 Photo {idx}/{len(image_urls)} — analysis unavailable")
        except Exception as e:
            log.warning(f"VL analysis failed for image {idx}: {e}")
            failures += 1
            descriptions.append(f"📷 Photo {idx}/{len(image_urls)} — analysis failed")
    
    # Format response with tweet text and all image analyses
    base_text = text if text else f"Tweet from {url}"
    header = f"{base_text}\nPhotos analyzed: {successes}/{len(image_urls)}"
    
    if descriptions:
        return f"{header}\n\n" + "\n\n".join(descriptions)
    else:
        return f"{base_text}\nNo images could be processed."
