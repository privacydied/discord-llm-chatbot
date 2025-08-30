"""
Extraction utilities for syndication content processing.
"""
from typing import List, Dict, Any
from .url_utils import upgrade_pbs_to_orig
import logging

log = logging.getLogger(__name__)


def extract_text_and_images_from_syndication(tw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns: { "text": str, "image_urls": List[str] }
    Backward compatible: if no photos, may return an empty list or a single fallback image.
    
    Args:
        tw: Syndication JSON data from Twitter/X
        
    Returns:
        Dictionary containing extracted text and high-resolution image URLs
    """
    text = tw.get("full_text") or tw.get("text") or ""
    image_urls: List[str] = []

    # Primary: photos[]
    photos = tw.get("photos") or []
    for ph in photos:
        # Common shapes from syndication: 'url' (pbs…?format=jpg&name=small),
        # sometimes 'media_url_https'
        raw = ph.get("url") or ph.get("media_url_https")
        if not raw:
            continue
        image_urls.append(upgrade_pbs_to_orig(raw))

    # Fallbacks when no photos:
    if not image_urls:
        # 1) 'image' at top-level (card/og)
        fallback = (tw.get("image") or {}).get("url") if isinstance(tw.get("image"), dict) else tw.get("image")
        if fallback:
            image_urls.append(upgrade_pbs_to_orig(fallback))

        # 2) quoted tweet images (optional, non-breaking)
        qt = tw.get("quoted_tweet") or {}
        if qt and not image_urls:  # only if we still have none to preserve old behavior
            qphotos = qt.get("photos") or []
            for ph in qphotos:
                raw = ph.get("url") or ph.get("media_url_https")
                if raw:
                    image_urls.append(upgrade_pbs_to_orig(raw))

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for u in image_urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    image_urls = deduped

    # Metrics (optional, non-breaking)
    try:
        from bot.metrics import METRICS
        METRICS.counter("x.syndication.photos_extracted").inc(len(photos))
        METRICS.counter("x.syndication.photos_highres").inc(len(image_urls))
    except Exception:
        pass

    log.debug("Syndication extract: text_len=%d photos=%d highres=%d",
              len(text), len(photos), len(image_urls))
    return {"text": text, "image_urls": image_urls}
