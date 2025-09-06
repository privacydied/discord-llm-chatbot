"""
Extraction utilities for syndication content processing.
Implements strict media selection policy for X/Twitter syndication payloads.
"""
from typing import List, Dict, Any, Optional
from .url_utils import upgrade_pbs_to_orig, pbs_base_key
import logging
import os

log = logging.getLogger(__name__)


def extract_text_and_images_from_syndication(tw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a dict with: { "text": str, "image_urls": List[str], "source": str, "had_card": bool }
    Policy:
      1) Primary native media first (photos array). If empty, check entities/extended_entities for photos.
         For video/animated_gif, use poster/thumbnail image when available.
      2) If primary has no native media and SYND_INCLUDE_QUOTED_MEDIA=true, use quoted tweet native media.
      3) If still none, fall back to link card image (largest available), excluding icons/placeholders.
      4) High-res: upgrade pbs URLs to name=orig; handle legacy :size suffix.
      5) Dedup: compare by base asset (strip query and :size) while preserving order.
    """
    text = tw.get("full_text") or tw.get("text") or ""
    include_quoted = os.getenv("SYND_INCLUDE_QUOTED_MEDIA", "true").lower() in ("1", "true", "yes", "on")

    def _collect_from_photos(node: Dict[str, Any]) -> List[str]:
        urls: List[str] = []
        for ph in (node.get("photos") or []):
            raw = ph.get("url") or ph.get("media_url_https")
            if raw:
                urls.append(upgrade_pbs_to_orig(raw))
        return urls

    def _collect_from_entities(node: Dict[str, Any]) -> List[str]:
        urls: List[str] = []
        ee = (node.get("extended_entities") or {}).get("media") or []
        en = (node.get("entities") or {}).get("media") or []
        for m in (ee or en):
            try:
                mtype = (m.get("type") or "").lower()
                raw: Optional[str] = None
                if mtype == "photo":
                    raw = m.get("media_url_https") or m.get("url")
                elif mtype in ("video", "animated_gif"):
                    # Prefer poster/thumbnail; fallback to media_url_https/url if present
                    raw = (
                        m.get("thumbnail_url")
                        or m.get("poster")
                        or m.get("media_url_https")
                        or m.get("url")
                    )
                if raw:
                    urls.append(upgrade_pbs_to_orig(raw))
            except Exception:
                continue
        return urls

    def _extract_card_url(node: Dict[str, Any]) -> Optional[str]:
        # Prefer card.binding_values.photo_image_full_size_large or similar; fallback to top-level image
        card = node.get("card") or {}
        bv = card.get("binding_values") or {}
        candidates: List[Optional[str]] = []
        # Known preferred keys in rough order
        pref_keys = [
            "photo_image_full_size_large",
            "photo_image_full_size",
            "thumbnail_image_large",
            "thumbnail_image",
        ]
        for k in pref_keys:
            v = bv.get(k)
            if isinstance(v, dict):
                url = (
                    (v.get("image_value") or {}).get("url")
                    or v.get("string_value")
                )
                if url:
                    candidates.append(url)
        # Fallback to top-level 'image'
        img = node.get("image")
        if isinstance(img, dict):
            candidates.append(img.get("url"))
        elif isinstance(img, str):
            candidates.append(img)

        # Filter out icon-ish assets
        filtered: List[str] = []
        for c in candidates:
            if not c:
                continue
            lc = c.lower()
            if any(tok in lc for tok in ("favicon", "apple-touch", "android-chrome", "icon-")):
                continue
            filtered.append(c)
        if filtered:
            return upgrade_pbs_to_orig(filtered[0])
        return None

    # 1) Primary native media
    primary_urls = _collect_from_photos(tw)
    source = "photos"
    had_card = bool(_extract_card_url(tw))

    if not primary_urls:
        ent_urls = _collect_from_entities(tw)
        if ent_urls:
            primary_urls = ent_urls
            source = "photos"  # treat as native photos/thumbnail selection

    # 2) Quoted tweet fallback (only if no primary native media)
    if not primary_urls and include_quoted:
        qt = tw.get("quoted_tweet") or {}
        q_urls = _collect_from_photos(qt)
        q_source = "quoted_photos"
        if not q_urls:
            ent_q_urls = _collect_from_entities(qt)
            if ent_q_urls:
                q_urls = ent_q_urls
        if q_urls:
            primary_urls = q_urls
            source = q_source
            # Also capture if quoted had card (not used for selection)
            had_card = had_card or bool(_extract_card_url(qt))

    # 3) Card image fallback (only if neither primary nor quoted have native media)
    if not primary_urls:
        card_url = _extract_card_url(tw)
        if card_url:
            primary_urls = [card_url]
            source = "card"

    # 4) Dedup by base asset (strip params and :size), preserve order
    seen_keys = set()
    deduped: List[str] = []
    for u in primary_urls:
        key = pbs_base_key(u)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(u)

    image_urls = deduped

    # Metrics (optional, non-breaking)
    try:
        from bot.metrics import METRICS  # type: ignore
        METRICS.counter("x.syndication.photos_extracted").inc(len(tw.get("photos") or []))
        METRICS.counter("x.syndication.photos_highres").inc(len(image_urls))
    except Exception:
        pass

    log.debug(
        "Syndication extract: text_len=%d chosen=%d source=%s",
        len(text), len(image_urls), source,
    )
    return {"text": text, "image_urls": image_urls, "source": source, "had_card": had_card}
