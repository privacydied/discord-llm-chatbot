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
    # Prefer long-form note tweets when present, then legacy/full_text, then text
    note = tw.get("note_tweet") or {}
    text = (
        (note.get("text") if isinstance(note, dict) else None)
        or (tw.get("legacy", {}) or {}).get("full_text")
        or tw.get("full_text")
        or tw.get("text")
        or ""
    )
    include_quoted = os.getenv("SYND_INCLUDE_QUOTED_MEDIA", "true").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    def _collect_from_photos(node: Dict[str, Any]) -> List[str]:
        urls: List[str] = []
        for ph in node.get("photos") or []:
            raw = ph.get("url") or ph.get("media_url_https")
            if raw:
                urls.append(upgrade_pbs_to_orig(raw))
        return urls

    def _collect_from_entities(node: Dict[str, Any]) -> List[str]:
        urls: List[str] = []
        ee = (node.get("extended_entities") or {}).get("media") or []
        en = (node.get("entities") or {}).get("media") or []
        for m in ee or en:
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
                url = (v.get("image_value") or {}).get("url") or v.get("string_value")
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
            if any(
                tok in lc
                for tok in ("favicon", "apple-touch", "android-chrome", "icon-")
            ):
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

        METRICS.counter("x.syndication.photos_extracted").inc(
            len(tw.get("photos") or [])
        )
        METRICS.counter("x.syndication.photos_highres").inc(len(image_urls))
    except Exception:
        pass

    log.debug(
        "Syndication extract: text_len=%d chosen=%d source=%s",
        len(text),
        len(image_urls),
        source,
    )
    return {
        "text": text,
        "image_urls": image_urls,
        "source": source,
        "had_card": had_card,
    }


def syndication_has_video(tw: Dict[str, Any]) -> bool:
    """Check if syndication data indicates video or animated_gif media. [IV][REH]

    Detection strategies in priority order:
    1. Top-level 'video' field
    2. Top-level 'video_info' field
    3. extended_entities/entities for video/animated_gif types or video_info
    4. Additional video indicators (video_vars, media_duration, etc.)
    5. Quoted tweet (recursive)
    """
    if not isinstance(tw, dict):
        return False

    # Info logging: log syndication keys for video detection troubleshooting [IV][REH]
    # Use INFO level to ensure visibility in production logs
    try:
        available_keys = list(tw.keys())[:20]  # Limit to avoid huge logs
        log.info(
            "syndication_has_video: checking keys=%s",
            available_keys,
        )
    except Exception:
        pass

    # 1. Check video field directly at top level
    if tw.get("video"):
        log.info("syndication_has_video: detected via top-level 'video' field")
        return True

    # 2. Check video_info field (alternative video indicator) [REH]
    if tw.get("video_info"):
        log.info("syndication_has_video: detected via 'video_info' field")
        return True

    # 3. Check extended_entities/entities for video/animated_gif types
    for entities_key in ("extended_entities", "entities"):
        ent = tw.get(entities_key) or {}
        media_list = ent.get("media") or []
        if not media_list:
            continue
        for m in media_list:
            if not isinstance(m, dict):
                continue
            mtype = (m.get("type") or "").lower()
            if mtype in ("video", "animated_gif"):
                log.info(
                    "syndication_has_video: detected via %s media type=%s",
                    entities_key,
                    mtype,
                )
                return True
            # Additional check: video_info inside media entity [REH]
            if m.get("video_info"):
                log.info("syndication_has_video: detected via media.video_info")
                return True
            # Check for video_variants (another possible indicator) [REH]
            if m.get("video_variants") or m.get("video_urls"):
                log.info("syndication_has_video: detected via media video_variants/video_urls")
                return True

    # 4. Additional video indicators that might be present [REH]
    # Check for media_duration (present in video tweets)
    if tw.get("media_duration") or tw.get("duration_ms"):
        log.info("syndication_has_video: detected via media_duration/duration_ms field")
        return True
    # Check for video_variants at top level
    if tw.get("video_variants") or tw.get("video_urls"):
        log.info("syndication_has_video: detected via top-level video_variants/video_urls")
        return True

    # 5. Check if there's media but no photos (strong video indicator) [REH]
    # Some video tweets only have a 'media' array without explicit type field
    has_media = bool(tw.get("media"))
    has_photos = bool(tw.get("photos"))
    if has_media and not has_photos:
        # Additional check: if media exists and we can verify at least one entry
        media_list = tw.get("media") or []
        if media_list and all(isinstance(m, dict) for m in media_list):
            # Check if any media entry lacks 'type' or has non-photo characteristics
            for m in media_list:
                mtype = (m.get("type") or "").lower()
                # If type is missing or not explicitly "photo", likely video
                if not mtype or mtype not in ("photo", "image"):
                    log.info(
                        "syndication_has_video: detected via media without photo type, type=%s",
                        mtype or "missing",
                    )
                    return True
                # Check for video indicators within media entry
                if m.get("video_info") or m.get("video_variants") or m.get("video_urls"):
                    log.info("syndication_has_video: detected via media entry with video indicators")
                    return True

    # 6. Check quoted tweet as well (recursively) [IV][REH]
    # Check for both quoted_tweet and quoted_status (different field names in different API versions)
    for qt_key in ("quoted_tweet", "quoted_status"):
        qt = tw.get(qt_key) or {}
        if isinstance(qt, dict) and qt:
            # Check quoted tweet's video field
            if qt.get("video") or qt.get("video_info"):
                log.info(f"syndication_has_video: detected via {qt_key} video field")
                return True
            # Check quoted tweet's entities
            for entities_key in ("extended_entities", "entities"):
                ent = qt.get(entities_key) or {}
                media_list = ent.get("media") or []
                for m in media_list:
                    if not isinstance(m, dict):
                        continue
                    mtype = (m.get("type") or "").lower()
                    if mtype in ("video", "animated_gif"):
                        log.info(
                            f"syndication_has_video: detected via {qt_key} {entities_key}",
                        )
                        return True
                    if m.get("video_info"):
                        log.info(f"syndication_has_video: detected via {qt_key} media.video_info")
                        return True

    # 7. Check retweeted status as well [REH]
    rt = tw.get("retweeted_status") or {}
    if isinstance(rt, dict) and rt:
        # Check retweeted status's video field
        if rt.get("video") or rt.get("video_info"):
            log.info("syndication_has_video: detected via retweeted_status video field")
            return True
        # Check retweeted status's entities
        for entities_key in ("extended_entities", "entities"):
            ent = rt.get(entities_key) or {}
            media_list = ent.get("media") or []
            for m in media_list:
                if not isinstance(m, dict):
                    continue
                mtype = (m.get("type") or "").lower()
                if mtype in ("video", "animated_gif"):
                    log.info(
                        "syndication_has_video: detected via retweeted_status %s",
                        entities_key,
                    )
                    return True
                if m.get("video_info"):
                    log.info("syndication_has_video: detected via retweeted_status media.video_info")
                    return True

    # 8. Check legacy format (data nested under 'legacy' key) [REH]
    # Some syndication responses wrap the actual data under a 'legacy' key
    legacy = tw.get("legacy") or {}
    if isinstance(legacy, dict) and legacy:
        # Check for video indicators in legacy data
        if legacy.get("video") or legacy.get("video_info"):
            log.info("syndication_has_video: detected via legacy video field")
            return True
        if legacy.get("media_duration") or legacy.get("duration_ms"):
            log.info("syndication_has_video: detected via legacy media_duration field")
            return True
        if legacy.get("video_variants") or legacy.get("video_urls"):
            log.info("syndication_has_video: detected via legacy video_variants")
            return True
        # Check legacy's extended_entities for video media
        for entities_key in ("extended_entities", "entities"):
            ent = legacy.get(entities_key) or {}
            media_list = ent.get("media") or []
            for m in media_list:
                if not isinstance(m, dict):
                    continue
                mtype = (m.get("type") or "").lower()
                if mtype in ("video", "animated_gif"):
                    log.info(
                        "syndication_has_video: detected via legacy %s",
                        entities_key,
                    )
                    return True
                if m.get("video_info"):
                    log.info("syndication_has_video: detected via legacy media.video_info")
                    return True

    log.info("syndication_has_video: no video detected")
    return False
