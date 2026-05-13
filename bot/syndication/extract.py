"""
Extraction utilities for syndication content processing.
Implements strict media selection policy for X/Twitter syndication payloads.
"""

from typing import List, Dict, Any, Optional, Iterable, Tuple
from .url_utils import upgrade_pbs_to_orig, pbs_base_key
import logging
import os
import re
from html import unescape

log = logging.getLogger(__name__)


def _iter_syndication_media_entries(
    node: Dict[str, Any],
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield media entries from common syndication containers with source labels."""
    if not isinstance(node, dict):
        return

    top_media = node.get("media") or []
    if isinstance(top_media, list):
        for m in top_media:
            if isinstance(m, dict):
                yield ("media", m)

    for entities_key in ("extended_entities", "entities"):
        entities = node.get(entities_key) or {}
        if not isinstance(entities, dict):
            continue
        media = entities.get("media") or []
        if not isinstance(media, list):
            continue
        for m in media:
            if isinstance(m, dict):
                yield (f"{entities_key}.media", m)


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

    def _extract_article_text(article_node: Dict[str, Any]) -> str:
        if not isinstance(article_node, dict):
            return ""
        parts: List[str] = []
        title = str(article_node.get("title") or "").strip()
        preview = str(article_node.get("preview_text") or "").strip()
        if title:
            parts.append(unescape(title))
        if preview:
            parts.append(unescape(preview))
        content = article_node.get("content") or {}
        blocks = content.get("blocks") if isinstance(content, dict) else []
        if isinstance(blocks, list):
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                btxt = unescape(str(block.get("text") or "")).strip()
                if btxt and btxt not in parts:
                    parts.append(btxt)
        merged = "\n\n".join(parts).strip()
        max_chars = 12000
        if len(merged) > max_chars:
            return merged[: max_chars - 1].rstrip() + "…"
        return merged

    # Prefer long-form note tweets when present, then legacy/full_text, then text/article
    note = tw.get("note_tweet") or {}
    base_text = (note.get("text") if isinstance(note, dict) else None) or (tw.get("legacy", {}) or {}).get("full_text") or tw.get("full_text") or tw.get("text") or ""
    base_text = (base_text or "").strip()
    article_text = _extract_article_text(tw.get("article") or {})
    if article_text:
        if base_text and not re.search(r"https?://t\.co/[A-Za-z0-9]+", base_text):
            text = base_text if article_text in base_text else f"{base_text}\n\n[Linked X Article]\n{article_text}"
        else:
            text = article_text
    else:
        text = base_text
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
        for _source, m in _iter_syndication_media_entries(node):
            try:
                mtype = (m.get("type") or "").lower()
                raw: Optional[str] = None
                if mtype == "photo":
                    raw = m.get("media_url_https") or m.get("url")
                elif mtype in ("video", "animated_gif"):
                    # Prefer poster/thumbnail; fallback to media_url_https/url if present
                    raw = m.get("thumbnail_url") or m.get("poster") or m.get("media_url_https") or m.get("url")
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

    # 2) Quoted/retweeted fallback (only if no primary native media)
    if not primary_urls and include_quoted:
        fallback_nodes = [
            ("quoted_tweet", tw.get("quoted_tweet") or {}),
            ("quoted_status", tw.get("quoted_status") or {}),
            ("retweeted_status", tw.get("retweeted_status") or {}),
        ]
        for node_name, node in fallback_nodes:
            if not isinstance(node, dict) or not node:
                continue
            q_urls = _collect_from_photos(node)
            if not q_urls:
                q_urls = _collect_from_entities(node)
            if q_urls:
                primary_urls = q_urls
                source_aliases = {
                    "quoted_tweet": "quoted_photos",
                    "quoted_status": "quoted_status_photos",
                    "retweeted_status": "retweeted_status_photos",
                }
                source = source_aliases.get(node_name, f"{node_name}_photos")
                had_card = had_card or bool(_extract_card_url(node))
                break

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

    def _log_node_keys(node_name: str, node: Dict[str, Any]) -> None:
        try:
            keys = sorted(list(node.keys()))
            log.info("syndication_has_video: %s.keys=%s", node_name, keys)
        except Exception:
            pass

    def _node_has_video(node_name: str, node: Dict[str, Any]) -> bool:
        if not isinstance(node, dict) or not node:
            return False

        _log_node_keys(node_name, node)

        # Direct node-level video markers
        if node.get("video"):
            log.info("syndication_has_video: detected via %s.video", node_name)
            return True
        if node.get("video_info"):
            log.info("syndication_has_video: detected via %s.video_info", node_name)
            return True
        if node.get("video_variants") or node.get("video_urls"):
            log.info(
                "syndication_has_video: detected via %s.video_variants/video_urls",
                node_name,
            )
            return True
        if node.get("media_duration") or node.get("duration_ms"):
            log.info(
                "syndication_has_video: detected via %s.media_duration/duration_ms",
                node_name,
            )
            return True

        for idx, (source, media) in enumerate(_iter_syndication_media_entries(node)):
            mtype = (media.get("type") or "").lower()
            has_video_info = bool(media.get("video_info"))
            has_variants = bool(media.get("video_variants") or media.get("video_urls"))
            has_duration = bool(media.get("duration_ms") or media.get("media_duration"))
            has_poster = bool(media.get("poster") or media.get("thumbnail_url") or media.get("media_url_https"))
            log.info(
                "syndication_has_video: media_check node=%s source=%s idx=%d type=%s has_video_info=%s has_variants=%s has_duration=%s has_poster=%s",
                node_name,
                source,
                idx,
                mtype or "missing",
                has_video_info,
                has_variants,
                has_duration,
                has_poster,
            )

            if mtype in ("video", "animated_gif"):
                log.info(
                    "syndication_has_video: detected via %s %s type=%s",
                    node_name,
                    source,
                    mtype,
                )
                return True
            if has_video_info or has_variants:
                log.info(
                    "syndication_has_video: detected via %s %s video markers",
                    node_name,
                    source,
                )
                return True
            if mtype and mtype not in ("photo", "image"):
                log.info(
                    "syndication_has_video: detected via %s %s non-photo type=%s",
                    node_name,
                    source,
                    mtype,
                )
                return True

        return False

    def _node_has_media_hints(node: Dict[str, Any]) -> bool:
        if not isinstance(node, dict) or not node:
            return False
        if any(
            k in node
            for k in (
                "media",
                "photos",
                "video",
                "video_info",
                "video_variants",
                "video_urls",
                "media_duration",
                "duration_ms",
                "extended_entities",
                "entities",
                "card",
                "image",
            )
        ):
            return True
        for _source, _media in _iter_syndication_media_entries(node):
            return True
        return False

    # Evaluate all known nesting shapes deterministically.
    nodes_to_scan: List[Tuple[str, Dict[str, Any]]] = [("tweet", tw)]
    for key in ("quoted_tweet", "quoted_status", "retweeted_status", "legacy"):
        node = tw.get(key) or {}
        if isinstance(node, dict) and node:
            log.info("syndication_has_video: checking nested node=%s", key)
            nodes_to_scan.append((key, node))

    saw_media_hints = False
    for node_name, node in nodes_to_scan:
        if _node_has_media_hints(node):
            saw_media_hints = True
        if _node_has_video(node_name, node):
            return True

    if not saw_media_hints:
        log.info("syndication_has_video: indeterminate (no media metadata)")
        return False
    log.info("syndication_has_video: no video detected")
    return False
