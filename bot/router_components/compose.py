"""Pure composition helpers extracted from Router for phased decomposition."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from bot.evidence import EvidenceBundle


def format_x_tweet_with_transcription(
    *,
    base_text: Optional[str],
    url: str,
    stt_res: Dict[str, Any],
    tweet_data: Optional[Dict[str, Any]],
    extract_primary_tweet_id: Optional[Callable[[str], Optional[str]]] = None,
) -> str:
    """Assemble a single evidence bundle for a tweet using caption + STT."""
    bundle = EvidenceBundle(source_platform="x", source_url=url)

    # Primary ID anchor for deterministic media selection [CMV]
    try:
        if callable(extract_primary_tweet_id):
            ptid = extract_primary_tweet_id(url)
            if ptid:
                bundle.primary_tweet_id = ptid
                bundle.selected_tweet_id = ptid
    except Exception:
        pass

    # Caption population: prefer tweet_data text; fallback to base_text heuristic [IV]
    try:
        caption = ""
        if tweet_data and isinstance(tweet_data, dict):
            caption = (tweet_data.get("full_text") or tweet_data.get("text") or "").strip()
        if not caption and base_text:
            try:
                base_str = str(base_text)
                m = re.search(
                    r"\[Tweet Caption\]\s*\n(?P<body>.*?)(?:\n\n\[|\Z)",
                    base_str,
                    flags=re.DOTALL,
                )
                if m:
                    caption = (m.group("body") or "").strip()
            except Exception:
                caption = ""
        if not caption and base_text:
            try:
                lines = [ln.strip() for ln in str(base_text).splitlines() if ln.strip()]
                for ln in lines:
                    if ln.startswith("[") and ln.endswith("]"):
                        continue
                    if ln.startswith("— "):
                        continue
                    if ln.lower().startswith("http://") or ln.lower().startswith(
                        "https://"
                    ):
                        continue
                    caption = ln
                    break
            except Exception:
                caption = (base_text or "").strip()
        if caption:
            bundle.caption_text = caption
    except Exception:
        pass

    # Quoted/retweet text when provided [IV]
    try:
        if tweet_data and isinstance(tweet_data, dict):
            q = tweet_data.get("quoted_status") or {}
            if isinstance(q, dict):
                qt = (q.get("full_text") or q.get("text") or "").strip()
                if qt:
                    bundle.quoted_text = qt
            if not bundle.quoted_text:
                r = tweet_data.get("retweeted_status") or {}
                if isinstance(r, dict):
                    rt = (r.get("full_text") or r.get("text") or "").strip()
                    if rt:
                        bundle.quoted_text = rt
    except Exception:
        pass

    # STT transcript with low-speech guard [REH]
    try:
        transcript = ((stt_res or {}).get("transcription") or "").strip()
        if transcript:
            bundle.media_transcript = transcript
        else:
            bundle.media_transcript = ""
            bundle.stt_no_speech = True
    except Exception:
        bundle.media_transcript = ""

    # Concatenate caption + transcript for video tweets before text flow [REH]
    try:
        if bundle.caption_text and bundle.media_transcript:
            combined = (
                f"{bundle.caption_text.strip()}\n\n{bundle.media_transcript.strip()}"
            )
            bundle.add_section(
                kind="caption_transcript",
                title="Tweet Caption + Audio Transcript",
                body=combined,
                provenance={"source": "tweet_text+stt"},
            )
            bundle.caption_text = ""
            bundle.media_transcript = ""
    except Exception:
        pass

    return bundle.compose_prompt_text()


def format_x_tweet_result(
    *,
    api_data: Dict[str, Any],
    url: str,
    canonicalize_status_url: Callable[[str], str],
) -> str:
    """Format X API tweet response into concise text."""
    try:
        payload = api_data or {}
        tweet = (
            payload.get("data")
            if isinstance(payload.get("data"), dict)
            else payload
        )
        tweet = tweet or {}
        includes = (
            payload.get("includes")
            if isinstance(payload.get("includes"), dict)
            else {}
        )
        media_list = (
            includes.get("media")
            if isinstance(includes.get("media"), list)
            else []
        )

        text = (tweet.get("full_text") or tweet.get("text") or "").strip()

        user = ""
        try:
            users = includes.get("users") if isinstance(includes, dict) else None
            if isinstance(users, list):
                author_id = str(tweet.get("author_id") or "").strip()
                for user_item in users:
                    if not isinstance(user_item, dict):
                        continue
                    if author_id and str(user_item.get("id") or "").strip() != author_id:
                        continue
                    user = (
                        user_item.get("name")
                        or user_item.get("username")
                        or user_item.get("screen_name")
                        or ""
                    ).strip()
                    if user:
                        break
        except Exception:
            user = ""

        if not user:
            user_obj = (
                tweet.get("user") if isinstance(tweet.get("user"), dict) else {}
            )
            user = (user_obj.get("name") or user_obj.get("screen_name") or "").strip()

        photo_count = 0
        try:
            photo_count = sum(
                1
                for media in media_list
                if isinstance(media, dict) and media.get("type") == "photo"
            )
        except Exception:
            photo_count = 0

        parts: List[str] = []
        if text:
            parts.append(text)
        if photo_count:
            parts.append(f"Photos: {photo_count}")
        if user:
            parts.append(f"— {user}")
        parts.append(canonicalize_status_url(url))

        out = "\n".join(parts).strip()
        return out or canonicalize_status_url(url)
    except Exception:
        return canonicalize_status_url(url)


def has_visual_facts_section(content: str) -> bool:
    """Detect whether prompt content already contains visual-facts evidence blocks."""
    text = content or ""
    text_lower = text.lower()
    return (
        "visual_facts:" in text_lower
        or "vl prompt output:" in text_lower
        or bool(re.search(r"^image\s+\d+:", text, re.IGNORECASE | re.MULTILINE))
        or "tweet caption:" in text_lower
        # Attachment and direct image URL routes put successful VL output into
        # these labels before the text flow synthesizes a final reply. Treat
        # them as visual facts too, otherwise the downstream model can still
        # drift into "I can't see the image" even after VL succeeded. [REH]
        or "image analysis" in text_lower
        or bool(re.search(r"^\[image:\s*[^\]]+\]", text, re.IGNORECASE | re.MULTILINE))
    )


def build_visual_analysis_anchor_prompt(base_system_prompt: str) -> str:
    """Build the canonical visual-analysis anchoring instruction block."""
    base_sys = base_system_prompt or "You are a helpful assistant."
    return (
        f"{base_sys}\n\n[VISUAL-ANALYSIS-ANCHOR]\n"
        "- If the user prompt includes a section titled 'vl prompt output:', lines beginning with 'Image n:',\n"
        "  '[IMAGE: ...]' blocks, 'Image Analysis' blocks, or perception notes, treat these as\n"
        "  non-negotiable visual facts extracted from the image(s).\n"
        "- Base your reply on those facts and the user's request.\n"
        "- Do not claim there is no image or that you cannot see images when such analysis is provided.\n"
        "- Screenshots of documents or text are still images; do not dismiss them as 'not a pic'.\n"
        "- Do not ask the user to resend or post the image; assume the VISUAL_FACTS reflect what was shown.\n"
        "- Keep persona, tone, and safety rules intact."
    )


def compose_x_tweet_with_visual_facts(
    *,
    user_text: Optional[str],
    tweet_caption: Optional[str],
    vl_notes: Optional[str],
) -> str:
    """Compose text-flow input for image tweets with caption + VL facts."""
    clean_user = (user_text or "").strip()
    clean_caption = (tweet_caption or "").strip()
    clean_vl = (vl_notes or "").strip()

    if not clean_caption and not clean_vl:
        return clean_user

    lines: List[str] = []
    if clean_user:
        lines.append(clean_user)
        lines.append("")

    lines.append("VISUAL_FACTS:")
    lines.append("tweet caption:")
    lines.append(clean_caption or "—")

    if clean_vl:
        lines.append("")
        lines.append("vl prompt output:")
        lines.append(clean_vl)

    return "\n".join(lines).strip()
