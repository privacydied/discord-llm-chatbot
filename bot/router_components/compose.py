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
