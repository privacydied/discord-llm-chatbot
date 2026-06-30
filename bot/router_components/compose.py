"""Pure composition helpers extracted from Router for phased decomposition."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from bot.evidence import EvidenceBundle

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def format_x_tweet_with_transcription(
    *,
    base_text: str | None,
    url: str,
    stt_res: dict[str, Any],
    tweet_data: dict[str, Any] | None,
    extract_primary_tweet_id: Callable[[str], str | None] | None = None,
) -> str:
    """Assemble a single evidence bundle for a tweet using caption + STT."""
    bundle = EvidenceBundle()

    # Primary ID anchor for deterministic media selection [CMV]
    try:
        if callable(extract_primary_tweet_id):
            ptid = extract_primary_tweet_id(url)
            if ptid:
                bundle.primary_tweet_id = ptid
                bundle.selected_tweet_id = ptid
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        logger.debug(f"primary tweet id extraction failed: {exc}")

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
            except (AttributeError, TypeError, ValueError, re.error) as exc:
                logger.debug(f"caption regex extraction failed: {exc}")
                caption = ""
        if not caption and base_text:
            try:
                lines = [ln.strip() for ln in str(base_text).splitlines() if ln.strip()]
                for ln in lines:
                    if ln.startswith("[") and ln.endswith("]"):
                        continue
                    if ln.startswith("— "):
                        continue
                    if ln.lower().startswith("http://") or ln.lower().startswith("https://"):
                        continue
                    caption = ln
                    break
            except (AttributeError, TypeError, ValueError) as exc:
                logger.debug(f"caption line extraction failed: {exc}")
                caption = (base_text or "").strip()
        if caption:
            bundle.caption_text = caption
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        logger.debug(f"caption population failed: {exc}")

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
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        logger.debug(f"quoted/retweet extraction failed: {exc}")

    # STT transcript with low-speech guard [REH]
    try:
        raw_transcript = (stt_res or {}).get("transcription")
        # Handle malformed tuple-shaped transcription [BUGFIX]
        # When transcription is a tuple (text, meta), extract the text part
        if isinstance(raw_transcript, tuple):
            transcript = str(raw_transcript[0] if raw_transcript else "").strip()
            # Optionally, we could extract confidence from the tuple for logging
        elif isinstance(raw_transcript, str):
            transcript = raw_transcript.strip()
        else:
            transcript = str(raw_transcript or "").strip()

        if transcript:
            bundle.media_transcript = transcript
        else:
            bundle.media_transcript = ""
            bundle.stt_no_speech = True
    except (AttributeError, TypeError, ValueError, KeyError, IndexError) as exc:
        logger.debug(f"STT transcript handling failed: {exc}")
        bundle.media_transcript = ""

    # Concatenate caption + transcript for video tweets before text flow [REH]
    # NOTE: EvidenceBundle.extra_sections is a plain dict[str, str]; the key
    # doubles as the section label rendered by compose(). [CMV]
    CAPTION_TRANSCRIPT_LABEL = "Tweet Caption + Audio Transcript"
    AUDIO_TRANSCRIPT_LABEL = "Audio Transcript"
    CAPTION_ONLY_LABEL = "Tweet Caption"
    try:
        if bundle.caption_text and bundle.media_transcript:
            combined = f"{bundle.caption_text.strip()}\n\n{bundle.media_transcript.strip()}"
            bundle.extra_sections[CAPTION_TRANSCRIPT_LABEL] = combined
            bundle.caption_text = ""
            bundle.media_transcript = ""
        elif bundle.media_transcript:
            # Transcript-only: move out of the default "transcript" label into a
            # clearly-named section so grounding instructions read naturally.
            bundle.extra_sections[AUDIO_TRANSCRIPT_LABEL] = bundle.media_transcript.strip()
            bundle.media_transcript = ""
        elif bundle.caption_text:
            # Caption-only: same relabeling for a consistent, human-readable section name.
            bundle.extra_sections[CAPTION_ONLY_LABEL] = bundle.caption_text.strip()
            bundle.caption_text = ""
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        logger.debug(f"caption_transcript concatenation failed: {exc}")

    # Add STT grounding instructions to prevent "I can't process audio" responses [REH]
    # This ensures the model knows STT succeeded and uses the transcript
    has_transcript_section = CAPTION_TRANSCRIPT_LABEL in bundle.extra_sections or AUDIO_TRANSCRIPT_LABEL in bundle.extra_sections
    if has_transcript_section:
        grounding = (
            "\n\n[STT GROUNDING]\n"
            "- The audio/video was transcribed by STT. Use the transcript above as the source.\n"
            "- If the transcript appears malformed or low-confidence, note uncertainty\n"
            "  but do NOT claim the audio cannot be processed or that you cannot access it.\n"
            "- For non-English transcripts, translate or summarize based on the content.\n"
            "- Do not ask the user to provide the audio in another format.\n"
        )
        # Append grounding to extra_sections as an instruction block
        bundle.extra_sections["STT Instructions"] = grounding

    return bundle.compose()


def format_x_tweet_result(
    *,
    api_data: dict[str, Any],
    url: str,
    canonicalize_status_url: Callable[[str], str],
) -> str:
    """Format X API tweet response into concise text."""
    try:
        payload = api_data or {}
        tweet = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        tweet = tweet or {}
        includes = payload.get("includes") if isinstance(payload.get("includes"), dict) else {}
        media_list = includes.get("media") if isinstance(includes.get("media"), list) else []

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
                    user = (user_item.get("name") or user_item.get("username") or user_item.get("screen_name") or "").strip()
                    if user:
                        break
        except (AttributeError, TypeError, ValueError, KeyError):
            user = ""

        if not user:
            user_obj = tweet.get("user") if isinstance(tweet.get("user"), dict) else {}
            user = (user_obj.get("name") or user_obj.get("screen_name") or "").strip()

        photo_count = 0
        try:
            photo_count = sum(1 for media in media_list if isinstance(media, dict) and media.get("type") == "photo")
        except (AttributeError, TypeError, ValueError, KeyError):
            photo_count = 0

        parts: list[str] = []
        if text:
            parts.append(text)
        if photo_count:
            parts.append(f"Photos: {photo_count}")
        if user:
            parts.append(f"— {user}")
        parts.append(canonicalize_status_url(url))

        out = "\n".join(parts).strip()
        return out or canonicalize_status_url(url)
    except (AttributeError, TypeError, ValueError, KeyError):
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
    user_text: str | None,
    tweet_caption: str | None,
    vl_notes: str | None,
) -> str:
    """Compose text-flow input for image tweets with caption + VL facts."""
    clean_user = (user_text or "").strip()
    clean_caption = (tweet_caption or "").strip()
    clean_vl = (vl_notes or "").strip()

    if not clean_caption and not clean_vl:
        return clean_user

    lines: list[str] = []
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
