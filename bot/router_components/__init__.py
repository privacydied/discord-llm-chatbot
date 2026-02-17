"""Router modularization components (compatibility-first extraction layer)."""

from .compose import (
    compose_x_tweet_with_visual_facts,
    format_x_tweet_with_transcription,
)
from .gating import is_reply_to_bot, mentions_bot, strip_leading_bot_mention
from .input_harvest import (
    all_attachments_are_text,
    extract_urls_loose,
    extract_urls_strict,
    has_explicit_media_intent,
    has_meaningful_text,
    is_text_attachment,
    strip_urls,
)
from .runtime import RouterRuntimeCompat, load_router_runtime_compat
from .x_routing import (
    canonicalize_twitter_status_url,
    collect_x_candidate_urls,
    extract_raw_urls_from_texts,
    extract_x_status_urls_from_text,
    filter_canonical_x_urls,
    is_tweet_media_url,
    is_twitter_media_cdn,
    is_twitter_thumbnail_url,
    is_twitter_url,
    normalize_x_url,
    parse_twitter_status_id,
    unwrap_x_media_url,
)

__all__ = [
    "RouterRuntimeCompat",
    "compose_x_tweet_with_visual_facts",
    "all_attachments_are_text",
    "canonicalize_twitter_status_url",
    "extract_urls_loose",
    "extract_urls_strict",
    "extract_raw_urls_from_texts",
    "extract_x_status_urls_from_text",
    "filter_canonical_x_urls",
    "format_x_tweet_with_transcription",
    "has_explicit_media_intent",
    "has_meaningful_text",
    "is_reply_to_bot",
    "is_text_attachment",
    "is_tweet_media_url",
    "is_twitter_media_cdn",
    "is_twitter_thumbnail_url",
    "is_twitter_url",
    "load_router_runtime_compat",
    "mentions_bot",
    "normalize_x_url",
    "parse_twitter_status_id",
    "strip_leading_bot_mention",
    "strip_urls",
    "collect_x_candidate_urls",
    "unwrap_x_media_url",
]
