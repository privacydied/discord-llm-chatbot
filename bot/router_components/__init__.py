"""Router modularization components (compatibility-first extraction layer)."""

from .compose import (
    compose_x_tweet_with_visual_facts,
    format_x_tweet_with_transcription,
)
from .gating import is_reply_to_bot, mentions_bot, strip_leading_bot_mention
from .input_harvest import (
    extract_urls_loose,
    extract_urls_strict,
    has_explicit_media_intent,
    has_meaningful_text,
    is_text_attachment,
    strip_urls,
)
from .runtime import RouterRuntimeCompat, load_router_runtime_compat

__all__ = [
    "RouterRuntimeCompat",
    "compose_x_tweet_with_visual_facts",
    "extract_urls_loose",
    "extract_urls_strict",
    "format_x_tweet_with_transcription",
    "has_explicit_media_intent",
    "has_meaningful_text",
    "is_reply_to_bot",
    "is_text_attachment",
    "load_router_runtime_compat",
    "mentions_bot",
    "strip_leading_bot_mention",
    "strip_urls",
]
