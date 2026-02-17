"""Router modularization components (compatibility-first extraction layer)."""

from .compose import (
    compose_x_tweet_with_visual_facts,
    format_x_tweet_with_transcription,
)
from .runtime import RouterRuntimeCompat, load_router_runtime_compat

__all__ = [
    "RouterRuntimeCompat",
    "compose_x_tweet_with_visual_facts",
    "format_x_tweet_with_transcription",
    "load_router_runtime_compat",
]
