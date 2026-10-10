"""Link-embed fixing: reply to x.com / instagram / bsky / ... links with
embed-friendly proxy URLs (Python port of SaucyBot's rewrite sites)."""

from .handler import FEATURE_NAME, EmbedFixSettings, maybe_fix_embeds, might_have_fixable_link
from .sites import SITE_IDS, SITES, EmbedFix, EmbedSite, find_fixes, is_ignored, rewrite_url

__all__ = [
    "FEATURE_NAME",
    "SITES",
    "SITE_IDS",
    "EmbedFix",
    "EmbedFixSettings",
    "EmbedSite",
    "find_fixes",
    "is_ignored",
    "maybe_fix_embeds",
    "might_have_fixable_link",
    "rewrite_url",
]
