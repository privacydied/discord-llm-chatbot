"""
URL utilities for syndication content processing.
"""

from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse


def upgrade_pbs_to_orig(url: str) -> str:
    """
    Given a pbs.twimg.com URL (media/card/tweet_video_thumb), ensure it points to the highest
    available resolution by setting query param name=orig while preserving any existing 'format'.
    If the host isn't pbs.twimg.com, return url unchanged.

    Args:
        url: Input URL to potentially upgrade

    Returns:
        URL with name=orig parameter if it's a pbs.twimg.com URL, otherwise unchanged
    """
    try:
        p = urlparse(url)
        # Check for pbs.twimg.com (with or without port)
        hostname = (p.netloc or "").split(":")[0]
        if hostname != "pbs.twimg.com":
            return url

        # Validate that we have a proper scheme and path
        if not p.scheme or not p.path:
            return url

        # Handle legacy suffix ":size" (e.g., :small, :large, :orig)
        path = p.path
        if ":" in path:
            base, size = path.rsplit(":", 1)
            # Only strip known size tokens
            if size.lower() in ("small", "medium", "large", "orig"):
                path = base

        qs = dict(parse_qsl(p.query, keep_blank_values=True))
        # normalize 'name' to 'orig'
        qs["name"] = "orig"
        # preserve 'format' param if already present; do not invent one here
        new_qs = urlencode(qs, doseq=True)

        return urlunparse((p.scheme, p.netloc, path, p.params, new_qs, p.fragment))
    except Exception:
        # absolutely never break on URL upgrade; just return original
        return url


def pbs_base_key(url: str) -> str:
    """
    Return a canonical key for deduplicating pbs.twimg.com media assets regardless of
    query params or legacy :size suffix. Non-pbs URLs return the URL without query.
    """
    try:
        p = urlparse(url)
        hostname = (p.netloc or "").split(":")[0]
        path = p.path or ""
        if ":" in path:
            base, size = path.rsplit(":", 1)
            if size.lower() in ("small", "medium", "large", "orig"):
                path = base
        if hostname == "pbs.twimg.com":
            return f"pbs:{path}"
        # For other hosts, ignore query for dedup
        return f"{hostname}:{path}"
    except Exception:
        return url
