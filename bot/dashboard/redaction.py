"""Content safety and redaction utilities for the Discord-like dashboard.

Provides:
- redact_secrets(): Strips known secret patterns from text
- make_preview(): Truncated + redacted content preview
- sanitize_for_html(): Safe HTML escaping
- contains_mention_warning(): Detects @everyone/@here/role mentions
- ContentSecurityPolicy: Builder class for CSP headers
"""

from __future__ import annotations

import html
import re

from bot.utils.logging import get_logger, redact_sensitive_values

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Secret patterns — regex-based redaction
# ---------------------------------------------------------------------------

# Discord bot/user token:  [A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{27,}
_DISCORD_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{27,}")

# OpenAI / OpenRouter keys: sk-... or sk-or-...
_OPENAI_KEY_RE = re.compile(r"sk-or-v1-[A-Za-z0-9\-]{20,}|sk-[A-Za-z0-9\-]{20,}")

# Generic API keys (hex or base64-like, >= 32 chars)
_GENERIC_API_KEY_RE = re.compile(
    r"(?P<prefix>api[_-]?key|apikey|api[_-]?secret|api_secret)[=:]\s*"
    r"(?P<key>[A-Za-z0-9_\-]{16,})",
    re.IGNORECASE,
)

# Bearer / Authorization header value
_BEARER_TOKEN_RE = re.compile(
    r"(?:Bearer|bearer)\s+[A-Za-z0-9_\-\.]{8,}",
)

# Authorization header
_AUTHORIZATION_HEADER_RE = re.compile(
    r"(?P<header>Authorization|authorization)[=:]\s*[A-Za-z0-9_\-\.]{8,}",
)

# Passwords in common formats
_PASSWORD_RE = re.compile(
    r"(?P<prefix>password|passwd|pwd)[=:]\s*(?P<value>[^\s&]{4,})",
    re.IGNORECASE,
)

# Private key markers
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----")

_REDACTED = "[REDACTED]"


def redact_secrets(text: str) -> str:
    """Redact known secret patterns from arbitrary text.

    Operates in two layers:
    1. Environment variable value replacement (via redact_sensitive_values from bot.utils.logging)
    2. Regex-based pattern matching for common credential formats

    Returns the redacted text.
    """
    if not text:
        return text

    # Layer 1: Replace known env-var values
    result = redact_sensitive_values(text)

    # Layer 2: Regex patterns
    result = _DISCORD_TOKEN_RE.sub(_REDACTED, result)
    result = _OPENAI_KEY_RE.sub(_REDACTED, result)
    result = _PRIVATE_KEY_RE.sub(_REDACTED, result)

    # Replace with prefix preservation where possible
    result = _BEARER_TOKEN_RE.sub(lambda m: "Bearer " + _REDACTED, result)

    def _replace_generic_key(m: re.Match) -> str:
        return f"{m.group('prefix')}={_REDACTED}"

    result = _GENERIC_API_KEY_RE.sub(_replace_generic_key, result)

    def _replace_auth_header(m: re.Match) -> str:
        return f"{m.group('header')}={_REDACTED}"

    result = _AUTHORIZATION_HEADER_RE.sub(_replace_auth_header, result)

    def _replace_password(m: re.Match) -> str:
        return f"{m.group('prefix')}={_REDACTED}"

    return _PASSWORD_RE.sub(_replace_password, result)


def make_preview(text: str, max_chars: int = 200) -> str:
    """Truncate and redact content for preview display.

    Args:
        text: The raw content string.
        max_chars: Maximum preview length (default: 200).

    Returns:
        Redacted and possibly truncated string.

    """
    redacted = redact_secrets(text or "")
    if len(redacted) > max_chars:
        return redacted[:max_chars] + "..."
    return redacted


def sanitize_for_html(text: str) -> str:
    """Escape text for safe embedding in HTML.

    Uses html.escape() for safe encoding of <, >, &, ", and ' characters.
    Also applies secret redaction before escaping.
    """
    safe = redact_secrets(text or "")
    return html.escape(safe, quote=True)


def contains_mention_warning(text: str) -> str | None:
    """Check if text contains @everyone, @here, or mass role mentions.

    Returns a warning string if detected, or None if safe.
    """
    if not text:
        return None

    checks = [
        ("@everyone", "@everyone mention detected"),
        ("@here", "@here mention detected"),
    ]

    # Role mention pattern: <@&123456789012345678>
    role_mention_re = re.compile(r"<@&\d{17,20}>")

    for pattern, warning in checks:
        if pattern in text:
            return warning

    if role_mention_re.search(text):
        return "Role mention detected"

    return None


# ---------------------------------------------------------------------------
# ContentSecurityPolicy builder
# ---------------------------------------------------------------------------


class ContentSecurityPolicy:
    r"""Builder for Content-Security-Policy HTTP headers.

    Provides a fluent interface for constructing CSP directives:

        csp = ContentSecurityPolicy() \\
            .default_src(\"'self'\") \\
            .script_src(\"'self'\", \"'unsafe-inline'\") \\
            .style_src(\"'self'\", \"'unsafe-inline'\") \\
            .build()

    Default policy is restrictive: default-src 'self'; no inline scripts.
    """

    def __init__(self) -> None:
        self._directives: dict[str, set[str]] = {
            "default-src": {"'self'"},
            "script-src": {"'self'"},
            "style-src": {"'self'"},
            "img-src": {"'self'", "data:", "https://cdn.discordapp.com"},
            "font-src": {"'self'", "data:"},
            "connect-src": {"'self'"},
            "frame-ancestors": {"'none'"},
            "form-action": {"'self'"},
            "base-uri": {"'self'"},
            "object-src": {"'none'"},
        }

    def default_src(self, *sources: str) -> ContentSecurityPolicy:
        """Set default-src directive (replaces current)."""
        self._directives["default-src"] = set(sources)
        return self

    def script_src(self, *sources: str) -> ContentSecurityPolicy:
        """Set script-src directive (replaces current)."""
        self._directives["script-src"] = set(sources)
        return self

    def style_src(self, *sources: str) -> ContentSecurityPolicy:
        """Set style-src directive (replaces current)."""
        self._directives["style-src"] = set(sources)
        return self

    def img_src(self, *sources: str) -> ContentSecurityPolicy:
        """Set img-src directive (replaces current)."""
        self._directives["img-src"] = set(sources)
        return self

    def font_src(self, *sources: str) -> ContentSecurityPolicy:
        """Set font-src directive (replaces current)."""
        self._directives["font-src"] = set(sources)
        return self

    def connect_src(self, *sources: str) -> ContentSecurityPolicy:
        """Set connect-src directive (replaces current)."""
        self._directives["connect-src"] = set(sources)
        return self

    def frame_ancestors(self, *sources: str) -> ContentSecurityPolicy:
        """Set frame-ancestors directive (replaces current)."""
        self._directives["frame-ancestors"] = set(sources)
        return self

    def form_action(self, *sources: str) -> ContentSecurityPolicy:
        """Set form-action directive (replaces current)."""
        self._directives["form-action"] = set(sources)
        return self

    def add_directive(self, name: str, *sources: str) -> ContentSecurityPolicy:
        """Add or replace a custom directive."""
        self._directives[name] = set(sources)
        return self

    def allow_inline_scripts(self) -> ContentSecurityPolicy:
        """Convenience: add 'unsafe-inline' to script-src."""
        self._directives.setdefault("script-src", set()).add("'unsafe-inline'")
        return self

    def allow_inline_styles(self) -> ContentSecurityPolicy:
        """Convenience: add 'unsafe-inline' to style-src."""
        self._directives.setdefault("style-src", set()).add("'unsafe-inline'")
        return self

    def build(self) -> str:
        """Build the CSP string. Order is deterministic: key-sorted directives."""
        parts: list[str] = []
        for name in sorted(self._directives.keys()):
            sources = " ".join(sorted(self._directives[name]))
            parts.append(f"{name} {sources}")
        return "; ".join(parts)

    def build_header_dict(self) -> dict[str, str]:
        """Return a dict with Content-Security-Policy as the key."""
        return {"Content-Security-Policy": self.build()}

    @classmethod
    def permissive(cls) -> ContentSecurityPolicy:
        """Create a permissive CSP suitable for development or dashboards
        that use inline scripts and connect to Discord CDN.

        Equivalent to:
            default-src 'self'; script-src 'self' 'unsafe-inline';
            style-src 'self' 'unsafe-inline'; img-src 'self' data: https:;
            font-src 'self' data:; connect-src 'self';
            frame-ancestors 'none'; form-action 'self';
            base-uri 'self'; object-src 'none'
        """
        csp = cls()
        csp._directives["script-src"].add("'unsafe-inline'")
        csp._directives["style-src"].add("'unsafe-inline'")
        csp._directives["img-src"].add("https:")
        return csp
