"""Compatibility runtime config loader for router settings.

This module is phase-1 scaffolding for router decomposition. It preserves
existing coercion/default behavior while centralizing read-once settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

_X_SYNDICATION_ACCEPT_DOMAINS_DEFAULT = "pbs.twimg.com,video.twimg.com,fxtwitter.com,vxtwitter.com"
_X_SYNDICATION_ACCEPT_DOMAINS_FALLBACK = {
    "pbs.twimg.com",
    "pbs-0.twimg.com",
    "pbs-1.twimg.com",
    "pbs-2.twimg.com",
    "pbs-3.twimg.com",
    "video.twimg.com",
    "fxtwitter.com",
    "vxtwitter.com",
}


def _cfg_float(config: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(config.get(key, default))
    except (ValueError, TypeError, AttributeError):
        return float(default)


def _cfg_int(config: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(config.get(key, default))
    except (ValueError, TypeError, AttributeError):
        return int(default)


def _cfg_bool(config: Mapping[str, Any], key: str, default: bool) -> bool:
    try:
        return bool(config.get(key, default))
    except (ValueError, TypeError, AttributeError):
        return bool(default)


def _cfg_str(config: Mapping[str, Any], key: str, default: str) -> str:
    try:
        return str(config.get(key, default)).strip()
    except (ValueError, TypeError, AttributeError):
        return default


def _cfg_domain_set(config: Mapping[str, Any]) -> set[str]:
    try:
        domains = (
            config.get(
                "X_SYNDICATION_ACCEPT_DOMAINS",
                _X_SYNDICATION_ACCEPT_DOMAINS_DEFAULT,
            )
            or ""
        )
        return {d.strip().lower() for d in str(domains).split(",") if d.strip()}
    except (ValueError, TypeError, AttributeError):
        return set(_X_SYNDICATION_ACCEPT_DOMAINS_FALLBACK)


@dataclass(frozen=True)
class RouterRuntimeCompat:
    syn_ttl_s: float
    x_syn_probe_enabled: bool
    x_syn_order: str
    x_syn_timeout_s: float
    x_syn_max_images: int
    x_syn_accept_domains: set[str]
    x_early_resolve_enabled: bool


def load_router_runtime_compat(config: Mapping[str, Any]) -> RouterRuntimeCompat:
    """Load router runtime settings with legacy-compatible defaults/coercion."""
    return RouterRuntimeCompat(
        syn_ttl_s=_cfg_float(config, "X_SYNDICATION_TTL_S", 900.0),
        x_syn_probe_enabled=_cfg_bool(config, "X_SYNDICATION_PROBE_ENABLED", True),
        x_syn_order=_cfg_str(config, "X_SYNDICATION_ORDER", "yt_dlp,html,api"),
        x_syn_timeout_s=_cfg_float(config, "X_SYNDICATION_TIMEOUT_S", 3.0),
        x_syn_max_images=_cfg_int(config, "X_SYNDICATION_MAX_IMAGES", 4),
        x_syn_accept_domains=_cfg_domain_set(config),
        x_early_resolve_enabled=_cfg_bool(config, "X_EARLY_RESOLVE_ENABLED", True),
    )
