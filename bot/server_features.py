from __future__ import annotations

from typing import Any, Dict

from .memory.profiles import get_server_profile, save_server_profile

FEATURE_ALIASES = {
    "stt": "stt",
    "speech_to_text": "stt",
    "tts": "tts",
    "text_to_speech": "tts",
    "vision": "vision",
    "image": "image_generation",
    "images": "image_generation",
    "image_generation": "image_generation",
    "img": "image_generation",
    "web": "web_extraction",
    "web_extraction": "web_extraction",
    "extract": "web_extraction",
    "url": "web_extraction",
    "x": "x_twitter_extraction",
    "twitter": "x_twitter_extraction",
    "x_twitter": "x_twitter_extraction",
    "x_twitter_extraction": "x_twitter_extraction",
    "rag": "rag",
    "knowledge": "rag",
}

FEATURE_DEFAULTS: Dict[str, bool] = {
    "stt": True,
    "tts": True,
    "vision": True,
    "image_generation": True,
    "web_extraction": True,
    "x_twitter_extraction": True,
    "rag": True,
}


def normalize_feature_name(name: str) -> str:
    key = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    return FEATURE_ALIASES.get(key, key)


def get_server_feature_toggles(guild_id: int | str) -> Dict[str, bool]:
    profile = get_server_profile(guild_id)
    custom_data = dict(profile.get("custom_data") or {})
    toggles = dict(FEATURE_DEFAULTS)
    existing = custom_data.get("feature_toggles") or {}
    for raw_name, value in existing.items():
        normalized = normalize_feature_name(str(raw_name))
        if normalized in toggles:
            toggles[normalized] = bool(value)
    return toggles


def set_server_feature_toggle(guild_id: int | str, feature: str, enabled: bool) -> Dict[str, bool]:
    normalized = normalize_feature_name(feature)
    if normalized not in FEATURE_DEFAULTS:
        raise KeyError(f"Unknown feature toggle: {feature}")

    profile = get_server_profile(guild_id)
    custom_data = dict(profile.get("custom_data") or {})
    toggles = dict(custom_data.get("feature_toggles") or {})
    toggles[normalized] = bool(enabled)
    custom_data["feature_toggles"] = toggles
    profile["custom_data"] = custom_data
    save_server_profile(profile)
    return get_server_feature_toggles(guild_id)


def is_server_feature_enabled(guild_id: int | str | None, feature: str) -> bool:
    normalized = normalize_feature_name(feature)
    if normalized not in FEATURE_DEFAULTS:
        return True
    if guild_id is None:
        return FEATURE_DEFAULTS[normalized]
    return get_server_feature_toggles(guild_id).get(normalized, FEATURE_DEFAULTS[normalized])


def feature_status_label(enabled: bool) -> str:
    return "enabled" if enabled else "disabled"


def feature_status_emoji(enabled: bool) -> str:
    return "✅" if enabled else "🚫"
