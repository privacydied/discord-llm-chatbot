"""
Compatibility shim for legacy test scripts in scripts/ that import from `main`.

Exports server memory helpers by delegating to the real bot modules.
"""
from __future__ import annotations

from pathlib import Path

from bot.config import load_config
from bot.memory.profiles import (
    get_server_profile as _get_server_profile,
    save_server_profile,
    default_server_profile,
)

# Expose SERVER_PROFILE_DIR as a Path, matching test expectations
SERVER_PROFILE_DIR: Path = load_config()["SERVER_PROFILE_DIR"]

# Persist-on-read wrapper to ensure the profile JSON exists immediately
def get_server_profile(guild_id: str, force_reload: bool = False) -> dict:
    profile = _get_server_profile(guild_id, force_reload=force_reload)
    profile_path = SERVER_PROFILE_DIR / f"{guild_id}.json"
    if not profile_path.exists():
        # Create directory and save profile to satisfy test expectations
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        save_server_profile(guild_id)
    return profile

__all__ = [
    "get_server_profile",
    "save_server_profile",
    "default_server_profile",
    "SERVER_PROFILE_DIR",
]
