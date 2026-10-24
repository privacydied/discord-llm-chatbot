"""TTS state management for per-user and global preferences."""

import logging
import os

_TRUE = {"1", "true", "yes", "on"}
MAX_PENDING_VOICE_REPLIES = 256


class TTSState:
    def __init__(self) -> None:
        # Global TTS toggle. None = follow TTS_GLOBAL_DEFAULT in the environment
        # (read per check, so .env hot-reload applies and import order doesn't
        # matter); "!tts all on|off" sets an in-memory override until restart.
        self._global_override: bool | None = None

        # Per-user TTS preferences {user_id: enabled}
        self.user_preferences: dict[int, bool] = {}

        # Per-user one-time TTS flags
        self.one_time_tts: set[int] = set()

        # Messages whose reply the model asked to send as a voice note
        # (send_voice_note tool). Keyed by message so a failed run can't leak
        # voice into a later reply; bounded. [REH]
        self._voice_reply_messages: dict[int, None] = {}

        # Admin user IDs (cached for performance)
        self.admin_users: set[int] = set()

        # Statistics
        self.total_requests: int = 0
        self.cache_hits: int = 0

    @property
    def global_enabled(self) -> bool:
        """Every reply is spoken unless the user opted out (TTS_GLOBAL_DEFAULT / !tts all)."""
        if self._global_override is not None:
            return self._global_override
        return (os.getenv("TTS_GLOBAL_DEFAULT", "") or "").strip().strip('"').lower() in _TRUE

    @global_enabled.setter
    def global_enabled(self, enabled: bool) -> None:
        self._global_override = bool(enabled)

    def is_user_tts_enabled(self, user_id: int) -> bool:
        """Check if TTS is enabled for a specific user.

        An explicit per-user choice wins, so "!tts off" opts out of global voice.
        """
        if user_id in self.user_preferences:
            return self.user_preferences[user_id]
        return self.global_enabled

    def set_user_tts(self, user_id: int, enabled: bool) -> None:
        """Set TTS preference for a specific user."""
        self.user_preferences[user_id] = enabled
        logging.info(f"User {user_id} TTS: {'enabled' if enabled else 'disabled'}")

    def set_global_tts(self, enabled: bool) -> None:
        """Set global TTS preference."""
        self.global_enabled = enabled
        logging.info(f"Global TTS: {'enabled' if enabled else 'disabled'}")

    def is_admin(self, user_id: int) -> bool:
        """Check if user is admin (cached)."""
        return user_id in self.admin_users

    def add_admin(self, user_id: int) -> None:
        """Add user to admin cache."""
        self.admin_users.add(user_id)

    def remove_admin(self, user_id: int) -> None:
        """Remove user from admin cache."""
        self.admin_users.discard(user_id)

    def set_one_time_tts(self, user_id: int) -> None:
        """Set a one-time TTS flag for the user."""
        self.one_time_tts.add(user_id)

    def request_voice_reply(self, message_id: int) -> None:
        """The reply to ``message_id`` should be a voice note."""
        self._voice_reply_messages[int(message_id)] = None
        while len(self._voice_reply_messages) > MAX_PENDING_VOICE_REPLIES:
            self._voice_reply_messages.pop(next(iter(self._voice_reply_messages)))

    def consume_voice_reply(self, message_id: int | None) -> bool:
        if message_id is None:
            return False
        try:
            return self._voice_reply_messages.pop(int(message_id), False) is None
        except (TypeError, ValueError):
            return False

    def get_and_clear_one_time_tts(self, user_id: int) -> bool:
        """Check and clear the one-time TTS flag for the user."""
        if user_id in self.one_time_tts:
            self.one_time_tts.remove(user_id)
            return True
        return False

    def get_one_time_tts(self, user_id: int) -> bool:
        """Backward-compat alias for one-time TTS check.
        Note: This clears the flag if set, matching the semantics of
        `get_and_clear_one_time_tts`.
        """
        return self.get_and_clear_one_time_tts(user_id)

    def get_stats(self) -> dict:
        """Get TTS usage statistics."""
        return {
            "global_enabled": self.global_enabled,
            "users_with_tts": len([u for u, enabled in self.user_preferences.items() if enabled]),
            "total_users": len(self.user_preferences),
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
        }


# Global state instance
tts_state = TTSState()


def apply_tts_preference(action, message, state: TTSState | None = None) -> bool:
    """Mark ``action`` as a voice reply when the author's TTS preference says so.

    The single rule for every reply path (router text flow, tool loop, bot.py send):
    one-time "!speak" flag, then per-user choice, then the global default. Only
    text replies without files/audio are voiced, and the per-server "tts" feature
    switch is respected. Returns True when the action is (now) a voice reply. [CA]
    """
    meta = getattr(action, "meta", None)
    if meta is None:
        return False
    if meta.get("requires_tts"):
        return True
    if not (getattr(action, "content", "") or "").strip() or getattr(action, "files", None) or getattr(action, "audio_path", None):
        return False
    guild_id = getattr(getattr(message, "guild", None), "id", None)
    if guild_id is not None:
        from bot.server_features import is_server_feature_enabled

        if not is_server_feature_enabled(guild_id, "tts"):
            return False
    user_id = getattr(getattr(message, "author", None), "id", None)
    st = state or tts_state
    asked = st.consume_voice_reply(getattr(message, "id", None))  # send_voice_note tool
    if not asked and (user_id is None or not (st.get_and_clear_one_time_tts(user_id) or st.is_user_tts_enabled(user_id))):
        return False
    meta["requires_tts"] = True
    meta["include_transcript"] = os.getenv("TTS_INCLUDE_TRANSCRIPT", "true").lower() in _TRUE
    return True
