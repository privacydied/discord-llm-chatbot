"""What the bot's own voice notes say, keyed by Discord message ID. [CA][REH]

A native Discord voice message can't carry text, so a bot voice note has empty
``content``. Without this registry a reply to one would either see nothing (the
reply-chain context and the history turn read ``message.content``) or, worse,
harvest the bot's own ``voice-message.ogg`` and run it through STT. The spoken
text is recorded when the note is sent and read back wherever message text is
needed. Bounded (``MAX_ENTRIES``) and persisted to a small JSON file so replies
still resolve after a restart.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_ENTRIES = 2000
MAX_TEXT_CHARS = 4000
DEFAULT_FILE = "data/voice_transcripts.json"
AUDIO_EXTENSIONS = (".ogg", ".opus", ".mp3", ".wav", ".m4a")
_VOICE_FLAG = 1 << 13  # MessageFlags.voice

_lock = threading.Lock()
_cache: OrderedDict[str, str] | None = None


def _path() -> Path:
    return Path(os.getenv("VOICE_TRANSCRIPTS_FILE") or DEFAULT_FILE)


def _load() -> OrderedDict[str, str]:
    global _cache
    if _cache is None:
        _cache = OrderedDict()
        try:
            data = json.loads(_path().read_text(encoding="utf-8"))
            for key, text in (data or {}).items():
                _cache[str(key)] = str(text)
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            logger.warning(f"voice.transcripts.load_failed | {exc}")
    return _cache


def remember(message_id: Any, text: str | None) -> None:
    """Record what voice note ``message_id`` says (sync; call via asyncio.to_thread)."""
    body = (text or "").strip()[:MAX_TEXT_CHARS]
    if message_id is None or not body:
        return
    with _lock:
        cache = _load()
        cache[str(message_id)] = body
        cache.move_to_end(str(message_id))
        while len(cache) > MAX_ENTRIES:
            cache.popitem(last=False)
        try:
            path = _path()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            logger.warning(f"voice.transcripts.save_failed | {exc}")


def lookup(message_id: Any) -> str | None:
    if message_id is None:
        return None
    with _lock:
        return _load().get(str(message_id))


def message_text(message: Any) -> str:
    """``message.content``, or the transcript when it is one of our voice notes."""
    content = getattr(message, "content", "") or ""
    if content.strip():
        return content
    return lookup(getattr(message, "id", None)) or content


def is_own_voice_note(message: Any, bot_user_id: Any) -> bool:
    """True for audio the bot itself posted (voice notes or audio-file fallbacks)."""
    try:
        author_id = getattr(getattr(message, "author", None), "id", None)
        if bot_user_id is None or author_id is None or int(author_id) != int(bot_user_id):
            return False
    except (TypeError, ValueError):
        return False
    if lookup(getattr(message, "id", None)) is not None:
        return True
    flags = getattr(getattr(message, "flags", None), "value", 0)
    if isinstance(flags, int) and flags & _VOICE_FLAG:
        return True
    for att in getattr(message, "attachments", None) or []:
        ctype = str(getattr(att, "content_type", "") or "").lower()
        name = str(getattr(att, "filename", "") or "").lower()
        if ctype.startswith("audio/") or name.endswith(AUDIO_EXTENSIONS):
            return True
    return False


def reset_cache_for_tests() -> None:
    global _cache
    with _lock:
        _cache = None
