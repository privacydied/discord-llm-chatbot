"""
Event handlers and utilities for the Discord bot.

This package exposes minimal utilities used by tests (`has_image_attachments`,
`get_image_urls`) and re-exports `setup_command_error_handler`.

For backward compatibility, we also lazily expose `BotEventHandler` from the
legacy sibling module `bot/events.py` via `__getattr__` so that
`from bot.events import BotEventHandler` works if needed.
"""
from .command_error_handler import setup_command_error_handler

# --- Minimal image attachment utilities (duplicated here to avoid import cycles) ---
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}

def _attachment_is_image(att) -> bool:
    try:
        ctype = getattr(att, "content_type", None)
        if isinstance(ctype, str) and ctype.lower().startswith("image/"):
            return True
        name = getattr(att, "filename", "") or ""
        import os
        _, ext = os.path.splitext(name.lower())
        return ext in IMAGE_EXTS
    except Exception:
        return False

def has_image_attachments(message) -> bool:
    atts = getattr(message, "attachments", None) or []
    return any(_attachment_is_image(att) for att in atts)

def get_image_urls(message) -> list:
    urls = []
    for att in getattr(message, "attachments", None) or []:
        if _attachment_is_image(att):
            url = getattr(att, "url", None)
            if url:
                urls.append(url)
    return urls

def __getattr__(name: str):
    """Lazy export for symbols that live in the legacy `bot/events.py` module.

    Avoids circular import between the package `bot.events` and the sibling
    module file `bot/events.py` that historically defined `BotEventHandler`.
    """
    if name == "BotEventHandler":
        import importlib.util
        from pathlib import Path
        events_py = Path(__file__).resolve().parent.parent / "events.py"
        spec = importlib.util.spec_from_file_location("bot.events_legacy", str(events_py))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, name)
        raise AttributeError(name)
    raise AttributeError(name)

__all__ = [
    'setup_command_error_handler',
    'has_image_attachments',
    'get_image_urls',
    # 'BotEventHandler' is provided lazily via __getattr__
]
