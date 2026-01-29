"""
Shared helpers for reading small text attachments (e.g., .txt) from Discord messages.

This promotes the minimal, battle-tested logic already used by the !img path
to decode text bytes with size checks and light sanitization.
"""

from __future__ import annotations

import re
from typing import Optional

import discord


async def read_attachment_text(
    att: discord.Attachment, limit_bytes: int = 262_144
) -> Optional[str]:
    """Read and decode attachment text with size checks and sanitization.

    Mirrors the behavior used by the existing !img flow:
    - Rejects if the attachment is over the size limit
    - Tries common encodings (utf-8, utf-16, latin-1)
    - Strips NULs and collapses excessive whitespace
    - Returns None if unreadable or empty after cleanup
    """
    try:
        data = await att.read()
        if not isinstance(data, (bytes, bytearray)):
            return None
        if len(data) > int(limit_bytes):
            return None

        text = None
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                text = data.decode(encoding)
                break
            except Exception:
                continue
        if not text:
            return None

        # Light cleanup consistent with existing logic
        text = text.replace("\x00", "")
        text = re.sub(r"\s+", " ", text).strip()
        return text or None
    except Exception:
        # Keep silent; caller decides on fallback/logging
        return None
