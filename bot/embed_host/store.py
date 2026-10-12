"""On-disk store for hosted embed pages: one directory per token.

Layout: ``<root>/<token>/post.json`` plus the media files it names. Tokens
are random (``secrets.token_urlsafe``), so a page exists only if the bot
created it. Old posts are pruned by age and by a total-size cap.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bot.utils.logging import get_logger

logger = get_logger(__name__)

TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,32}$")
MEDIA_NAME_RE = re.compile(r"^[a-z0-9_]{1,16}\.(?:mp4|mov|webm|jpg|jpeg|png|webp)$")
_MIME = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "webm": "video/webm",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}
_META_FILE = "post.json"


def mime_for(name: str) -> str:
    return _MIME.get(name.rsplit(".", 1)[-1].lower(), "application/octet-stream")


# ---- media dimensions (Discord sizes the player from og:video:width/height) ----


def _mp4_boxes(data: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack(">I4s", data[pos : pos + 8])
        header = 8
        if size == 1 and pos + 16 <= end:
            size, header = struct.unpack(">Q", data[pos + 8 : pos + 16])[0], 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            return
        yield kind, pos + header, pos + size
        pos += size


def mp4_dimensions(data: bytes) -> tuple[int, int] | None:
    """Width/height of the first visual track from its ``tkhd`` box."""
    for kind, start, end in _mp4_boxes(data, 0, len(data)):
        if kind != b"moov":
            continue
        for trak, t_start, t_end in _mp4_boxes(data, start, end):
            if trak != b"trak":
                continue
            for box, b_start, b_end in _mp4_boxes(data, t_start, t_end):
                if box == b"tkhd" and b_end - b_start >= 8:
                    width, height = struct.unpack(">II", data[b_end - 8 : b_end])
                    if width >> 16 and height >> 16:
                        return width >> 16, height >> 16
    return None


def image_dimensions(data: bytes) -> tuple[int, int] | None:
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data[:2] != b"\xff\xd8":
        return None
    pos = 2
    while pos + 9 < len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            height, width = struct.unpack(">HH", data[pos + 5 : pos + 9])
            return width, height
        pos += 2 + struct.unpack(">H", data[pos + 2 : pos + 4])[0]
    return None


def dimensions(name: str, data: bytes) -> tuple[int, int] | None:
    try:
        return mp4_dimensions(data) if mime_for(name).startswith("video/") else image_dimensions(data)
    except (struct.error, IndexError):
        return None


# ---- records ---------------------------------------------------------------


@dataclass(slots=True)
class StoredMedia:
    name: str
    mime: str
    width: int | None = None
    height: int | None = None

    @property
    def is_video(self) -> bool:
        return self.mime.startswith("video/")


@dataclass(slots=True)
class StoredPost:
    token: str
    source_url: str
    site_name: str
    colour: str = "#E1306C"
    author_name: str | None = None
    author_handle: str | None = None
    author_url: str | None = None
    avatar_url: str | None = None
    text: str | None = None
    stats: str | None = None
    published: str | None = None
    media: list[StoredMedia] = field(default_factory=list)
    thumbnail: StoredMedia | None = None
    avatar: StoredMedia | None = None
    created: float = field(default_factory=time.time)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> StoredPost:
        media = [StoredMedia(**m) for m in raw.pop("media", [])]
        thumb, avatar = raw.pop("thumbnail", None), raw.pop("avatar", None)
        return cls(
            **raw,
            media=media,
            thumbnail=StoredMedia(**thumb) if thumb else None,
            avatar=StoredMedia(**avatar) if avatar else None,
        )


@dataclass(frozen=True, slots=True)
class NewMedia:
    name: str
    data: bytes


class EmbedStore:
    def __init__(self, root: str | Path, *, ttl_days: float = 14.0, max_total_bytes: int = 5 * 1024**3) -> None:
        self.root = Path(root)
        self.ttl_s = max(0.0, float(ttl_days)) * 86400
        self.max_total_bytes = int(max_total_bytes)

    def _dir(self, token: str) -> Path | None:
        return self.root / token if TOKEN_RE.match(token or "") else None

    def put(
        self,
        post: StoredPost,
        media: list[NewMedia],
        thumbnail: NewMedia | None = None,
        avatar: NewMedia | None = None,
    ) -> StoredPost:
        """Write *media* and *post*; assigns ``post.token``. Blocking (call via to_thread)."""
        post.token = secrets.token_urlsafe(16)
        target = self.root / post.token
        target.mkdir(parents=True, exist_ok=False)
        post.media = [self._write(target, f"{index}", item) for index, item in enumerate(media, start=1)]
        post.thumbnail = self._write(target, "thumb", thumbnail) if thumbnail else None
        post.avatar = self._write(target, "avatar", avatar) if avatar else None
        (target / _META_FILE).write_text(json.dumps(asdict(post)), encoding="utf-8")
        return post

    @staticmethod
    def _write(target: Path, stem: str, item: NewMedia) -> StoredMedia:
        ext = item.name.rsplit(".", 1)[-1].lower()
        ext = ext if ext in _MIME else "mp4"
        name = f"{stem}.{ext}"
        (target / name).write_bytes(item.data)
        width, height = dimensions(name, item.data) or (None, None)
        return StoredMedia(name=name, mime=_MIME[ext], width=width, height=height)

    def get(self, token: str) -> StoredPost | None:
        folder = self._dir(token)
        meta = folder / _META_FILE if folder else None
        if meta is None or not meta.is_file():
            return None
        try:
            post = StoredPost.from_json(json.loads(meta.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(f"embed_host.bad_record | token={token} error={exc}")
            return None
        if self.ttl_s and time.time() - post.created > self.ttl_s:
            return None
        return post

    def media_path(self, token: str, name: str) -> Path | None:
        folder = self._dir(token)
        if folder is None or not MEDIA_NAME_RE.match(name or ""):
            return None
        path = folder / name
        return path if path.is_file() else None

    def prune(self) -> int:
        """Delete expired posts, then the oldest until under the size cap. Blocking."""
        if not self.root.is_dir():
            return 0
        entries = []
        for folder in self.root.iterdir():
            if folder.is_dir() and TOKEN_RE.match(folder.name):
                size = sum(f.stat().st_size for f in folder.iterdir() if f.is_file())
                entries.append((folder.stat().st_mtime, size, folder))
        entries.sort()
        now, total, removed = time.time(), sum(e[1] for e in entries), 0
        for mtime, size, folder in entries:
            expired = self.ttl_s and now - mtime > self.ttl_s
            if not expired and total <= self.max_total_bytes:
                continue
            shutil.rmtree(folder, ignore_errors=True)
            total -= size
            removed += 1
        return removed
