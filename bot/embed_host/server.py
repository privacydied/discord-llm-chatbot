"""Public aiohttp server for bot-hosted embed pages.

Routes (all read-only, all keyed by a stored token):
  GET /p/{token}                         OG page for crawlers; humans get a 302 to the source
  GET /users/{token}                     ActivityPub actor (author name, @handle, avatar)
  GET /users/{token}/statuses/{token}    ActivityPub Note (text, stats, media) - Discord
                                         builds its rich card (author + avatar + video) from it
  GET /m/{token}/{name}                  stored media (range requests via FileResponse)
  GET /healthz                           readiness nonce, used to verify the public URL

The bot only links to a hosted page after it has fetched ``/healthz`` through
the public URL and got its own nonce back, so a missing reverse-proxy rule or
DNS record means "keep the old behaviour", never "post dead links".
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import secrets
from dataclasses import dataclass
from typing import Any

import aiohttp
from aiohttp import web

from bot.utils.logging import get_logger

from .store import EmbedStore, StoredMedia, StoredPost

logger = get_logger(__name__)

_CRAWLER_MARKERS = ("discordbot", "twitterbot", "telegrambot", "facebookexternalhit", "slackbot", "whatsapp", "embedly")
_AP_TYPE = "application/activity+json"
_PUBLIC = "https://www.w3.org/ns/activitystreams#Public"
_MAINTENANCE_INTERVAL_S = 600.0
_CHECK_TIMEOUT_S = 10.0

_host: EmbedHost | None = None


@dataclass(frozen=True, slots=True)
class EmbedHostSettings:
    enabled: bool = False
    bind: str = "127.0.0.1"
    port: int = 8012
    public_url: str = ""
    root: str = "./data/embed_host"
    ttl_days: float = 14.0
    max_total_bytes: int = 5 * 1024**3
    max_file_bytes: int = 100 * 1024**2

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> EmbedHostSettings:
        return cls(
            enabled=bool(cfg.get("EMBED_HOST_ENABLED", False)),
            bind=str(cfg.get("EMBED_HOST_BIND") or "127.0.0.1"),
            port=int(cfg.get("EMBED_HOST_PORT") or 8012),
            public_url=str(cfg.get("EMBED_HOST_PUBLIC_URL") or "").rstrip("/"),
            root=str(cfg.get("EMBED_HOST_DIR") or "./data/embed_host"),
            ttl_days=float(cfg.get("EMBED_HOST_TTL_DAYS") or 14),
            max_total_bytes=int(float(cfg.get("EMBED_HOST_MAX_TOTAL_GB") or 5) * 1024**3),
            max_file_bytes=int(float(cfg.get("EMBED_HOST_MAX_FILE_MB") or 100) * 1024**2),
        )


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _is_crawler(request: web.Request) -> bool:
    agent = request.headers.get("User-Agent", "").lower()
    return any(marker in agent for marker in _CRAWLER_MARKERS)


class EmbedHost:
    def __init__(self, settings: EmbedHostSettings) -> None:
        self.settings = settings
        self.store = EmbedStore(settings.root, ttl_days=settings.ttl_days, max_total_bytes=settings.max_total_bytes)
        self.nonce = secrets.token_urlsafe(16)
        self.ready = False
        self._runner: web.AppRunner | None = None
        self._task: asyncio.Task | None = None

    # ---- urls -------------------------------------------------------------

    @property
    def public_url(self) -> str:
        return self.settings.public_url

    def page_url(self, token: str) -> str:
        return f"{self.public_url}/p/{token}"

    def _media_url(self, token: str, media: StoredMedia) -> str:
        return f"{self.public_url}/m/{token}/{media.name}"

    def _actor_url(self, token: str) -> str:
        return f"{self.public_url}/users/{token}"

    def _status_url(self, token: str) -> str:
        return f"{self._actor_url(token)}/statuses/{token}"

    def _avatar_url(self, post: StoredPost) -> str | None:
        return self._media_url(post.token, post.avatar) if post.avatar else post.avatar_url

    def owns(self, url: str) -> bool:
        return bool(self.public_url) and url.startswith(f"{self.public_url}/p/")

    # ---- lifecycle --------------------------------------------------------

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/healthz", self._healthz)
        app.router.add_get("/p/{token}", self._page)
        app.router.add_get("/users/{token}", self._actor)
        app.router.add_get("/users/{token}/statuses/{status}", self._status)
        app.router.add_get("/m/{token}/{name}", self._media)
        return app

    async def start(self) -> None:
        self._runner = web.AppRunner(self.app(), access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.settings.bind, self.settings.port).start()
        logger.info(f"embed_host.started | bind={self.settings.bind}:{self.settings.port} public={self.public_url or '-'}")
        self._task = asyncio.create_task(self._maintenance_loop(), name="embed_host_maintenance")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._runner:
            await self._runner.cleanup()
        self.ready = False

    async def _maintenance_loop(self) -> None:
        while True:
            await self.check_public()
            with contextlib.suppress(OSError):
                removed = await asyncio.to_thread(self.store.prune)
                if removed:
                    logger.info(f"embed_host.pruned | posts={removed}")
            await asyncio.sleep(_MAINTENANCE_INTERVAL_S)

    async def check_public(self) -> bool:
        """Ready only if the public URL reaches *this* server (nonce round-trip)."""
        ok = False
        if self.public_url.startswith("https://"):
            timeout = aiohttp.ClientTimeout(total=_CHECK_TIMEOUT_S)
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"{self.public_url}/healthz") as resp:
                        ok = resp.status == 200 and (await resp.text()).strip() == self.nonce
            except (aiohttp.ClientError, TimeoutError) as exc:
                logger.debug(f"embed_host.check_failed | error={exc}")
        if ok != self.ready:
            logger.info(f"embed_host.ready | ready={ok} public={self.public_url or '-'}")
        self.ready = ok
        return ok

    # ---- handlers ---------------------------------------------------------

    async def _healthz(self, _request: web.Request) -> web.Response:
        return web.Response(text=self.nonce, headers={"Cache-Control": "no-store"})

    async def _load(self, token: str) -> StoredPost:
        post = await asyncio.to_thread(self.store.get, token)
        if post is None:
            raise web.HTTPNotFound()
        return post

    async def _page(self, request: web.Request) -> web.Response:
        post = await self._load(request.match_info["token"])
        if not _is_crawler(request):
            raise web.HTTPFound(post.source_url)
        return web.Response(text=self.render_page(post), content_type="text/html", charset="utf-8")

    async def _actor(self, request: web.Request) -> web.Response:
        post = await self._load(request.match_info["token"])
        return web.json_response(self.actor_json(post), content_type=_AP_TYPE)

    async def _status(self, request: web.Request) -> web.Response:
        token = request.match_info["token"]
        if request.match_info["status"] != token:
            raise web.HTTPNotFound()
        post = await self._load(token)
        return web.json_response(self.note_json(post), content_type=_AP_TYPE)

    async def _media(self, request: web.Request) -> web.StreamResponse:
        path = await asyncio.to_thread(self.store.media_path, request.match_info["token"], request.match_info["name"])
        if path is None:
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})

    # ---- rendering --------------------------------------------------------

    def _title(self, post: StoredPost) -> str:
        if post.author_name and post.author_handle and post.author_name != post.author_handle:
            return f"{post.author_name} (@{post.author_handle})"
        if post.author_handle:
            return f"@{post.author_handle}"
        return post.author_name or post.site_name

    def render_page(self, post: StoredPost) -> str:
        token = post.token
        tags: list[tuple[str, str, Any]] = [
            ("property", "og:site_name", post.site_name),
            ("property", "og:url", post.source_url),
            ("property", "og:title", self._title(post)),
            ("name", "twitter:title", self._title(post)),
            ("name", "theme-color", post.colour),
            ("property", "og:type", "article"),
        ]
        if post.text:
            tags += [("property", "og:description", post.text), ("name", "twitter:description", post.text)]
        if post.published:
            tags.append(("property", "article:published_time", post.published))
        if post.author_url:
            tags.append(("property", "article:author", post.author_url))
        video = next((m for m in post.media if m.is_video), None)
        image = post.thumbnail or next((m for m in post.media if not m.is_video), None)
        if image is not None:
            tags += [("property", "og:image", self._media_url(token, image)), ("name", "twitter:image", self._media_url(token, image))]
            tags += [("property", f"og:image:{k}", v) for k, v in (("width", image.width), ("height", image.height)) if v]
        if video is not None:
            url = self._media_url(token, video)
            tags += [("property", "og:video", url), ("property", "og:video:secure_url", url), ("property", "og:video:type", video.mime)]
            tags += [("property", f"og:video:{k}", v) for k, v in (("width", video.width), ("height", video.height)) if v]
        tags.append(("name", "twitter:card", "summary_large_image"))
        lines = [f'<meta {kind}="{_esc(key)}" content="{_esc(value)}">' for kind, key, value in tags]
        lines.append(f'<link rel="alternate" type="{_AP_TYPE}" href="{_esc(self._status_url(token))}">')
        avatar = self._avatar_url(post)
        if avatar:
            lines.append(f'<link rel="apple-touch-icon" href="{_esc(avatar)}">')
        head = "\n".join(lines)
        return f'<!DOCTYPE html>\n<html><head>\n<meta charset="utf-8">\n{head}\n</head><body></body></html>\n'

    def actor_json(self, post: StoredPost) -> dict[str, Any]:
        actor = self._actor_url(post.token)
        body: dict[str, Any] = {
            "@context": "https://www.w3.org/ns/activitystreams",
            "id": actor,
            "type": "Person",
            "name": post.author_name or post.author_handle or post.site_name,
            "preferredUsername": post.author_handle or post.author_name or post.site_name,
            "url": post.author_url or post.source_url,
            "inbox": f"{actor}/inbox",
            "outbox": f"{actor}/outbox",
        }
        avatar = self._avatar_url(post)
        if avatar:
            mime = post.avatar.mime if post.avatar else "image/jpeg"
            body["icon"] = {"type": "Image", "mediaType": mime, "url": avatar}
        return body

    def note_json(self, post: StoredPost) -> dict[str, Any]:
        parts = []
        if post.stats:
            parts.append(f"<p><b>{_esc(post.stats)}</b></p>")
        if post.text:
            parts.append("<p>" + _esc(post.text).replace("\n", "<br>") + "</p>")
        attachments = [{k: v for k, v in {"type": "Document", "mediaType": m.mime, "url": self._media_url(post.token, m), "width": m.width, "height": m.height}.items() if v} for m in post.media]
        body: dict[str, Any] = {
            "@context": "https://www.w3.org/ns/activitystreams",
            "id": self._status_url(post.token),
            "type": "Note",
            "attributedTo": self._actor_url(post.token),
            "content": "".join(parts),
            "url": post.source_url,
            "to": [_PUBLIC],
            "attachment": attachments,
        }
        if post.published:
            body["published"] = post.published
        return body


def get_embed_host() -> EmbedHost | None:
    """The running host, or None when disabled / not started."""
    return _host


async def start_embed_host(config: dict[str, Any]) -> EmbedHost | None:
    global _host
    settings = EmbedHostSettings.from_config(config)
    if not settings.enabled:
        logger.info("embed_host.disabled")
        return None
    host = EmbedHost(settings)
    try:
        await host.start()
    except OSError as exc:
        logger.error(f"embed_host.start_failed | bind={settings.bind}:{settings.port} error={exc}")
        return None
    _host = host
    return host


async def stop_embed_host() -> None:
    global _host
    host, _host = _host, None
    if host is not None:
        await host.stop()


__all__ = ["EmbedHost", "EmbedHostSettings", "get_embed_host", "start_embed_host", "stop_embed_host"]
