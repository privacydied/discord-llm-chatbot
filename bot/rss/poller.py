"""Poll RSS/Atom feeds and post new items to Discord (Zapier replacement). [CA][REH][PA]

Config: ``RSS_FEEDS_FILE`` (default ``data/rss_feeds.json``, gitignored -- private
feed URLs such as Patreon's carry an auth token)::

    {"feeds": [{"name": "True Anon", "url": "https://...", "channel_id": 123,
                "username": "True Anon", "avatar_url": "https://...",
                "template": "**{title}**\\u2014{pubDate}\\n\\n```{description}```\\n{enclosure_url}",
                "trigger": "new", "enabled": true}]}

The file is re-read every cycle, so edits apply without a restart.

Behaviour:
- First sight of a feed records its current items WITHOUT posting (Zapier does
  the same when a zap is switched on), so a 500-episode back catalogue never floods.
- Conditional GET (ETag / Last-Modified): an unchanged feed costs one 304.
- Posts through a channel webhook so each feed keeps its own name and avatar;
  without Manage Webhooks it falls back to a normal bot message.
- At most ``RSS_MAX_POSTS_PER_POLL`` items per feed per cycle (oldest first);
  an item is marked seen only after its post succeeds, so failures retry.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bot.atomic_json import read_json_safe, write_json_atomic
from bot.utils.logging import get_logger

from .feeds import DEFAULT_TEMPLATE, TRIGGER_NEW, TRIGGERS, FeedItem, FeedParseError, item_key, parse_feed, render, validate_template

logger = get_logger(__name__)

WEBHOOK_NAME = "ronin feeds"
USER_AGENT = "Mozilla/5.0 (compatible; ronin-rss/1.0)"
FETCH_TIMEOUT_S = 30.0
DISCORD_TIMEOUT_S = 15.0
MAX_FEED_BYTES = 10 * 1024 * 1024
MIN_INTERVAL_S = 120.0
SEEN_KEYS_CAP = 5000


@dataclass(frozen=True)
class FeedConfig:
    name: str
    url: str
    channel_id: int
    username: str = ""
    avatar_url: str = ""
    template: str = DEFAULT_TEMPLATE
    trigger: str = TRIGGER_NEW
    enabled: bool = True

    @property
    def state_id(self) -> str:
        """State key: hash of the URL (never store a token-bearing URL in state)."""
        return hashlib.sha256(self.url.encode()).hexdigest()[:16]


def load_feed_configs(path: Path) -> list[FeedConfig]:
    """Valid, enabled feeds from the config file; bad entries are logged and skipped."""
    raw = read_json_safe(path, default=None)
    if not isinstance(raw, dict):
        return []
    feeds: list[FeedConfig] = []
    for index, entry in enumerate(raw.get("feeds") or []):
        feed = _feed_from_entry(entry, index)
        if feed is not None and feed.enabled:
            feeds.append(feed)
    return feeds


def _feed_from_entry(entry: Any, index: int) -> FeedConfig | None:
    try:
        url = str(entry["url"]).strip()
        if not url.startswith(("https://", "http://")):
            raise ValueError("url must be http(s)")
        feed = FeedConfig(
            name=str(entry.get("name") or f"feed {index}"),
            url=url,
            channel_id=int(entry["channel_id"]),
            username=str(entry.get("username") or "")[:80],
            avatar_url=str(entry.get("avatar_url") or ""),
            template=str(entry.get("template") or DEFAULT_TEMPLATE),
            trigger=str(entry.get("trigger") or TRIGGER_NEW),
            enabled=bool(entry.get("enabled", True)),
        )
        if feed.trigger not in TRIGGERS:
            raise ValueError(f"trigger must be one of {TRIGGERS}")
        unknown = validate_template(feed.template)
        if unknown:
            raise ValueError(f"unknown template fields {unknown}")
        return feed
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(f"rss.config.invalid_feed index={index} error={exc}")
        return None


@dataclass
class FeedState:
    seen: list[str] = field(default_factory=list)
    initialized: bool = False
    etag: str = ""
    last_modified: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> FeedState:
        if not isinstance(data, dict):
            return cls()
        return cls(
            seen=[str(k) for k in data.get("seen") or []],
            initialized=bool(data.get("initialized")),
            etag=str(data.get("etag") or ""),
            last_modified=str(data.get("last_modified") or ""),
        )

    def remember(self, key: str) -> None:
        self.seen.append(key)
        if len(self.seen) > SEEN_KEYS_CAP:
            del self.seen[: len(self.seen) - SEEN_KEYS_CAP]


class RssPoller:
    def __init__(self, bot: Any, config: dict[str, Any]) -> None:
        self.bot = bot
        self.config_path = Path(str(config.get("RSS_FEEDS_FILE") or "data/rss_feeds.json"))
        self.state_path = Path(str(config.get("RSS_STATE_FILE") or "data/rss_state.json"))
        self.interval_s = max(MIN_INTERVAL_S, float(config.get("RSS_POLL_INTERVAL_S") or 600))
        self.max_posts = max(1, int(config.get("RSS_MAX_POSTS_PER_POLL") or 5))
        self._webhooks: dict[int, Any] = {}
        self._no_webhook_channels: set[int] = set()

    async def run(self) -> None:
        await self.bot.wait_until_ready()
        logger.info(f"rss.poller.started file={self.config_path} interval_s={self.interval_s:.0f}")
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad cycle must not kill the loop
                logger.error(f"rss.poll_cycle_failed error={exc}", exc_info=True)
            await asyncio.sleep(self.interval_s)

    async def poll_once(self) -> int:
        """One pass over every feed. Returns the number of posts made."""
        feeds = load_feed_configs(self.config_path)
        if not feeds:
            return 0
        import aiohttp

        states_raw = read_json_safe(self.state_path, default={}) or {}
        states = {k: FeedState.from_dict(v) for k, v in (states_raw.get("feeds") or {}).items()}
        posted = 0
        timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": USER_AGENT}) as session:
            for feed in feeds:
                state = states.setdefault(feed.state_id, FeedState())
                try:
                    posted += await self._poll_feed(session, feed, state)
                except Exception as exc:  # noqa: BLE001 - isolate feeds from each other
                    logger.warning(f"rss.feed_failed feed={feed.name!r} error={exc}")
        await write_json_atomic(self.state_path, {"feeds": {k: v.__dict__ for k, v in states.items()}})
        return posted

    async def _poll_feed(self, session: Any, feed: FeedConfig, state: FeedState) -> int:
        fetched = await self._fetch(session, feed, state)
        if fetched is None:
            return 0
        feed_title, items = fetched
        if not state.initialized:
            for item in items:
                state.remember(item_key(item, feed.trigger))
            state.initialized = True
            logger.info(f"rss.feed_seeded feed={feed.name!r} items={len(items)} (existing items not posted)")
            return 0
        seen = set(state.seen)
        fresh = [item for item in reversed(items) if item_key(item, feed.trigger) not in seen]
        if len(fresh) > self.max_posts:
            logger.warning(f"rss.backlog_capped feed={feed.name!r} new={len(fresh)} posting={self.max_posts}")
            for skipped in fresh[: -self.max_posts]:
                state.remember(item_key(skipped, feed.trigger))
            fresh = fresh[-self.max_posts :]
        posted = 0
        for item in fresh:
            await self._post(feed, item, feed_title)
            state.remember(item_key(item, feed.trigger))
            posted += 1
        return posted

    async def _fetch(self, session: Any, feed: FeedConfig, state: FeedState) -> tuple[str, list[FeedItem]] | None:
        headers = {}
        if state.etag:
            headers["If-None-Match"] = state.etag
        if state.last_modified:
            headers["If-Modified-Since"] = state.last_modified
        async with session.get(feed.url, headers=headers) as response:
            if response.status == 304:
                return None
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            body = await _read_capped(response)
            etag, last_modified = response.headers.get("ETag", ""), response.headers.get("Last-Modified", "")
        try:
            parsed = await asyncio.to_thread(parse_feed, body)
        except FeedParseError as exc:
            raise RuntimeError(str(exc)) from exc
        state.etag, state.last_modified = etag, last_modified
        return parsed

    async def _post(self, feed: FeedConfig, item: FeedItem, feed_title: str) -> None:
        import discord

        from bot.core.output import safe_send
        from bot.core.text_chunking import split_for_discord

        content = render(feed.template, item, feed_title)
        none = discord.AllowedMentions.none()
        webhook = await self._webhook_for(feed.channel_id)
        if webhook is not None:
            for part in split_for_discord(content):
                kwargs: dict[str, Any] = {"allowed_mentions": none, "wait": True}
                if feed.username:
                    kwargs["username"] = feed.username
                if feed.avatar_url:
                    kwargs["avatar_url"] = feed.avatar_url
                await asyncio.wait_for(webhook.send(part, **kwargs), timeout=DISCORD_TIMEOUT_S)
        else:
            channel = await self._channel(feed.channel_id)
            prefix = f"**{feed.username}** \u00b7 " if feed.username else ""
            await asyncio.wait_for(safe_send(channel, prefix + content, allowed_mentions=none), timeout=DISCORD_TIMEOUT_S)
        logger.info(f"rss.posted feed={feed.name!r} title={item.title[:80]!r} via={'webhook' if webhook else 'bot'}")

    async def _channel(self, channel_id: int) -> Any:
        channel = self.bot.get_channel(channel_id)
        if channel is None:
            channel = await asyncio.wait_for(self.bot.fetch_channel(channel_id), timeout=DISCORD_TIMEOUT_S)
        return channel

    async def _webhook_for(self, channel_id: int) -> Any:
        """This bot's feed webhook in the channel (created once), or None without Manage Webhooks."""
        if channel_id in self._webhooks:
            return self._webhooks[channel_id]
        if channel_id in self._no_webhook_channels:
            return None
        import discord

        channel = await self._channel(channel_id)
        try:
            hooks = await asyncio.wait_for(channel.webhooks(), timeout=DISCORD_TIMEOUT_S)
            me = getattr(self.bot.user, "id", None)
            hook = next((h for h in hooks if h.name == WEBHOOK_NAME and getattr(h.user, "id", None) == me), None)
            if hook is None:
                hook = await asyncio.wait_for(channel.create_webhook(name=WEBHOOK_NAME, reason="RSS feed posts"), timeout=DISCORD_TIMEOUT_S)
        except discord.Forbidden:
            logger.warning(f"rss.no_manage_webhooks channel={channel_id} (posting as the bot instead)")
            self._no_webhook_channels.add(channel_id)
            return None
        self._webhooks[channel_id] = hook
        return hook


async def _read_capped(response: Any) -> bytes:
    """Whole body, refusing anything over MAX_FEED_BYTES.

    ``content.read(n)`` returns only what has arrived so far (one chunk), not n
    bytes -- reading that way handed the parser truncated XML on real feeds.
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.content.iter_chunked(64 * 1024):
        total += len(chunk)
        if total > MAX_FEED_BYTES:
            raise RuntimeError(f"feed larger than {MAX_FEED_BYTES} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


async def start_rss_poller(bot: Any) -> None:
    """Background entry point (bot.setup_background_tasks)."""
    await RssPoller(bot, dict(getattr(bot, "config", {}) or {})).run()
