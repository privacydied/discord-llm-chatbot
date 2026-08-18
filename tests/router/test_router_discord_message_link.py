"""Router short-circuits Discord jump URLs to the API instead of the scraper. [CA][SFT]"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from bot.modality import InputItem
from bot.router import Router


CHANNEL_ID = 1392318624508678154
MESSAGE_ID = 1539083511003349112
LINK = f"https://discord.com/channels/1156692144740892796/{CHANNEL_ID}/{MESSAGE_ID}"


class _Channel:
    name = "general"

    def __init__(self, member, content: str) -> None:
        self.guild = SimpleNamespace(name="Guild", get_member=lambda _id: member)
        self._content = content

    def permissions_for(self, _member):
        return SimpleNamespace(view_channel=True, read_message_history=True)

    async def fetch_message(self, _mid):
        return SimpleNamespace(
            author=SimpleNamespace(display_name="alice", name="alice"),
            channel=self,
            guild=self.guild,
            created_at=None,
            clean_content=self._content,
            content=self._content,
            attachments=[],
            embeds=[],
            stickers=[],
        )


def _router(bot) -> Router:
    router = object.__new__(Router)
    router.bot = bot
    router.config = {}
    router.logger = logging.getLogger("test.router.discord_link")
    return router


async def test_general_url_handler_resolves_discord_link_without_scraping() -> None:
    member = SimpleNamespace(id=7)
    channel = _Channel(member, "the linked answer")
    bot = SimpleNamespace(get_channel=lambda _cid: channel)
    message = SimpleNamespace(author=member)

    item = InputItem(source_type="url", payload=LINK, order_index=0)
    out = await _router(bot)._handle_general_url(item, message)

    assert "the linked answer" in out
    assert "Linked Discord message" in out


async def test_non_discord_url_is_not_short_circuited() -> None:
    bot = SimpleNamespace(get_channel=lambda _cid: None)
    assert await _router(bot)._resolve_discord_message_link("https://example.com/a", None) is None


async def test_unreadable_channel_reports_reason_instead_of_scraping() -> None:
    bot = SimpleNamespace(get_channel=lambda _cid: None, fetch_channel=None)
    out = await _router(bot)._resolve_discord_message_link(LINK, SimpleNamespace(author=SimpleNamespace(id=7)))
    assert out is not None
    assert "could not be read" in out
