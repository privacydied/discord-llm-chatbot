"""Tests for search_chat_history, convert_currency and get_price tools [IV][SFT][REH]."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from bot.server_archive.models import ArchiveSearchResult
from bot.tools import ToolContext
from bot.tools.builtins import markets
from bot.tools.registry import execute_tool

# ── search_chat_history ──────────────────────────────────────────────────────


def _result(message_id: str, channel_id: str, snippet: str = "knicks in 5", author: str = "Dre") -> ArchiveSearchResult:
    return ArchiveSearchResult.from_row(
        {
            "message_id": message_id,
            "guild_id": "1",
            "channel_id": channel_id,
            "thread_id": None,
            "author_id": "21",
            "author_name": author,
            "channel_name": "general" if channel_id == "10" else "staff",
            "content": snippet,
            "clean_content": snippet,
            "snippet": snippet,
            "created_at": "2026-10-12T20:00:00+00:00",
            "jump_url": None,
            "score": 0.0,
        }
    )


def _archive_ctx(visible_channels: set[int]) -> ToolContext:
    def channel(cid: int):
        ch = MagicMock()
        allowed = cid in visible_channels
        ch.permissions_for.return_value = SimpleNamespace(view_channel=allowed, read_message_history=allowed)
        return ch

    guild = MagicMock()
    guild.id = 1
    guild.get_channel_or_thread.side_effect = channel
    message = SimpleNamespace(id=999, guild=guild, author=SimpleNamespace(id=5))
    return ToolContext(message=message, config={})


async def test_chat_history_hides_channels_the_asker_cannot_read(monkeypatch):
    found = [_result("1", "10"), _result("2", "11", "secret staff stuff"), _result("999", "10", "the question itself")]
    search = AsyncMock(return_value=found)
    monkeypatch.setattr("bot.server_archive.search_archive", search)
    result = await execute_tool("search_chat_history", {"query": "knicks", "author": "dre"}, _archive_ctx({10}))
    assert result.ok
    assert "knicks in 5" in result.content and "[2026-10-12] Dre in #general" in result.content
    assert "secret" not in result.content and "the question itself" not in result.content
    assert search.await_args.kwargs["author_name"] == "dre"


async def test_chat_history_author_mention_becomes_author_id(monkeypatch):
    search = AsyncMock(return_value=[_result("1", "10")])
    monkeypatch.setattr("bot.server_archive.search_archive", search)
    await execute_tool("search_chat_history", {"query": "arsenal", "author": "<@161959731450085378>"}, _archive_ctx({10}))
    assert search.await_args.kwargs["author_id"] == "161959731450085378"
    assert search.await_args.kwargs["author_name"] is None


async def test_chat_history_sanitises_mentions(monkeypatch):
    monkeypatch.setattr("bot.server_archive.search_archive", AsyncMock(return_value=[_result("1", "10", "yo <@123> and @everyone")]))
    result = await execute_tool("search_chat_history", {"query": "yo"}, _archive_ctx({10}))
    assert "<@123>" not in result.content and "@everyone" not in result.content


async def test_chat_history_needs_a_server_and_query():
    assert not (await execute_tool("search_chat_history", {"query": "x"}, ToolContext())).ok
    assert not (await execute_tool("search_chat_history", {"query": " "}, _archive_ctx({10}))).ok


async def test_chat_history_no_matches(monkeypatch):
    monkeypatch.setattr("bot.server_archive.search_archive", AsyncMock(return_value=[]))
    result = await execute_tool("search_chat_history", {"query": "nothing", "author": "dre"}, _archive_ctx({10}))
    assert result.ok and "No messages from dre" in result.content


# ── markets ──────────────────────────────────────────────────────────────────


def _fake_api(responses: dict[str, object]):
    async def fake(url, params=None):
        for key, value in responses.items():
            if key in url:
                return value(params) if callable(value) else value
        return None

    return fake


async def test_convert_fiat_uses_ecb_rate(monkeypatch):
    monkeypatch.setattr(markets, "_get_json", _fake_api({"frankfurter": {"base": "CAD", "date": "2026-10-16", "rates": {"GBP": 0.52956}}}))
    result = await execute_tool("convert_currency", {"amount": 10, "from_currency": "cad", "to_currency": "GBP"}, ToolContext())
    assert result.ok and "10.00 CAD = 5.30 GBP" in result.content and "2026-10-16" in result.content


async def test_convert_crypto_to_fiat(monkeypatch):
    def price(params):
        return {"bitcoin": {"usd": 80000.0}}

    def fx(params):
        return {"rates": {"USD": 1.25}}  # 1 GBP = 1.25 USD

    monkeypatch.setattr(markets, "_get_json", _fake_api({"simple/price": price, "frankfurter": fx}))
    result = await execute_tool("convert_currency", {"amount": 0.5, "from_currency": "BTC", "to_currency": "GBP"}, ToolContext())
    assert result.ok and "= 32,000.00 GBP" in result.content


async def test_convert_rejects_junk_codes():
    result = await execute_tool("convert_currency", {"from_currency": "../x", "to_currency": "GBP"}, ToolContext())
    assert not result.ok


async def test_convert_unsupported_currency(monkeypatch):
    monkeypatch.setattr(markets, "_get_json", AsyncMock(return_value=None))
    result = await execute_tool("convert_currency", {"from_currency": "XYZ", "to_currency": "GBP"}, ToolContext())
    assert not result.ok


async def test_stock_price(monkeypatch):
    meta = {"regularMarketPrice": 343.46, "regularMarketChangePercent": 0.427, "currency": "USD", "shortName": "Apple Inc."}
    monkeypatch.setattr(markets, "_get_json", _fake_api({"finance.yahoo": {"chart": {"result": [{"meta": meta}]}}}))
    result = await execute_tool("get_price", {"symbol": "aapl"}, ToolContext())
    assert result.ok and "Apple Inc. (AAPL): 343.46 USD, +0.43% today" in result.content


async def test_crypto_price_via_search(monkeypatch):
    monkeypatch.setattr(
        markets,
        "_get_json",
        _fake_api(
            {
                "search": {"coins": [{"id": "sui", "name": "Sui", "symbol": "SUI"}]},
                "simple/price": {"sui": {"usd": 3.5, "usd_24h_change": -2.0}},
            }
        ),
    )
    result = await execute_tool("get_price", {"symbol": "SUI", "asset_type": "crypto"}, ToolContext())
    assert result.ok and "Sui (SUI): $3.50, -2.00% in 24h" in result.content


async def test_known_coin_defaults_to_crypto(monkeypatch):
    monkeypatch.setattr(markets, "_get_json", _fake_api({"simple/price": {"bitcoin": {"usd": 82514}}}))
    result = await execute_tool("get_price", {"symbol": "BTC"}, ToolContext())
    assert result.ok and "$82,514.00" in result.content
