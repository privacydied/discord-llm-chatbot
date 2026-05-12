from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import bot.commands.rag_commands as rag_module
from bot.commands.rag_commands import RAGCommands


@pytest.fixture
def fake_bot():
    bot = SimpleNamespace()
    bot.config = {}
    bot.hybrid_search = None
    return bot


@pytest.fixture
def fake_search():
    search = SimpleNamespace()
    search.add_document = AsyncMock(return_value=True)
    return search


@pytest.fixture
def rag_cog(fake_bot, monkeypatch):
    monkeypatch.setattr(
        rag_module.asyncio,
        "create_task",
        lambda coro: (coro.close(), None)[1],
    )
    return RAGCommands(fake_bot)


@pytest.fixture
def ctx(fake_bot):
    message = SimpleNamespace(id=777, attachments=[])
    return SimpleNamespace(
        bot=fake_bot,
        guild=SimpleNamespace(id=123),
        channel=SimpleNamespace(id=456),
        author=SimpleNamespace(id=999),
        message=message,
        send=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_rag_cog_registers_index_command(rag_cog):
    names = {cmd.name for cmd in rag_cog.get_commands()}
    assert "index" in names
    assert "rag" in names


@pytest.mark.asyncio
async def test_index_message_content_indexes_direct_text(
    rag_cog, fake_bot, fake_search, ctx
):
    fake_bot.hybrid_search = fake_search
    await rag_cog.index_message_content.callback(rag_cog, ctx, text="hello world")

    fake_search.add_document.assert_awaited_once()
    kwargs = fake_search.add_document.await_args.kwargs
    assert kwargs["text"] == "hello world"
    assert kwargs["metadata"]["source_type"] == "text"
    ctx.send.assert_awaited_once()
    assert "Indexed 1 item" in ctx.send.await_args.args[0]


@pytest.mark.asyncio
async def test_index_message_content_honors_rag_disable(
    rag_cog, fake_bot, fake_search, ctx, monkeypatch
):
    fake_bot.hybrid_search = fake_search
    monkeypatch.setattr(
        "bot.commands.rag_commands.is_server_feature_enabled",
        lambda guild_id, feature: False,
    )

    await rag_cog.index_message_content.callback(rag_cog, ctx, text="hello world")

    fake_search.add_document.assert_not_called()
    ctx.send.assert_awaited_once()
    assert "RAG is disabled" in ctx.send.await_args.args[0]


@pytest.mark.asyncio
async def test_index_message_content_indexes_url_and_attachment(
    rag_cog, fake_bot, fake_search, ctx, monkeypatch
):
    fake_bot.hybrid_search = fake_search
    ctx.message.attachments = [SimpleNamespace(filename="notes.pdf")]

    monkeypatch.setattr(
        "bot.document_ingest.ingest_document_from_url",
        AsyncMock(return_value={"text": "url text", "metadata": {"source": "url"}}),
    )
    monkeypatch.setattr(
        "bot.document_ingest.ingest_document_attachment",
        AsyncMock(return_value={"text": "attachment text", "metadata": {"pages": 1}}),
    )

    await rag_cog.index_message_content.callback(
        rag_cog,
        ctx,
        text="see https://example.com/doc and also this",
    )

    assert fake_search.add_document.await_count == 3
    assert any(
        call.kwargs["metadata"]["source_type"] == "url"
        for call in fake_search.add_document.await_args_list
    )
    assert any(
        call.kwargs["metadata"]["source_type"] == "attachment"
        for call in fake_search.add_document.await_args_list
    )
    assert any(
        call.kwargs["metadata"]["source_type"] == "text"
        for call in fake_search.add_document.await_args_list
    )
    ctx.send.assert_awaited_once()
    assert "Indexed 3 items" in ctx.send.await_args.args[0]
