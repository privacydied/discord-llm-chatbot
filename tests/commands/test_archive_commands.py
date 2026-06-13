from __future__ import annotations

from types import SimpleNamespace

import pytest

import bot.commands.archive_commands as archive_commands_module
from bot.commands.archive_commands import ArchiveCommands
from bot.server_archive.service import ServerArchiveService


class DummyPermissions:
    administrator = True


class NonAdminPermissions:
    administrator = False


class DummyAuthor:
    guild_permissions = DummyPermissions()


class NonAdminAuthor:
    guild_permissions = NonAdminPermissions()


class DummyContext:
    def __init__(self, *, author=None, channel_id=10) -> None:
        self.guild = SimpleNamespace(id=1)
        self.channel = SimpleNamespace(id=channel_id, guild=self.guild)
        self.author = author or DummyAuthor()
        self.sent = []
        self.replies = []

    async def reply(self, content=None, *, mention_author=False, **kwargs):
        self.sent.append(content)
        self.replies.append({"content": content, "mention_author": mention_author, "kwargs": kwargs})
        return content


@pytest.fixture
def archive_config(tmp_path):
    return {
        "SERVER_ARCHIVE_ENABLED": True,
        "SERVER_ARCHIVE_ENABLE": True,
        "SERVER_ARCHIVE_DB_PATH": str(tmp_path / "archive.sqlite3"),
        "SERVER_ARCHIVE_QUEUE_MAX": 10,
        "SERVER_ARCHIVE_BATCH_SIZE": 1,
        "SERVER_ARCHIVE_SEARCH_LIMIT": 10,
        "SERVER_ARCHIVE_ADMIN_ONLY": True,
        "SERVER_ARCHIVE_SYNC_ON_START": False,
        "SERVER_ARCHIVE_LIVE_TAIL": True,
        "SERVER_ARCHIVE_MAX_MESSAGE_CHARS": 8000,
        "SERVER_ARCHIVE_ARCHIVE_BOT_MESSAGES": False,
        "SERVER_ARCHIVE_INCLUDE_BOT_MESSAGES": False,
    }


async def make_cog(monkeypatch: pytest.MonkeyPatch, config: dict[str, object]) -> tuple[ArchiveCommands, ServerArchiveService]:
    import bot.server_archive.service as service_module

    monkeypatch.setattr(service_module, "load_config", lambda: config)
    service = ServerArchiveService(SimpleNamespace(command_prefix="!"))

    async def start(_bot=None):
        await service.start()
        return service

    async def get(_bot=None):
        return service

    monkeypatch.setattr(archive_commands_module, "start_server_archive_service", start)
    monkeypatch.setattr(archive_commands_module, "get_server_archive_service", get)
    cog = ArchiveCommands(bot=SimpleNamespace(command_prefix="!"))
    await cog.cog_load()
    return cog, service


@pytest.mark.asyncio
async def test_archive_status_output_is_short(monkeypatch, archive_config) -> None:
    cog, service = await make_cog(monkeypatch, archive_config)
    try:
        ctx = DummyContext()
        await ArchiveCommands.archive_status.callback(cog, ctx)
        assert ctx.sent
        assert ctx.replies[0]["kwargs"]["embed"] is not None
        embed_text = "\n".join(
            [
                ctx.replies[0]["kwargs"]["embed"].title or "",
                ctx.replies[0]["kwargs"]["embed"].description or "",
                ctx.replies[0]["kwargs"]["embed"].footer.text if ctx.replies[0]["kwargs"]["embed"].footer else "",
            ],
        )
        assert len(embed_text) < 2000
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_archive_search_output_is_short(monkeypatch, archive_config) -> None:
    cog, service = await make_cog(monkeypatch, archive_config)
    try:

        async def fake_search(*args, **kwargs):
            return [
                SimpleNamespace(
                    channel_name="general",
                    channel_id="10",
                    author_name="Alice",
                    author_id="20",
                    snippet="x" * 400,
                    clean_content="x" * 400,
                    content="x" * 400,
                    jump_url="https://discord.com/channels/1/10/100",
                    created_at="2026-05-08T00:00:00+00:00",
                ),
            ]

        monkeypatch.setattr(archive_commands_module, "search_archive", fake_search)
        ctx = DummyContext()
        await ArchiveCommands.archive_search.callback(cog, ctx, query="needle")
        assert ctx.sent
        assert len(ctx.sent[0]) < 2000
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_archive_commands_are_admin_only(monkeypatch, archive_config) -> None:
    cog, service = await make_cog(monkeypatch, archive_config)
    try:
        ctx = DummyContext(author=NonAdminAuthor())
        await ArchiveCommands.archive_status.callback(cog, ctx)
        assert ctx.sent == ["Archive commands are admin-only on this server."]
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_archive_search_disabled_message(monkeypatch, archive_config) -> None:
    disabled = {
        **archive_config,
        "SERVER_ARCHIVE_ENABLED": False,
        "SERVER_ARCHIVE_ENABLE": False,
    }
    cog, service = await make_cog(monkeypatch, disabled)
    try:
        ctx = DummyContext()
        await ArchiveCommands.archive_search.callback(cog, ctx, query="needle")
        assert ctx.sent == ["Server archive is disabled on this bot."]
    finally:
        await service.stop()
