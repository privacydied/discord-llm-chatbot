from typing import Never

import pytest

from bot.stt_pipeline.runtime import (
    ensure_stt_manager_ready,
    load_stt_runtime_compat,
    parse_stt_max_ram_mb,
)


def test_parse_stt_max_ram_mb_valid(monkeypatch) -> None:
    monkeypatch.setenv("STT_MAX_RAM_MB", "512")
    assert parse_stt_max_ram_mb() == 512


def test_parse_stt_max_ram_mb_invalid_or_non_positive(monkeypatch) -> None:
    monkeypatch.setenv("STT_MAX_RAM_MB", "0")
    assert parse_stt_max_ram_mb() is None

    monkeypatch.setenv("STT_MAX_RAM_MB", "-1")
    assert parse_stt_max_ram_mb() is None

    monkeypatch.setenv("STT_MAX_RAM_MB", "abc")
    assert parse_stt_max_ram_mb() is None


def test_load_stt_runtime_compat(monkeypatch) -> None:
    monkeypatch.setenv("YOUTUBE_TRANSCRIPT_FIRST", "0")
    monkeypatch.setenv("STT_MAX_RAM_MB", "256")
    cfg = load_stt_runtime_compat()
    assert cfg.youtube_transcript_first is False
    assert cfg.max_ram_mb == 256


@pytest.mark.asyncio
async def test_ensure_stt_manager_ready_no_ensure_ready_fail_open() -> None:
    class Manager:
        def is_available(self) -> bool:
            return True

    assert await ensure_stt_manager_ready(Manager()) is True


@pytest.mark.asyncio
async def test_ensure_stt_manager_ready_async_ensure_ready() -> None:
    class Manager:
        def is_available(self) -> bool:
            return True

        async def ensure_ready(self) -> bool:
            return True

    assert await ensure_stt_manager_ready(Manager()) is True


@pytest.mark.asyncio
async def test_ensure_stt_manager_ready_sync_ensure_ready_false() -> None:
    class Manager:
        def is_available(self) -> bool:
            return True

        def ensure_ready(self) -> bool:
            return False

    assert await ensure_stt_manager_ready(Manager()) is False


@pytest.mark.asyncio
async def test_ensure_stt_manager_ready_is_available_false() -> None:
    class Manager:
        def is_available(self) -> bool:
            return False

    assert await ensure_stt_manager_ready(Manager()) is False


@pytest.mark.asyncio
async def test_ensure_stt_manager_ready_ensure_ready_raises() -> None:
    class Manager:
        def is_available(self) -> bool:
            return True

        async def ensure_ready(self) -> Never:
            msg = "boom"
            raise RuntimeError(msg)

    assert await ensure_stt_manager_ready(Manager()) is False


@pytest.mark.asyncio
async def test_ensure_stt_manager_ready_is_available_raises() -> None:
    class Manager:
        def is_available(self) -> Never:
            msg = "boom"
            raise RuntimeError(msg)

    assert await ensure_stt_manager_ready(Manager()) is False
