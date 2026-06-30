"""Tests for bot.tasks._reclaim_memory (memory audit fix).

Covers: the health check's high-RSS reclaim path must (a) actually do
something beyond logging, and (b) never instantiate the STT manager as a
side effect of checking whether it has idle models to evict -- that would
cold-start a model-load thread for a bot that never used STT.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import bot.tasks as task_module


def _mock_process(rss_sequence_mb: list[float]) -> MagicMock:
    process = MagicMock()
    infos = []
    for mb in rss_sequence_mb:
        info = MagicMock()
        info.rss = mb * 1024 * 1024
        infos.append(info)
    process.memory_info.side_effect = infos
    return process


class TestReclaimMemory:
    def test_evicts_idle_models_when_manager_already_initialized(self) -> None:
        manager = MagicMock()
        process = _mock_process([1500.0, 1300.0])

        with (
            patch("bot.stt.get_stt_manager_if_initialized", return_value=manager),
            patch("psutil.Process", return_value=process),
        ):
            reclaimed = task_module._reclaim_memory({"MEMORY_EVICT_STT_CACHE_ON_WARNING": True})

        manager.evict_idle_models.assert_called_once()
        assert reclaimed == 200.0

    def test_does_not_create_stt_manager_if_never_initialized(self) -> None:
        """The whole point: a bot that never used STT must not cold-start it
        just because the health check fired under memory pressure."""
        process = _mock_process([1500.0, 1500.0])

        with (
            patch("bot.stt.get_stt_manager_if_initialized", return_value=None) as get_if_init,
            patch("bot.stt.get_stt_manager") as get_or_create,
            patch("psutil.Process", return_value=process),
        ):
            task_module._reclaim_memory({"MEMORY_EVICT_STT_CACHE_ON_WARNING": True})

        get_if_init.assert_called_once()
        get_or_create.assert_not_called()

    def test_respects_evict_flag_disabled(self) -> None:
        manager = MagicMock()
        process = _mock_process([1500.0, 1500.0])

        with (
            patch("bot.stt.get_stt_manager_if_initialized", return_value=manager) as get_if_init,
            patch("psutil.Process", return_value=process),
        ):
            task_module._reclaim_memory({"MEMORY_EVICT_STT_CACHE_ON_WARNING": False})

        get_if_init.assert_not_called()
        manager.evict_idle_models.assert_not_called()

    def test_never_raises_on_internal_failure(self) -> None:
        process = _mock_process([1500.0, 1500.0])

        with (
            patch("bot.stt.get_stt_manager_if_initialized", side_effect=RuntimeError("boom")),
            patch("psutil.Process", return_value=process),
        ):
            # Must not raise -- a broken memory-reclaim path must never crash
            # the health check loop.
            reclaimed = task_module._reclaim_memory({"MEMORY_EVICT_STT_CACHE_ON_WARNING": True})

        assert reclaimed >= 0.0

    def test_returns_zero_when_rss_does_not_drop(self) -> None:
        process = _mock_process([1500.0, 1500.0])

        with (
            patch("bot.stt.get_stt_manager_if_initialized", return_value=None),
            patch("psutil.Process", return_value=process),
        ):
            reclaimed = task_module._reclaim_memory({})

        assert reclaimed == 0.0
