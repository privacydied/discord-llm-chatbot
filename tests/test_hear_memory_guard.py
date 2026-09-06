"""Tests for bot.hear._resolve_memory_abort (memory audit fix).

Regression coverage for: the memory guard previously deferred its abort
decision to a hypothetical *second* call (closure state `pending_memory_abort`)
that, for short/single-chunk clips, often never arrived -- a real high-RSS
breach would silently never abort. The fix makes each call self-contained:
it confirms (or dismisses) the breach via its own brief re-measurement.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bot.hear import MEMORY_ABORT_THRESHOLD_MB, _resolve_memory_abort


def _process_reporting(*rss_mb_sequence: float) -> MagicMock:
    """A psutil.Process double whose .memory_info().rss reports each value
    in sequence (MB), one per call, repeating the last value thereafter."""
    process = MagicMock()
    calls = {"n": 0}

    def _memory_info():
        idx = min(calls["n"], len(rss_mb_sequence) - 1)
        calls["n"] += 1
        info = MagicMock()
        info.rss = rss_mb_sequence[idx] * 1024 * 1024
        return info

    process.memory_info.side_effect = _memory_info
    return process


@pytest.mark.asyncio
async def test_below_threshold_never_aborts() -> None:
    process = _process_reporting(100.0)
    result = await _resolve_memory_abort(
        MEMORY_ABORT_THRESHOLD_MB - 1,
        confirm=True,
        process=process,
        confirm_delay=0.0,
    )
    assert result is False


@pytest.mark.asyncio
async def test_breach_without_confirm_aborts_immediately() -> None:
    """confirm=False (long clips): abort on the very first breach, no wait."""
    process = _process_reporting(9999.0)  # would never be consulted
    result = await _resolve_memory_abort(
        MEMORY_ABORT_THRESHOLD_MB + 50,
        confirm=False,
        process=process,
        confirm_delay=0.0,
    )
    assert result is True
    process.memory_info.assert_not_called()


@pytest.mark.asyncio
async def test_single_call_resolves_a_sustained_breach() -> None:
    """The core regression: a single call (no second chunk to follow up)
    must still be able to abort when memory stays high on re-measurement --
    this is exactly the short/single-chunk-clip scenario that previously
    left the breach unresolved forever. _resolve_memory_abort re-measures
    RSS exactly once internally (the confirm step); the initial rss_mb is
    supplied by the caller, not read from `process`."""
    over_threshold = MEMORY_ABORT_THRESHOLD_MB + 100
    process = _process_reporting(over_threshold)  # the one confirm re-measurement
    result = await _resolve_memory_abort(
        over_threshold,
        confirm=True,
        process=process,
        confirm_delay=0.0,
    )
    assert result is True
    process.memory_info.assert_called_once()


@pytest.mark.asyncio
async def test_single_call_dismisses_a_transient_spike() -> None:
    """A transient spike that's gone by the re-measurement must NOT abort --
    this is the whole point of the confirm/debounce window."""
    over_threshold = MEMORY_ABORT_THRESHOLD_MB + 100
    back_to_normal = MEMORY_ABORT_THRESHOLD_MB - 50
    process = _process_reporting(back_to_normal)  # the one confirm re-measurement
    result = await _resolve_memory_abort(
        over_threshold,
        confirm=True,
        process=process,
        confirm_delay=0.0,
    )
    assert result is False


@pytest.mark.asyncio
async def test_remeasure_failure_falls_back_to_first_reading() -> None:
    over_threshold = MEMORY_ABORT_THRESHOLD_MB + 100
    process = MagicMock()
    process.memory_info.side_effect = OSError("boom")
    result = await _resolve_memory_abort(
        over_threshold,
        confirm=True,
        process=process,
        confirm_delay=0.0,
    )
    assert result is True  # first reading (over threshold) wins as fallback


@pytest.mark.asyncio
async def test_high_baseline_small_growth_does_not_abort() -> None:
    """The production case: process idles at ~1.23 GB from unrelated residents
    (threshold 900). A job that grew it by ~30 MB must NOT abort -- the abort
    requires job-attributable growth past the floor. No limits were raised."""
    from bot.hear import MEMORY_GROWTH_FLOOR_MB

    baseline = 1200.0
    reading = baseline + (MEMORY_GROWTH_FLOOR_MB - 10)
    assert reading > MEMORY_ABORT_THRESHOLD_MB
    process = _process_reporting(reading)  # confirm re-measurement stays put
    result = await _resolve_memory_abort(
        reading,
        confirm=True,
        process=process,
        confirm_delay=0.0,
        baseline_rss_mb=baseline,
    )
    assert result is False


@pytest.mark.asyncio
async def test_real_growth_past_threshold_still_aborts() -> None:
    """The guard still fires when the job itself drove RSS over the threshold:
    low baseline + large growth aborts immediately (no confirm)."""
    baseline = 400.0
    reading = MEMORY_ABORT_THRESHOLD_MB + 50
    process = _process_reporting(9999.0)  # never consulted
    result = await _resolve_memory_abort(
        reading,
        confirm=False,
        process=process,
        confirm_delay=0.0,
        baseline_rss_mb=baseline,
    )
    assert result is True
    process.memory_info.assert_not_called()


@pytest.mark.asyncio
async def test_baseline_none_keeps_legacy_absolute_behavior() -> None:
    """baseline=None (or unreadable) preserves the old absolute-only semantics."""
    over_threshold = MEMORY_ABORT_THRESHOLD_MB + 100
    process = _process_reporting(over_threshold)
    result = await _resolve_memory_abort(
        over_threshold,
        confirm=True,
        process=process,
        confirm_delay=0.0,
        baseline_rss_mb=None,
    )
    assert result is True
