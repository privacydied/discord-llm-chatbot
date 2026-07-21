"""Tests for ambient deadline propagation (bot/time_budget.py). [PA][REH]"""

from __future__ import annotations

import asyncio

from bot.time_budget import clear_deadline, remaining_seconds, set_deadline


class TestDeadlineBasics:
    def test_no_deadline_by_default(self) -> None:
        assert remaining_seconds() is None

    def test_set_and_clear(self) -> None:
        token = set_deadline(120.0)
        try:
            rem = remaining_seconds()
            assert rem is not None
            assert 0.0 < rem <= 120.0
        finally:
            clear_deadline(token)
        assert remaining_seconds() is None

    def test_expired_deadline_clamps_to_zero(self) -> None:
        token = set_deadline(0.0)
        try:
            assert remaining_seconds() == 0.0
        finally:
            clear_deadline(token)

    def test_negative_input_treated_as_now(self) -> None:
        token = set_deadline(-5.0)
        try:
            assert remaining_seconds() == 0.0
        finally:
            clear_deadline(token)


class TestDeadlineTaskPropagation:
    async def test_deadline_visible_inside_child_task(self) -> None:
        """A task created after set_deadline (the wait_for pattern) sees it."""
        token = set_deadline(60.0)
        try:

            async def child() -> float | None:
                return remaining_seconds()

            rem = await asyncio.wait_for(child(), timeout=5)
            assert rem is not None
            assert 0.0 < rem <= 60.0
        finally:
            clear_deadline(token)

    async def test_clear_with_foreign_token_never_raises(self) -> None:
        """clear_deadline must swallow cross-context token errors. [REH]"""

        async def make_token():
            return set_deadline(30.0)

        token = await asyncio.wait_for(make_token(), timeout=5)
        clear_deadline(token)  # token from a dead task context — must not raise
