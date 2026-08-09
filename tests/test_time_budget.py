"""Tests for ambient deadline propagation (bot/time_budget.py). [PA][REH]"""

from __future__ import annotations

import asyncio

from bot.time_budget import (
    DISPATCH_RESERVE_S,
    LADDER_MIN_BUDGET_S,
    clamp_to_deadline,
    clear_deadline,
    narrow_deadline,
    remaining_seconds,
    set_deadline,
)


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


class TestNarrowDeadline:
    """narrow_deadline must only tighten — a nested guard can never buy more time."""

    def test_narrows_when_shorter(self) -> None:
        outer = set_deadline(240.0)
        try:
            inner = narrow_deadline(120.0)
            try:
                rem = remaining_seconds()
                assert rem is not None
                assert 110.0 < rem <= 120.0
            finally:
                clear_deadline(inner)
            rem_after = remaining_seconds()
            assert rem_after is not None
            assert rem_after > 120.0
        finally:
            clear_deadline(outer)

    def test_does_not_widen(self) -> None:
        outer = set_deadline(30.0)
        try:
            inner = narrow_deadline(300.0)
            try:
                rem = remaining_seconds()
                assert rem is not None
                assert rem <= 30.0
            finally:
                clear_deadline(inner)
        finally:
            clear_deadline(outer)

    def test_arms_when_no_ambient_deadline(self) -> None:
        token = narrow_deadline(45.0)
        try:
            rem = remaining_seconds()
            assert rem is not None
            assert 35.0 < rem <= 45.0
        finally:
            clear_deadline(token)


class TestClampToDeadline:
    """The clamp both ladders share: sub-budget must fit inside the ambient guard."""

    def test_untouched_without_deadline(self) -> None:
        budget, ambient = clamp_to_deadline(300.0)
        assert budget == 300.0
        assert ambient is None

    def test_clamps_vision_budget_to_item_guard(self) -> None:
        """The live bug: VISION_PER_ITEM_BUDGET=300 nested in a 120s item wait_for."""
        token = narrow_deadline(120.0)
        try:
            budget, ambient = clamp_to_deadline(300.0)
            assert ambient is not None
            assert budget <= 120.0 - DISPATCH_RESERVE_S + 1.0
            assert budget >= LADDER_MIN_BUDGET_S
        finally:
            clear_deadline(token)

    def test_does_not_inflate_smaller_budget(self) -> None:
        token = set_deadline(240.0)
        try:
            budget, _ = clamp_to_deadline(45.0)
            assert budget == 45.0
        finally:
            clear_deadline(token)

    def test_floor_respected_when_nearly_expired(self) -> None:
        token = set_deadline(1.0)
        try:
            budget, _ = clamp_to_deadline(300.0)
            assert budget == LADDER_MIN_BUDGET_S
        finally:
            clear_deadline(token)
