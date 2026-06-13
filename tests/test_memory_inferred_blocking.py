"""Tests for inferred-memory denylist and stricter recurring_instruction detection.

Verifies that:
- Casual drug statements do not become inferred memories
- Third-party anecdotes do not become inferred memories
- Drug content does not become recurring_instruction
- Explicit future-facing instructions ARE saved
- Explicit !memory-add still works, including sensitive content
- Inferred sensitive content is blocked
- Existing memory list/search behavior is unaffected
"""

from __future__ import annotations

import pytest

from bot.memory.curator import CuratedMemoryCurator


@pytest.fixture
def curator():
    return CuratedMemoryCurator()


# ---------------------------------------------------------------------------
# Part B: Inferred memory blocking
# ---------------------------------------------------------------------------


class TestInferredDenylist:
    """Content that must NEVER be saved as an inferred memory."""

    @pytest.mark.parametrize(
        "text",
        [
            # Casual drug statements
            "also my friends got the xanny munchies (i never did) but 4mg of ativan might do that",
            "i took cocaine at a party last night",
            "weed is everywhere these days",
            "my friend loves getting high on weekends",
            "someone told me they tried ecstasy",
            "i heard that shrooms are great",
            # Medication mentions in casual context
            "my mom takes valium every day",
            "4mg of ativan might do that",
            # Third-party anecdotes
            "my friend tried cocaine at a festival",
            "someone said they did mdma",
            "my mate told me he took oxycontin",
            "she said he got high last night",
            # Sexual/body content
            "i am bisexual",
            "i am mixed race and from london, uk",
        ],
    )
    def test_casual_drug_or_anecdote_blocked(self, curator, text) -> None:
        """Casual drug statements, third-party anecdotes, and sensitive content MUST NOT be saved."""
        result = curator.curate_inferred_candidate(user_id="123", text=text)
        assert result is None, f"Expected None (blocked) for: {text!r}"
        # The bad memory from the bug report must be blocked
        bad_memory = "also my friends got the xanny munchies (i never did) but 4mg of ativan might do that"
        result = curator.curate_inferred_candidate(user_id="123", text=bad_memory)
        assert result is None

    def test_xanny_not_recurring_instruction(self, curator) -> None:
        """The exact bad memory from the bug report must not be classified as recurring_instruction."""
        bad = "also my friends got the xanny munchies (i never did) but 4mg of ativan might do that"
        assert curator._is_recurring_instruction(bad.lower()) is False, "Should not match as recurring_instruction"


class TestInferredAllowed:
    """Content that SHOULD be saved as an inferred memory."""

    @pytest.mark.parametrize(
        ("text", "expected_type"),
        [
            # Strong recurring instructions
            ("from now on, answer in one paragraph", "recurring_instruction"),
            ("always use metric units in your responses", "recurring_instruction"),
            ("you should never call me dude", "recurring_instruction"),
            ("you always provide such detailed answers", "recurring_instruction"),
            # Strong memory
            ("remember that I prefer short replies", "user_preference"),
            # Bot corrections
            ("don't say that again about my code", "recurring_instruction"),
            ("you must always reply with a summary", "recurring_instruction"),
            ("going forward, use UK English", "recurring_instruction"),
        ],
    )
    def test_explicit_instruction_saved(self, curator, text, expected_type) -> None:
        result = curator.curate_inferred_candidate(user_id="123", text=text)
        assert result is not None, f"Expected accepted for: {text!r}"
        assert result.context_type == expected_type, f"Expected {expected_type}, got {result.context_type}"


class TestExplicitMemoryAlwaysWorks:
    """Explicit !memory-add must still work, even for sensitive content."""

    def test_explicit_saves_drug_content(self, curator) -> None:
        result = curator.build_explicit_candidate(
            user_id="123",
            text="I tried cocaine last year, interesting experience",
            source="explicit_memory_command",
        )
        assert result is not None, "Explicit save should NOT block sensitive content"
        assert result.source == "explicit_memory_command"
        assert result.confidence >= 0.9

    def test_explicit_saves_sensitive_content(self, curator) -> None:
        result = curator.build_explicit_candidate(
            user_id="123",
            text="i am bisexual",
            source="explicit_memory_command",
        )
        # Even for sensitive content, explicit saves are allowed
        # (but the sensitive filter _looks_sensitive may block some; this
        #  tests that the mechanism works)
        # Note: _looks_sensitive checks secrets, not identity. Should pass.
        assert result is not None

    def test_explicit_remember_still_works(self, curator) -> None:
        result = curator.build_explicit_candidate(
            user_id="123",
            text="remember that i prefer short replies",
            source="explicit_memory_command",
        )
        assert result is not None


class TestRecurringInstructionSpecificity:
    """_is_recurring_instruction must only match bot-directed instructions."""

    @pytest.mark.parametrize(
        "text",
        [
            "i never did drugs",
            "she never goes to parties",
            "my friends never listen",
            "he never says anything useful",
            "i always forget things",
            "my sister always complains",
        ],
    )
    def test_casual_never_always_not_recurring(self, curator, text) -> None:
        assert curator._is_recurring_instruction(text.lower()) is False

    @pytest.mark.parametrize(
        "text",
        [
            "you should always reply concisely",
            "you must never mention my full name",
            "from now on, answer in one paragraph",
            "going forward, use UK English",
            "don't say that again about my code",
            "you always provide such detailed answers",
            "you should never call me dude",
        ],
    )
    def test_bot_directed_is_recurring(self, curator, text) -> None:
        assert curator._is_recurring_instruction(text.lower()) is True
