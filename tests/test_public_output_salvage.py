"""Leak salvage: strip leaking lines instead of discarding whole replies. [REH]

Regression for: nemotron echoed 'MODE GATE' (a system-prompt phrase) inside an
otherwise-good 944-char answer -> entire reply replaced with the safe fallback.
"""

from __future__ import annotations

from bot.public_output import SAFE_FALLBACK_MESSAGE, extract_public_reply_text


class TestLeakSalvage:
    def test_single_leak_line_is_stripped_not_fatal(self) -> None:
        content = (
            "The match ended 2-1 with a late winner in stoppage time.\n"
            "Checking the MODE GATE for this one.\n"
            "Overall it was a deserved result given the second-half pressure."
        )
        result = extract_public_reply_text(content)
        assert result != SAFE_FALLBACK_MESSAGE
        assert "MODE GATE" not in result
        assert "2-1" in result
        assert "deserved result" in result

    def test_thinking_block_stripped_answer_kept(self) -> None:
        content = "<thinking>MODE GATE check, user wants facts about tides.</thinking>\nTides are caused primarily by the Moon's gravitational pull on Earth's oceans."
        result = extract_public_reply_text(content)
        assert result != SAFE_FALLBACK_MESSAGE
        assert "thinking" not in result.lower()
        assert "gravitational" in result

    def test_pure_reasoning_still_falls_back(self) -> None:
        content = "Okay, the user wants me to check the MODE GATE.\nNORMAL MODE applies."
        result = extract_public_reply_text(content)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_tiny_remnant_falls_back(self) -> None:
        # Salvage that leaves almost nothing should not be sent
        content = "MODE GATE\nok."
        result = extract_public_reply_text(content)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_clean_content_untouched(self) -> None:
        content = "Here's a normal helpful reply with plenty of substance in it."
        assert extract_public_reply_text(content) == content


class TestContiguousLeakRegion:
    """Regression: interleaved CoT narration between pattern-matching lines
    must not survive salvage (2026-07-22 leak: 'So MODE = "POLITICAL" ...
    Wait, let me re-read')."""

    def test_interleaved_narration_removed_with_region(self) -> None:
        content = (
            "Checking the MODE GATE for this request.\n"
            'So MODE = "POLITICAL" since B is true but A is false?\n'
            'Wait, let me re-read: "if A true AND B true → MODE = \'POLITICAL\'"\n'
            "POLITICAL MODE it is then.\n"
            "the actual answer is that the election result was certified last week and turnout hit a record."
        )
        result = extract_public_reply_text(content)
        assert result != SAFE_FALLBACK_MESSAGE
        assert "MODE" not in result
        assert "re-read" not in result
        assert "certified last week" in result

    def test_pure_interleaved_reasoning_falls_back(self) -> None:
        # The 2026-07-22 leak shape: narration with no real answer at all
        content = 'So MODE = "POLITICAL" since B is true but A is false? Wait, let me re-read: "if A true AND B true → MODE = \'POLITICAL\'" — so B alone is not enough, right.'
        result = extract_public_reply_text(content)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_mode_equals_pattern_now_caught(self) -> None:
        content = 'Some intro text that is fine and long enough to keep here.\nMODE = "NORMAL" applies to this.\nAnd a closing line with genuine content for the user.'
        result = extract_public_reply_text(content)
        assert "MODE" not in result
        assert "genuine content" in result
