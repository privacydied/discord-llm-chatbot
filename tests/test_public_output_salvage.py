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
