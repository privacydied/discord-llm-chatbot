"""
Tests for public output sanitizer.
"""

import pytest

from bot.public_output import (
    SAFE_FALLBACK_MESSAGE,
    extract_public_reply_text,
    has_reasoning_leakage,
)


class TestExtractPublicReplyText:
    """Test the public reply text extraction function."""

    def test_normal_text_passes_through(self):
        """Normal public text should pass through unchanged."""
        text = "Hello, this is a normal response."
        result = extract_public_reply_text(text)
        assert result == text

    def test_empty_string_returns_fallback(self):
        """Empty string should return fallback message."""
        result = extract_public_reply_text("")
        assert result == SAFE_FALLBACK_MESSAGE

    def test_none_returns_fallback(self):
        """None input should return fallback message."""
        result = extract_public_reply_text(None)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_whitespace_only_returns_fallback(self):
        """Whitespace only should return fallback message."""
        result = extract_public_reply_text("   \n\n  ")
        assert result == SAFE_FALLBACK_MESSAGE

    def test_reasoning_leak_blocked_okay_user(self):
        """Reasoning starting with 'Okay, the user' should be blocked."""
        text = "Okay, the user shared a video. Let me analyze this."
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_reasoning_leak_blocked_okay_mixed_case(self):
        """Case-insensitive reasoning detection."""
        text = "OKAY, THE USER shared a video."
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_reasoning_leak_blocked_first_i_need(self):
        """Reasoning starting with 'First, I need' should be blocked."""
        text = "First, I need to figure out what this video is about."
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_reasoning_leak_blocked_mode_gate(self):
        """MODE GATE mention should be blocked."""
        text = "Checking the MODE GATE criteria..."
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_reasoning_leak_blocked_political_mode(self):
        """POLITICAL MODE mention should be blocked."""
        text = "This is POLITICAL MODE content."
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_normal_political_discussion_allowed(self):
        """Normal text mentioning politics should be allowed."""
        text = "The political situation in Europe is complex."
        result = extract_public_reply_text(text)
        assert result == text

    def test_thinking_tags_blocked(self):
        """Content with <thinking> tags should be blocked."""
        text = "<thinking>I need to analyze this</thinking> Here's the answer."
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_reasoning_tags_blocked(self):
        """Content with <reasoning> tags should be blocked."""
        text = "<reasoning>Analyzing the request...</reasoning> Result is 42."
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_whitespace_normalized(self):
        """Excessive whitespace should be normalized."""
        text = "Hello.\n\n\n\n\nWorld."
        result = extract_public_reply_text(text)
        assert result == "Hello.\n\nWorld."


class TestHasReasoningLeakage:
    """Test the reasoning leakage detection function."""

    def test_empty_string_no_leak(self):
        """Empty string has no leakage."""
        assert has_reasoning_leakage("") is False

    def test_normal_text_no_leak(self):
        """Normal text has no leakage."""
        text = "Hello, how can I help you today?"
        assert has_reasoning_leakage(text) is False

    def test_okay_the_user_is_leak(self):
        """'Okay, the user' is a leak pattern."""
        assert has_reasoning_leakage("Okay, the user") is True

    def test_mode_gate_is_leak(self):
        """MODE GATE is a leak pattern."""
        assert has_reasoning_leakage("Checking the MODE GATE") is True

    def test_explicit_lens_is_leak(self):
        """EXPLICIT_LENS_REQUEST is a leak pattern."""
        assert has_reasoning_leakage("EXPLICIT_LENS_REQUEST detected") is True

    def test_thinking_tag_is_leak(self):
        """<thinking> is a leak pattern."""
        assert has_reasoning_leakage("<thinking>") is True

    # v2: tool-call / JSON-like leakage
    def test_tool_call_json_leak(self):
        assert has_reasoning_leakage('{"tool": "search"}') is True

    def test_function_type_json_leak(self):
        assert has_reasoning_leakage('{"type": "function", "name": "search"}') is True

    def test_tool_call_array_leak(self):
        assert has_reasoning_leakage('[{"name": "search"}]') is True

    # v2: internal routing/status fragments
    def test_dispatch_status_leak(self):
        assert has_reasoning_leakage("dispatch: status=ok") is True

    def test_router_internal_leak(self):
        assert has_reasoning_leakage("router=internal") is True

    def test_pipeline_timeout_leak(self):
        assert has_reasoning_leakage("pipeline=timeout") is True

    # v2: analysis/final/commentary role leakage
    def test_final_answer_leak(self):
        assert has_reasoning_leakage("final answer: 42") is True

    def test_analysis_summary_leak(self):
        assert has_reasoning_leakage("analysis summary: all good") is True

    def test_commentary_only_leak(self):
        assert has_reasoning_leakage("commentary only: this is internal") is True

    # v2: raw prompt scaffolding markers
    def test_system_tag_leak(self):
        assert has_reasoning_leakage("<system>instructions</system>") is True

    def test_instruction_tag_leak(self):
        assert has_reasoning_leakage("<instruction>do this</instruction>") is True
