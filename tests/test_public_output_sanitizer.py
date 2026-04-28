"""Tests for public output sanitizer to prevent reasoning leaks to Discord."""

import pytest
from bot.public_output import (
    extract_public_reply_text,
    has_reasoning_leakage,
    SAFE_FALLBACK_MESSAGE,
)


class TestExtractPublicReplyText:
    """Tests for extract_public_reply_text function."""

    def test_returns_normal_text(self):
        """Normal assistant replies should pass through unchanged."""
        text = "This is a normal public response to the user."
        result = extract_public_reply_text(text)
        assert result == text

    def test_strips_leading_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        text = "   Normal response text here   "
        result = extract_public_reply_text(text)
        assert result == "Normal response text here"

    def test_collapse_multiple_blank_lines(self):
        """Multiple blank lines should be collapsed."""
        text = "Line 1\n\n\n\n\nLine 2"
        result = extract_public_reply_text(text)
        assert result == "Line 1\n\nLine 2"

    def test_empty_input_returns_fallback(self):
        """Empty string should return fallback message."""
        result = extract_public_reply_text("")
        assert result == SAFE_FALLBACK_MESSAGE

    def test_none_input_returns_fallback(self):
        """None input should return fallback message."""
        result = extract_public_reply_text(None)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_whitespace_only_returns_fallback(self):
        """Whitespace-only input should return fallback message."""
        result = extract_public_reply_text("   \n\n   ")
        assert result == SAFE_FALLBACK_MESSAGE


class TestReasoningLeakageBlocking:
    """Tests that reasoning leaks are properly blocked."""

    def test_blocks_okay_the_user_start(self):
        """Text starting with 'Okay, the user...' should be blocked."""
        text = "Okay, the user shared a lengthy, disjointed audio transcript"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_the_user_shared(self):
        """Text containing 'The user shared...' should be blocked."""
        text = "The user shared a video and asked about it."
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_first_i_need_to(self):
        """Text starting with 'First, I need to...' should be blocked."""
        text = "First, I need to figure out if this falls under POLITICAL MODE"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_i_need_to_figure_out(self):
        """Text containing 'I need to figure out...' should be blocked."""
        text = "I need to figure out what the user wants."
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_checking_mode_gate(self):
        """Text containing 'Checking the MODE GATE' should be blocked."""
        text = "Checking the MODE GATE criteria"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_mode_gate(self):
        """Text containing 'MODE GATE' should be blocked."""
        text = "Checking MODE GATE"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_political_mode(self):
        """Text containing 'POLITICAL MODE' should be blocked."""
        text = "This is POLITICAL MODE content"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_normal_mode(self):
        """Text containing 'NORMAL MODE' should be blocked."""
        text = "Switching to NORMAL MODE"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_explicit_lens_request(self):
        """Text containing 'EXPLICIT_LENS_REQUEST' should be blocked."""
        text = "EXPLICIT_LENS_REQUEST detected"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_politics_core_topic(self):
        """Text containing 'POLITICS_CORE_TOPIC' should be blocked."""
        text = "POLITICS_CORE_TOPIC is true"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_chain_of_thought(self):
        """Text containing 'chain-of-thought' should be blocked."""
        text = "This uses chain-of-thought reasoning"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_hidden_reasoning(self):
        """Text containing 'hidden reasoning' should be blocked."""
        text = "Using hidden reasoning here"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_scratchpad(self):
        """Text containing 'scratchpad' should be blocked."""
        text = "Let me check my scratchpad"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_analysis_prefix(self):
        """Text starting with 'analysis:' should be blocked."""
        text = "Analysis: This is internal analysis"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_reasoning_prefix(self):
        """Text starting with 'reasoning:' should be blocked."""
        text = "Reasoning: This is my reasoning"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_thinking_tags(self):
        """Text with <thinking> tags should be blocked."""
        text = "<thinking>This is internal thinking</thinking>"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_reasoning_tags(self):
        """Text with <reasoning> tags should be blocked."""
        text = "<reasoning>This is internal reasoning</reasoning>"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_scratchpad_tags(self):
        """Text with <scratchpad> tags should be blocked."""
        text = "<scratchpad>Internal notes here</scratchpad>"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE


class TestEnhancedReasoningPatterns:
    """Tests for additional reasoning patterns added in the fix."""

    def test_blocks_i_should_analyze(self):
        """Text with 'I should analyze' should be blocked."""
        text = "I should analyze this video content"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_let_me_analyze(self):
        """Text with 'Let me analyze' should be blocked."""
        text = "Let me analyze this for you"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_i_will_analyze(self):
        """Text with 'I will analyze' should be blocked."""
        text = "I will analyze this audio file"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_i_need_to_analyze(self):
        """Text with 'I need to analyze' should be blocked."""
        text = "I need to analyze the content first"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_based_on_the_above(self):
        """Text with 'Based on the above' should be blocked."""
        text = "Based on the above analysis"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_according_to_the_rules(self):
        """Text with 'According to the rules' should be blocked."""
        text = "According to the rules, I should"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_as_an_ai_assistant(self):
        """Text with 'As an AI assistant' should be blocked."""
        text = "As an AI assistant, I cannot"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_as_an_ai_language_model(self):
        """Text with 'As an AI language model' should be blocked."""
        text = "As an AI language model"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_thought_prefix(self):
        """Text starting with 'thought:' should be blocked."""
        text = "Thought: I should analyze this"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_plan_prefix(self):
        """Text starting with 'plan:' should be blocked."""
        text = "Plan: First, analyze the content"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_steps_prefix(self):
        """Text starting with 'steps:' should be blocked."""
        text = "Steps: 1. Analyze the video"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_chinese_think_tags(self):
        """Text with  <think> /思考 tags should be blocked."""
        text = " <think> This is internal thinking思考"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE

    def test_blocks_analysis_tags(self):
        """Text with <analysis> tags should be blocked."""
        text = "<analysis>This is internal analysis</analysis>"
        result = extract_public_reply_text(text)
        assert result == SAFE_FALLBACK_MESSAGE


class TestHasReasoningLeakage:
    """Tests for has_reasoning_leakage helper function."""

    def test_returns_true_for_leak(self):
        """Should return True for text with reasoning leak."""
        text = "Okay, the user shared a video"
        assert has_reasoning_leakage(text) is True

    def test_returns_false_for_public_text(self):
        """Should return False for normal public text."""
        text = "This is a normal public response."
        assert has_reasoning_leakage(text) is False

    def test_returns_false_for_empty(self):
        """Should return False for empty/None input."""
        assert has_reasoning_leakage("") is False
        assert has_reasoning_leakage(None) is False


class TestLogMetadata:
    """Tests that logging metadata is properly handled."""

    def test_metadata_included_in_log(self):
        """Log should include metadata when provided."""
        # This test just verifies the function doesn't crash with all parameters
        text = "Okay, the user shared a video"
        result = extract_public_reply_text(
            text,
            request_id="req-123",
            message_id="msg-456",
            channel_id="chan-789",
            guild_id="guild-abc",
            provider="openai",
            model="gpt-4",
        )
        assert result == SAFE_FALLBACK_MESSAGE

    def test_normal_text_with_metadata(self):
        """Normal text should still pass through with metadata."""
        text = "Normal public response"
        result = extract_public_reply_text(
            text,
            request_id="req-123",
            message_id="msg-456",
        )
        assert result == text
