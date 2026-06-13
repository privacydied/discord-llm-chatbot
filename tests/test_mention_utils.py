"""Unit tests for mention utilities."""

import pytest

pytestmark = pytest.mark.skip(reason="Stale mention utility tests")
from bot.utils.mention_utils import (
    deduplicate_mentions_in_text,
    ensure_single_mention,
    extract_user_ids_from_mentions,
    format_mentions,
)


class TestFormatMentions:
    """Test the format_mentions function."""

    def test_empty_list(self) -> None:
        """Test with empty list."""
        assert format_mentions([]) == ""

    def test_single_user(self) -> None:
        """Test with single user."""
        assert format_mentions(["123"]) == "<@123>"

    def test_multiple_users(self) -> None:
        """Test with multiple different users."""
        result = format_mentions(["123", "456", "789"])
        assert result == "<@123> <@456> <@789>"

    def test_duplicate_users(self) -> None:
        """Test that duplicates are removed while preserving order."""
        result = format_mentions(["123", "456", "123", "789", "456"])
        assert result == "<@123> <@456> <@789>"

    def test_all_same_user(self) -> None:
        """Test with all same user IDs."""
        result = format_mentions(["123", "123", "123"])
        assert result == "<@123>"


class TestExtractUserIdsFromMentions:
    """Test the extract_user_ids_from_mentions function."""

    def test_no_mentions(self) -> None:
        """Test text with no mentions."""
        assert extract_user_ids_from_mentions("hello world") == []

    def test_single_mention(self) -> None:
        """Test text with single mention."""
        assert extract_user_ids_from_mentions("<@123> hello") == ["123"]

    def test_multiple_mentions(self) -> None:
        """Test text with multiple mentions."""
        result = extract_user_ids_from_mentions("<@123> hello <@456> world")
        assert result == ["123", "456"]

    def test_mention_with_exclamation(self) -> None:
        """Test mention with exclamation mark format."""
        assert extract_user_ids_from_mentions("<@!123> hello") == ["123"]

    def test_mixed_mention_formats(self) -> None:
        """Test mixed mention formats."""
        result = extract_user_ids_from_mentions("<@123> and <@!456> are here")
        assert result == ["123", "456"]

    def test_duplicate_mentions(self) -> None:
        """Test that duplicate mentions are captured."""
        result = extract_user_ids_from_mentions("<@123> hello <@123> again")
        assert result == ["123", "123"]


class TestDeduplicateMentionsInText:
    """Test the deduplicate_mentions_in_text function."""

    def test_no_mentions(self) -> None:
        """Test text with no mentions."""
        text = "hello world"
        assert deduplicate_mentions_in_text(text) == text

    def test_single_mention(self) -> None:
        """Test text with single mention (no change expected)."""
        text = "<@123> hello"
        assert deduplicate_mentions_in_text(text) == text

    def test_duplicate_same_user(self) -> None:
        """Test removing duplicate mentions of same user."""
        text = "<@123> hello <@123> world"
        expected = "<@123> hello world"
        assert deduplicate_mentions_in_text(text) == expected

    def test_multiple_different_users(self) -> None:
        """Test that different users are preserved."""
        text = "<@123> hello <@456> world"
        assert deduplicate_mentions_in_text(text) == text

    def test_complex_duplicates(self) -> None:
        """Test complex duplicate pattern."""
        text = "<@123> <@456> hello <@123> world <@789> <@456>"
        expected = "<@123> <@456> hello world <@789>"
        assert deduplicate_mentions_in_text(text) == expected

    def test_mixed_mention_formats(self) -> None:
        """Test with mixed mention formats."""
        text = "<@123> hello <@!123> world"
        expected = "<@123> hello world"
        assert deduplicate_mentions_in_text(text) == expected

    def test_spacing_cleanup(self) -> None:
        """Test that spacing is properly cleaned up."""
        text = "<@123>  <@123>  hello"
        expected = "<@123> hello"
        assert deduplicate_mentions_in_text(text) == expected

    def test_empty_string(self) -> None:
        """Test with empty string."""
        assert deduplicate_mentions_in_text("") == ""

    def test_only_mentions(self) -> None:
        """Test with only mentions."""
        text = "<@123> <@123> <@456>"
        expected = "<@123> <@456>"
        assert deduplicate_mentions_in_text(text) == expected


class TestEnsureSingleMention:
    """Test the ensure_single_mention function."""

    def test_no_existing_mention(self) -> None:
        """Test adding mention when none exists."""
        result = ensure_single_mention("hello world", "123")
        assert result == "<@123> hello world"

    def test_existing_mention_at_start(self) -> None:
        """Test when mention already exists at start."""
        text = "<@123> hello world"
        result = ensure_single_mention(text, "123")
        assert result == "<@123> hello world"

    def test_existing_mention_in_middle(self) -> None:
        """Test when mention exists in middle."""
        text = "hello <@123> world"
        result = ensure_single_mention(text, "123")
        assert result == "<@123> hello world"

    def test_multiple_mentions_same_user(self) -> None:
        """Test when user is mentioned multiple times."""
        text = "<@123> hello <@123> world <@123>"
        result = ensure_single_mention(text, "123")
        assert result == "<@123> hello world"

    def test_mixed_mention_formats(self) -> None:
        """Test with mixed mention formats."""
        text = "<@!123> hello <@123> world"
        result = ensure_single_mention(text, "123")
        assert result == "<@123> hello world"

    def test_empty_content(self) -> None:
        """Test with empty content."""
        result = ensure_single_mention("", "123")
        assert result == "<@123>"

    def test_whitespace_only_content(self) -> None:
        """Test with whitespace-only content."""
        result = ensure_single_mention("   ", "123")
        assert result == "<@123>"

    def test_empty_user_id(self) -> None:
        """Test with empty user ID."""
        text = "hello world"
        result = ensure_single_mention(text, "")
        assert result == text

    def test_other_users_preserved(self) -> None:
        """Test that other users' mentions are preserved."""
        text = "<@456> hello <@123> world <@789>"
        result = ensure_single_mention(text, "123")
        assert result == "<@123> <@456> hello world <@789>"


if __name__ == "__main__":
    pytest.main([__file__])
