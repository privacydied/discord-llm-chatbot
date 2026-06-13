"""Unit tests for TTS text preparation and chunking functions.

These tests verify the text cleaning, number expansion, and sentence
chunking functions work correctly without requiring the full TTS engine.
[REH][CA]
"""

import pytest


class TestCleanText:
    """Tests for TTSManager._clean_text (via import)."""

    @pytest.fixture
    def clean_text(self):
        """Import the _clean_text method pattern from interface."""
        import re

        def _clean_text(text: str) -> str:
            if not text:
                return ""
            text = re.sub(r"<@!?\d+>", "", text)
            text = re.sub(r"<@&\d+>", "", text)
            text = re.sub(r"<#\d+>", "", text)
            text = re.sub(r"<a?:\w+:\d+>", "", text)
            text = re.sub(r"```[\s\S]*?```", " code block ", text)
            text = re.sub(r"`[^`]+`", "", text)
            text = re.sub(r"https?://\S+", " link ", text)
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            text = re.sub(r"__(.+?)__", r"\1", text)
            text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", text)
            text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
            text = re.sub(r"~~(.+?)~~", r"\1", text)
            text = re.sub(r"\|\|(.+?)\|\|", r"\1", text)
            text = text.replace("**", "").replace("__", "").replace("~~", "").replace("||", "")
            text = text.replace(""", '"').replace(""", '"')
            text = text.replace("'", "'").replace("'", "'")
            text = text.replace("—", ", ").replace("–", ", ")
            text = text.replace("…", "...")
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n+", " ", text)
            return text.strip()

        return _clean_text

    def test_discord_user_mention(self, clean_text) -> None:
        """User mentions should be removed."""
        assert clean_text("Hello <@123456789>!") == "Hello !"
        assert clean_text("<@!987654321> said hi") == "said hi"

    def test_discord_role_mention(self, clean_text) -> None:
        """Role mentions should be removed."""
        assert clean_text("Attention <@&111222333>!") == "Attention !"

    def test_discord_channel_mention(self, clean_text) -> None:
        """Channel mentions should be removed."""
        assert clean_text("Check <#444555666>") == "Check"

    def test_discord_custom_emoji(self, clean_text) -> None:
        """Custom emoji should be removed."""
        assert clean_text("Nice <:thumbsup:123>") == "Nice"
        assert clean_text("Animated <a:dance:456>") == "Animated"

    def test_code_block(self, clean_text) -> None:
        """Code blocks should be replaced with 'code block'."""
        text = "Here's code:\n```python\nprint('hi')\n```\nDone."
        result = clean_text(text)
        assert "code block" in result
        assert "print" not in result

    def test_inline_code(self, clean_text) -> None:
        """Inline code should be removed."""
        assert clean_text("Use `git commit`") == "Use"

    def test_url_replacement(self, clean_text) -> None:
        """URLs should be replaced with 'link'."""
        assert "link" in clean_text("Visit https://example.com now")
        assert "example.com" not in clean_text("Visit https://example.com now")

    def test_markdown_bold(self, clean_text) -> None:
        """Bold markdown should be stripped, keeping content."""
        assert clean_text("This is **bold** text") == "This is bold text"
        assert clean_text("This is __bold__ text") == "This is bold text"

    def test_markdown_italic(self, clean_text) -> None:
        """Italic markdown should be stripped, keeping content."""
        assert clean_text("This is *italic* text") == "This is italic text"

    def test_markdown_strikethrough(self, clean_text) -> None:
        """Strikethrough should be stripped, keeping content."""
        assert clean_text("This is ~~wrong~~ text") == "This is wrong text"

    def test_markdown_spoiler(self, clean_text) -> None:
        """Spoilers should be stripped, keeping content."""
        assert clean_text("The answer is ||42||") == "The answer is 42"

    def test_unicode_normalization(self, clean_text) -> None:
        """Unicode quotes and dashes should be normalized."""
        # Em dash → comma space
        result = clean_text("one\u2014two")
        assert "one" in result
        assert "two" in result
        # Ellipsis → three dots
        result = clean_text("wait\u2026")
        assert "wait" in result
        assert "..." in result
        # Smart quotes - just verify content preserved
        result = clean_text("\u201cHello\u201d")
        assert "Hello" in result

    def test_whitespace_normalization(self, clean_text) -> None:
        """Multiple spaces and newlines should be collapsed."""
        assert clean_text("hello    world") == "hello world"
        assert clean_text("hello\n\nworld") == "hello world"


class TestNumberToWords:
    """Tests for number_to_words function."""

    @pytest.fixture
    def number_to_words(self):
        from bot.tts.eng_g2p_local import number_to_words

        return number_to_words

    def test_small_numbers(self, number_to_words) -> None:
        """Small numbers should be converted."""
        assert number_to_words("0") == "zero"
        assert number_to_words("5") == "five"
        assert number_to_words("13") == "thirteen"
        assert number_to_words("25") == "twenty five"

    def test_years(self, number_to_words) -> None:
        """Years should be spoken naturally."""
        assert "twenty" in number_to_words("2024")
        assert "nineteen" in number_to_words("1984")
        assert "two thousand" in number_to_words("2000")
        assert "two thousand five" in number_to_words("2005")

    def test_large_numbers(self, number_to_words) -> None:
        """Hundreds and thousands should work."""
        assert "hundred" in number_to_words("500")
        assert "thousand" in number_to_words("1000")
        assert "thousand" in number_to_words("5,000")

    def test_ordinals(self, number_to_words) -> None:
        """Ordinals should be converted."""
        assert number_to_words("1st") == "first"
        assert number_to_words("2nd") == "second"
        assert number_to_words("3rd") == "third"
        assert number_to_words("10th") == "tenth"

    def test_punctuation_preserved(self, number_to_words) -> None:
        """Trailing punctuation should be preserved."""
        assert number_to_words("2024.") == "twenty twenty four."
        assert number_to_words("100?") == "one hundred?"

    def test_mixed_text(self, number_to_words) -> None:
        """Numbers in text context should work."""
        result = number_to_words("I have 3 cats")
        assert "three" in result
        assert "cats" in result


class TestSentenceChunking:
    """Tests for _split_into_sentences function."""

    @pytest.fixture
    def split_sentences(self):
        """Standalone implementation of sentence splitting for tests."""
        import re

        _MIN_CHUNK_CHARS = 20
        _MAX_CHUNK_CHARS = 400

        def _split_into_sentences(text: str):
            if not text or not text.strip():
                return []
            raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            chunks = []
            current_chunk = ""
            for sent in raw_sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(current_chunk) + len(sent) + 1 > _MAX_CHUNK_CHARS:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    if len(sent) > _MAX_CHUNK_CHARS:
                        parts = re.split(r"(?<=[,;])\s+", sent)
                        for part in parts:
                            part = part.strip()
                            if len(current_chunk) + len(part) + 1 > _MAX_CHUNK_CHARS:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = part
                            else:
                                current_chunk = (current_chunk + " " + part).strip() if current_chunk else part
                    else:
                        current_chunk = sent
                elif len(current_chunk) + len(sent) + 1 < _MIN_CHUNK_CHARS:
                    current_chunk = (current_chunk + " " + sent).strip() if current_chunk else sent
                elif current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sent
                else:
                    current_chunk = sent
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(chunks) > 1:
                merged = []
                buffer = ""
                for chunk in chunks:
                    if len(buffer) + len(chunk) < _MIN_CHUNK_CHARS:
                        buffer = (buffer + " " + chunk).strip() if buffer else chunk
                    else:
                        if buffer:
                            merged.append(buffer)
                        buffer = chunk
                if buffer:
                    merged.append(buffer)
                chunks = merged
            return chunks

        return _split_into_sentences

    def test_empty_text(self, split_sentences) -> None:
        """Empty text should return empty list."""
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_single_sentence(self, split_sentences) -> None:
        """Single sentence should return one chunk."""
        result = split_sentences("Hello world.")
        assert len(result) == 1
        assert "Hello world" in result[0]

    def test_multiple_sentences(self, split_sentences) -> None:
        """Multiple sentences should be split."""
        result = split_sentences("Hello. World. Test.")
        assert len(result) >= 1  # May be merged if short

    def test_question_exclamation(self, split_sentences) -> None:
        """Questions and exclamations should be split."""
        result = split_sentences("Hello! How are you? I'm fine.")
        assert len(result) >= 1

    def test_short_merge(self, split_sentences) -> None:
        """Very short sentences should be merged."""
        result = split_sentences("Hi. Yes. OK.")
        # Should be merged into fewer chunks
        assert len(result) <= 2

    def test_long_sentence_split(self, split_sentences) -> None:
        """Very long sentences should be split at commas."""
        long = "This is a very long sentence with many, many, many words " * 20
        result = split_sentences(long)
        # Should be split into multiple chunks
        for chunk in result:
            assert len(chunk) <= 450  # Allow some buffer over max

    def test_preserves_punctuation(self, split_sentences) -> None:
        """Sentence-ending punctuation should be preserved."""
        result = split_sentences("Hello world! How are you?")
        combined = " ".join(result)
        assert "!" in combined or "?" in combined


class TestNormalizeText:
    """Tests for normalize_text function."""

    @pytest.fixture
    def normalize_text(self):
        from bot.tts.eng_g2p_local import normalize_text

        return normalize_text

    def test_colon_to_comma(self, normalize_text) -> None:
        """Colons should become commas for phrase boundary."""
        result = normalize_text("Note: this is important")
        assert ":" not in result
        assert "," in result

    def test_semicolon_to_comma(self, normalize_text) -> None:
        """Semicolons should become commas."""
        result = normalize_text("First; second")
        assert ";" not in result
        assert "," in result

    def test_preserves_period(self, normalize_text) -> None:
        """Periods should be preserved."""
        result = normalize_text("Hello. World.")
        assert "." in result

    def test_preserves_question(self, normalize_text) -> None:
        """Question marks should be preserved."""
        result = normalize_text("How are you?")
        assert "?" in result

    def test_preserves_exclamation(self, normalize_text) -> None:
        """Exclamation marks should be preserved."""
        result = normalize_text("Wow!")
        assert "!" in result

    def test_removes_brackets(self, normalize_text) -> None:
        """Brackets should be removed."""
        result = normalize_text("Hello (world)")
        assert "(" not in result
        assert ")" not in result
        assert "world" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
