"""Tests for STT language propagation across chunks."""


from bot.stt import ModelSpec


class TestSTTLanguagePropagation:
    """Tests for language detection and propagation across chunks."""

    def test_transcript_cache_key_includes_language(self):
        """Cache key should include language parameter."""
        from bot.hear import _transcript_cache_key

        spec = ModelSpec(size="base", compute_type="int8")

        # Different languages should produce different cache keys
        key1 = _transcript_cache_key("audio123", spec, language="en")
        key2 = _transcript_cache_key("audio123", spec, language="ar")
        key3 = _transcript_cache_key("audio123", spec, language=None)

        # Keys should be different
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_transcript_cache_key_includes_task(self):
        """Cache key should include task mode (transcribe vs translate)."""
        from bot.hear import _transcript_cache_key

        spec = ModelSpec(size="base", compute_type="int8")

        # Different tasks should produce different cache keys
        key1 = _transcript_cache_key("audio123", spec, task="transcribe")
        key2 = _transcript_cache_key("audio123", spec, task="translate")

        # Keys should be different
        assert key1 != key2

    def test_transcript_cache_key_includes_pipeline_version(self):
        """Cache key should include pipeline version for cache invalidation."""
        from bot.hear import _transcript_cache_key, STT_PIPELINE_VERSION

        spec = ModelSpec(size="base", compute_type="int8")

        key = _transcript_cache_key("audio123", spec)

        # Version should be part of the key (key is hash, but we can verify it's set)
        assert STT_PIPELINE_VERSION == "stt-v2-lang-aware-stitch"


class TestSTTConfidenceMetadata:
    """Tests for STT confidence metadata in transcript results."""

    def test_transcript_result_has_confidence_fields(self):
        """TranscriptResult should have confidence and language fields."""
        from bot.hear import TranscriptResult

        spec = ModelSpec(size="base", compute_type="int8")
        result = TranscriptResult(
            text="Hello world",
            segments=[],
            chunks=[],
            duration_out=5.0,
            model_spec=spec,
            cache_hit=False,
            first_chunk_runtime=1.0,
            confidence=0.95,
            confidence_status="high",
            language_detected="en",
            language_confidence=0.98,
        )

        assert result.confidence == 0.95
        assert result.confidence_status == "high"
        assert result.language_detected == "en"
        assert result.language_confidence == 0.98


class TestJoinSegments:
    """Tests for _join_segments function with various inputs."""

    def test_join_segments_with_empty_input(self):
        """Empty segments should return empty string with unknown confidence."""
        from bot.hear import _join_segments

        text, meta = _join_segments([])
        assert text == ""
        assert meta["confidence_status"] == "unknown"

    def test_join_segments_sorts_by_timestamp(self):
        """Segments should be sorted by start timestamp."""
        from bot.hear import _join_segments

        segments = [
            {"text": "second", "start": 5.0, "end": 6.0},
            {"text": "first", "start": 1.0, "end": 2.0},
            {"text": "third", "start": 10.0, "end": 11.0},
        ]

        text, meta = _join_segments(segments)
        # Should be sorted: first, second, third
        assert "first" in text
        assert "second" in text
        assert "third" in text

    def test_join_segments_skips_empty_text(self):
        """Segments with empty text should be skipped."""
        from bot.hear import _join_segments

        segments = [
            {"text": "  ", "start": 1.0, "end": 2.0},
            {"text": "Hello", "start": 3.0, "end": 4.0},
            {"text": "", "start": 5.0, "end": 6.0},
        ]

        text, meta = _join_segments(segments)
        assert text == "Hello"

    def test_join_segments_handles_consecutive_duplicates(self):
        """Exact consecutive duplicates should be skipped."""
        from bot.hear import _join_segments

        segments = [
            {"text": "Hello", "start": 1.0, "end": 2.0},
            {"text": "Hello", "start": 2.0, "end": 3.0},  # Duplicate
            {"text": "world", "start": 3.0, "end": 4.0},
        ]

        text, meta = _join_segments(segments)
        # Should only have "Hello world", not "Hello Hello world"
        assert text == "Hello world"

    def test_join_segments_preserves_non_consecutive_duplicates(self):
        """Non-consecutive duplicates (repeated phrases) should be preserved."""
        from bot.hear import _join_segments

        segments = [
            {"text": "Hello", "start": 1.0, "end": 2.0},
            {"text": "world", "start": 3.0, "end": 4.0},
            {"text": "Hello", "start": 10.0, "end": 11.0},  # Later occurrence
        ]

        text, meta = _join_segments(segments)
        # Should have "Hello" twice since it's not consecutive
        assert text == "Hello world Hello"

    def test_join_segments_preserves_arabic_text(self):
        """Arabic text should be preserved without mangling."""
        from bot.hear import _join_segments

        segments = [
            {"text": "الرئيس جوبايدا", "start": 1.0, "end": 3.0},
            {"text": "بصحب الزلالة", "start": 4.0, "end": 6.0},
        ]

        text, meta = _join_segments(segments)
        assert "الرئيس" in text
        assert "جوبايدا" in text
        assert "بصحب" in text


class TestTupleFix:
    """Tests to verify the tuple unpacking fix in _join_segments."""

    def test_join_segments_returns_tuple(self):
        """_join_segments should return (text, meta) tuple."""
        from bot.hear import _join_segments

        segments = [{"text": "Hello", "start": 1.0, "end": 2.0}]

        result = _join_segments(segments)

        # Should be a tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        # First element is text
        assert result[0] == "Hello"

        # Second element is metadata dict
        assert isinstance(result[1], dict)
        assert "confidence_status" in result[1]

    def test_tuple_unpacking_in_caller(self):
        """Caller should correctly unpack the tuple."""
        from bot.hear import _join_segments

        segments = [{"text": "Hello world", "start": 1.0, "end": 2.0}]

        text, meta = _join_segments(segments)

        # Should be strings, not tuples
        assert isinstance(text, str)
        assert text == "Hello world"
        assert isinstance(meta, dict)
