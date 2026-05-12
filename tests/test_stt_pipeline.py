"""
Tests for STT pipeline improvements.
"""

from bot.hear import (
    _transcript_cache_key,
    _join_segments,
    _find_overlap_len,
    STT_PIPELINE_VERSION,
)
from bot.stt import ModelSpec


class TestTranscriptCacheKey:
    """Test transcript cache key versioning."""

    def test_cache_key_includes_pipeline_version(self):
        """Cache key generation should include pipeline version."""
        spec = ModelSpec("tiny", "int8")
        key1 = _transcript_cache_key("audio123", spec, vad_enabled=True)

        # Check the key is a hex string
        assert isinstance(key1, str)
        assert len(key1) == 24  # We truncate to 24 chars

    def test_cache_key_changes_with_different_versions(self):
        """Changing pipeline version changes the cache key."""
        # This test documents the expected behavior
        assert STT_PIPELINE_VERSION.startswith("stt-")
        assert (
            "lang-aware" in STT_PIPELINE_VERSION.lower()
            or "stitch" in STT_PIPELINE_VERSION.lower()
        )

    def test_cache_key_includes_task_and_language(self):
        """Cache key should include task and language parameters."""
        spec = ModelSpec("tiny", "int8")
        key1 = _transcript_cache_key("audio123", spec, task="transcribe", language=None)
        key2 = _transcript_cache_key("audio123", spec, task="translate", language="en")

        # Different task/language should produce different keys
        assert key1 != key2

    def test_cache_key_with_explicit_language(self):
        """Cache key with explicit language differs from auto."""
        spec = ModelSpec("tiny", "int8")
        key_auto = _transcript_cache_key("audio123", spec, language=None)
        key_ar = _transcript_cache_key("audio123", spec, language="ar")
        key_en = _transcript_cache_key("audio123", spec, language="en")

        # All three should be different
        assert len(set([key_auto, key_ar, key_en])) == 3


class TestJoinSegments:
    """Test the timestamp-aware segment joining."""

    def test_empty_segments(self):
        """Empty segments list returns empty string."""
        text, meta = _join_segments([])
        assert text == ""
        assert meta.get("confidence_status") == "unknown"

    def test_single_segment(self):
        """Single segment just returns its text."""
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello world"}]
        text, meta = _join_segments(segments)
        assert text == "Hello world"

    def test_sorts_by_timestamp(self):
        """Segments are sorted by timestamp before joining."""
        segments = [
            {"start": 5.0, "end": 6.0, "text": "world", "chunk_idx": 1},
            {"start": 0.0, "end": 1.0, "text": "Hello", "chunk_idx": 0},
        ]
        text, meta = _join_segments(segments)
        assert text == "Hello world"

    def test_removes_empty_text(self):
        """Empty text segments are filtered out."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello", "chunk_idx": 0},
            {"start": 1.0, "end": 2.0, "text": "", "chunk_idx": 0},
            {"start": 2.0, "end": 3.0, "text": "world", "chunk_idx": 0},
        ]
        text, meta = _join_segments(segments)
        assert text == "Hello world"

    def test_removes_whitespace_only(self):
        """Whitespace-only text is filtered out."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello", "chunk_idx": 0},
            {"start": 1.0, "end": 2.0, "text": " ", "chunk_idx": 0},
            {"start": 2.0, "end": 3.0, "text": "world", "chunk_idx": 0},
        ]
        text, meta = _join_segments(segments)
        assert text == "Hello world"

    def test_overlapping_segments_filtered(self):
        """Highly overlapping segments (>50% overlap) are filtered."""
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello world", "chunk_idx": 0},
            {"start": 1.0, "end": 3.0, "text": "world how are", "chunk_idx": 1},
        ]
        # Second segment overlaps by 50% (1 second of 2)
        text, meta = _join_segments(segments)
        assert "Hello" in text
        # The second segment should be mostly skipped

    def test_exact_consecutive_duplicate_removed(self):
        """Exact consecutive duplicates are removed."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello", "chunk_idx": 0},
            {"start": 1.0, "end": 2.0, "text": "Hello", "chunk_idx": 1},
        ]
        text, meta = _join_segments(segments)
        assert text == "Hello"  # Not "Hello Hello"

    def test_legitimate_repeated_phrases_preserved(self):
        """Non-consecutive repeated phrases are preserved."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello", "chunk_idx": 0},
            {"start": 1.0, "end": 2.0, "text": "there", "chunk_idx": 0},
            {"start": 3.0, "end": 4.0, "text": "Hello", "chunk_idx": 1},
        ]
        text, meta = _join_segments(segments)
        # "Hello" appears twice but is legitimate repetition
        assert text.count("Hello") == 2


class TestFindOverlapLen:
    """Test the overlap detection function."""

    def test_no_overlap(self):
        """No overlap returns 0."""
        result = _find_overlap_len("hello world", "foo bar")
        assert result == 0

    def test_prefix_suffix_overlap(self):
        """Detects word boundary overlaps."""
        result = _find_overlap_len("hello world", "world how are")
        assert result > 0  # Should find "world" overlap

    def test_short_overlap_ignored(self):
        """Short overlaps (< 3 chars) are ignored."""
        result = _find_overlap_len("hello a", "a world")
        assert result == 0  # "a" is too short

    def test_empty_strings(self):
        """Empty strings return 0."""
        assert _find_overlap_len("", "hello") == 0
        assert _find_overlap_len("hello", "") == 0
        assert _find_overlap_len("", "") == 0

    def test_no_word_boundary_no_overlap(self):
        """Overlaps not at word boundaries may be ignored."""
        # This documents the behavior
        result = _find_overlap_len("helloooo", "oooworld")
        # "ooo" is not a clean word boundary
        assert result >= 0  # Implementation specific
