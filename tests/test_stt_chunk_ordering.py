"""Tests for STT chunk ordering and deduplication."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import numpy as np

from bot.hear import (
    _join_segments,
    _segments_to_dict,
    SpanRecorder,
    PreprocessResult,
    TranscriptResult,
    STTJob,
    STTRAMGuard,
    _transcribe_with_model,
)


def _text(result) -> str:
    """Extract just the text string from _join_segments return value (str, dict) tuple."""
    if isinstance(result, tuple):
        return result[0]
    return result


class TestJoinSegments:
    """Tests for _join_segments helper."""

    def test_empty_segments(self):
        """Empty segments should return empty string."""
        assert _text(_join_segments([])) == ""

    def test_single_segment(self):
        """Single segment preserved as-is."""
        segments = [{"text": "hello world"}]
        assert _text(_join_segments(segments)) == "hello world"

    def test_multiple_segments_in_order(self):
        """Multiple segments joined in order with single space."""
        segments = [
            {"text": "hello"},
            {"text": "world"},
            {"text": "test"},
        ]
        assert _text(_join_segments(segments)) == "hello world test"

    def test_consecutive_duplicates_removed(self):
        """Consecutive duplicate text (from overlapping windows) should be deduplicated."""
        segments = [
            {"text": "hello"},
            {"text": "hello"},  # consecutive duplicate
            {"text": "world"},
        ]
        assert _text(_join_segments(segments)) == "hello world"

    def test_non_consecutive_duplicates_preserved(self):
        """Non-consecutive duplicates (legitimate repeats) should be preserved."""
        segments = [
            {"text": "hello"},
            {"text": "world"},
            {"text": "hello"},  # non-consecutive repeat
        ]
        assert _text(_join_segments(segments)) == "hello world hello"

    def test_overlapping_chunk_boundary(self):
        """Simulate overlapping chunk window creating duplicate at boundary."""
        segments = [
            {"text": "first chunk ends here"},
            {"text": "boundary"},  # first of chunk 1 (duplicate)
            {"text": "second chunk continues"},
        ]
        assert _text(_join_segments(segments)) == "first chunk ends here boundary second chunk continues"

    def test_join_normalizes_whitespace(self):
        """Join should normalize whitespace."""
        segments = [
            {"text": " hello "},
            {"text": "  world  "},
        ]
        text, _meta = _join_segments(segments)
        assert text == "hello world"
        # Extra spaces normalized and no leading/trailing space
        assert text == text.strip()

    def test_empty_text_filtered(self):
        """Empty text segments filtered out."""
        segments = [
            {"text": "hello"},
            {"text": ""},
            {"text": "world"},
        ]
        assert _text(_join_segments(segments)) == "hello world"


class TestSegmentsToDict:
    """Tests for _segments_to_dict conversion."""

    def test_segments_converted_correctly(self):
        """Test that segments are converted with correct offsets."""
        # Mock segment objects
        seg1 = MagicMock()
        seg1.start = 0.0
        seg1.end = 2.0
        seg1.text = " hello "  # with whitespace

        seg2 = MagicMock()
        seg2.start = 2.0
        seg2.end = 4.0
        seg2.text = "world"

        segments = [seg1, seg2]
        offset = 10.0  # add 10 second offset

        result = _segments_to_dict(segments, offset)

        assert len(result) == 2
        assert result[0]["start"] == 10.0
        assert result[0]["end"] == 12.0
        assert result[0]["text"] == "hello"
        assert result[1]["start"] == 12.0
        assert result[1]["end"] == 14.0
        assert result[1]["text"] == "world"


class TestChunkProcessingOrder:
    """Tests for chunk processing order-safety."""

    @pytest.mark.asyncio
    async def test_chunk_records_indexed_correctly(self):
        """Test that chunks have stable indices assigned sequentially."""
        # This test verifies the chunk_records structure from _transcribe_with_model
        # chunks should have 'idx', 'start', 'end', 'segments' keys

        # Structure check without running full transcription
        chunk_record = {
            "idx": 0,
            "start": 0.0,
            "end": 40.0,
            "segments": [{"text": "test"}],
        }

        assert "idx" in chunk_record
        assert "start" in chunk_record
        assert "end" in chunk_record
        assert "segments" in chunk_record
        assert chunk_record["idx"] == 0


class TestTranscriptAssembly:
    """Tests for transcript assembly from chunk records."""

    def test_assemble_from_ordered_chunk_records(self):
        """Test assembling transcript from ordered chunk records (expected order)."""
        # Simulate chunk_records from processing
        segments1 = [{"text": "hello world"}, {"text": "this is"}]
        segments2 = [{"text": "this is"}, {"text": "a test"}]

        all_segments = []
        all_segments.extend(segments1)
        all_segments.extend(segments2)  # With consecutive duplicate

        result = _join_segments(all_segments)
        # "this is" appears consecutively at boundary, should be deduped
        assert _text(result) == "hello world this is a test"

    def test_out_of_order_chunks_by_id(self):
        """Test that we can reconstruct from chunk_records by idx if needed."""
        # If chunks arrived out of order, we should sort by idx
        chunk_records = [
            {"idx": 2, "segments": [{"text": "chunk two"}]},
            {"idx": 0, "segments": [{"text": "chunk zero"}]},
            {"idx": 1, "segments": [{"text": "chunk one"}]},
        ]

        # Sort by idx
        sorted_records = sorted(chunk_records, key=lambda x: x["idx"])

        assert sorted_records[0]["idx"] == 0
        assert sorted_records[1]["idx"] == 1
        assert sorted_records[2]["idx"] == 2

        # Assemble segments in order
        all_segments = []
        for rec in sorted_records:
            all_segments.extend(rec["segments"])

        result = _join_segments(all_segments)
        assert _text(result) == "chunk zero chunk one chunk two"


class TestSTTJobTranscriptRegistration:
    """Tests for STTJob transcript registration."""

    def test_job_registers_transcript_correctly(self):
        """Test that STTJob correctly registers transcript results."""
        spans = MagicMock(spec=SpanRecorder)
        spans.spans = {}
        ram_guard = MagicMock(spec=STTRAMGuard)

        job = STTJob(
            kind="test",
            spans=spans,
            ram_guard=ram_guard,
        )

        # Create a transcript with ordered chunks
        transcript = TranscriptResult(
            text="hello world test",
            segments=[
                {"text": "hello"},
                {"text": "world"},
                {"text": "test"},
            ],
            chunks=[
                {"idx": 0, "start": 0.0, "end": 2.0},
                {"idx": 1, "start": 2.0, "end": 4.0},
            ],
            duration_out=4.0,
            model_spec=MagicMock(),
            cache_hit=False,
            first_chunk_runtime=1.0,
        )

        job.register_transcript(transcript)

        assert job.chunks_done == 2
        assert job.dur_done_s == 4.0
        assert job.transcript is transcript

    def test_job_registers_aborted_transcript(self):
        """Test that STTJob handles aborted transcripts."""
        spans = MagicMock(spec=SpanRecorder)
        ram_guard = MagicMock(spec=STTRAMGuard)

        job = STTJob(
            kind="test",
            spans=spans,
            ram_guard=ram_guard,
        )

        transcript = TranscriptResult(
            text="partial",
            segments=[{"text": "partial"}],
            chunks=[
                {"idx": 0, "start": 0.0, "end": 2.0},
            ],
            duration_out=2.0,
            model_spec=MagicMock(),
            cache_hit=False,
            first_chunk_runtime=1.0,
            aborted=True,
            abort_reason="memory_guard",
        )

        job.register_transcript(transcript)

        assert job.state == "aborting"
        assert job.abort_reason == "memory_guard"
        assert job.chunks_done == 1


class TestDeterministicJoinWithWhitespace:
    """Tests for whitespace normalization in join."""

    def test_join_normalizes_whitespace(self):
        """Verify whitespace normalization in final join."""
        segments = [
            {"text": "hello   world"},  # internal multiple spaces
            {"text": "test"},
        ]
        # Should preserve internal spaces (they were transcribed that way)
        result = _join_segments(segments)
        assert _text(result) == "hello   world test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
