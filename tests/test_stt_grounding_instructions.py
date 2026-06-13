"""Tests for STT grounding instructions to prevent "I can't process audio" responses."""

import re

from bot.router_components.compose import format_x_tweet_with_transcription


def test_stt_grounding_instructions_added_when_transcript_present() -> None:
    """When STT transcript is present, grounding instructions should be added."""
    result = format_x_tweet_with_transcription(
        base_text=None,
        url="https://x.com/user/status/12345",
        stt_res={"transcription": "This is a transcript of the audio."},
        tweet_data=None,
    )

    # Should contain STT grounding instructions
    assert "[STT GROUNDING]" in result
    assert "transcribed by STT" in result
    assert "Use the transcript above as the source" in result
    assert "do NOT claim the audio cannot be processed" in result


def test_stt_grounding_instructions_in_caption_transcript_case() -> None:
    """Grounding instructions should be added for caption+transcript combinations."""
    result = format_x_tweet_with_transcription(
        base_text=None,
        url="https://x.com/user/status/12345",
        stt_res={"transcription": "Audio transcript text here."},
        tweet_data={"full_text": "Tweet caption here."},
    )

    # Should contain the combined caption_transcript section
    assert "[Tweet Caption + Audio Transcript]" in result
    # Should also contain STT grounding
    assert "[STT GROUNDING]" in result
    assert "transcribed by STT" in result


def test_no_grounding_when_empty_transcript() -> None:
    """Grounding instructions should not be added for empty transcripts."""
    result = format_x_tweet_with_transcription(
        base_text=None,
        url="https://x.com/user/status/12345",
        stt_res={"transcription": ""},
        tweet_data=None,
    )

    # Should NOT contain STT grounding for empty transcripts
    assert "[STT GROUNDING]" not in result


def test_no_grounding_for_caption_only() -> None:
    """Grounding instructions should not be added when only caption is present (no transcript)."""
    result = format_x_tweet_with_transcription(
        base_text=None,
        url="https://x.com/user/status/12345",
        stt_res={},  # No transcript
        tweet_data={"full_text": "Just a tweet caption."},
    )

    # Should NOT contain STT grounding without transcript
    assert "[STT GROUNDING]" not in result


def test_tuple_transcript_handled_correctly() -> None:
    """Tuple-shaped transcription should be handled with grounding added."""
    result = format_x_tweet_with_transcription(
        base_text=None,
        url="https://x.com/user/status/12345",
        stt_res={"transcription": ("Tuple transcript text", {"confidence": 0.95})},
        tweet_data=None,
    )

    # Should extract text from tuple
    assert "Tuple transcript text" in result
    # Should add grounding instructions
    assert "[STT GROUNDING]" in result


def test_grounding_includes_key_instructions() -> None:
    """Grounding should include all critical instructions to prevent 'cannot access' responses."""
    result = format_x_tweet_with_transcription(
        base_text=None,
        url="https://x.com/user/status/12345",
        stt_res={"transcription": "Test transcript."},
        tweet_data=None,
    )

    # Check for all required grounding elements
    assert "transcribed by STT" in result
    assert "Use the transcript above as the source" in result
    assert "do NOT claim the audio cannot be processed" in result
    assert "cannot access it" in result  # Specific phrase to avoid
    assert "translate" in result.lower()
    assert "Do not ask the user to provide the audio" in result


def test_grounding_included_as_instruction_section() -> None:
    """Grounding should be added as an instruction section in the evidence."""
    result = format_x_tweet_with_transcription(
        base_text=None,
        url="https://x.com/user/status/12345",
        stt_res={"transcription": "Test transcript."},
        tweet_data=None,
    )

    # Should have the instruction section
    assert "[STT Instructions]" in result
    grounding_match = re.search(r"\[STT Instructions\]\n(.+?)(?=\n\[|$)", result, re.DOTALL)
    assert grounding_match is not None


def test_grounding_preserves_non_english_transcript() -> None:
    """Grounding should work with non-English transcripts (Arabic, etc.)."""
    result = format_x_tweet_with_transcription(
        base_text=None,
        url="https://x.com/user/status/12345",
        stt_res={"transcription": "الرئيس جوبايدا بصحب الزلالة والسمو"},
        tweet_data=None,
    )

    # Should preserve Arabic text
    assert "الرئيس" in result
    # Should add grounding
    assert "[STT GROUNDING]" in result
    # Should include translation instruction
    assert "For non-English transcripts" in result
