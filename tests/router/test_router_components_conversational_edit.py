"""Unit tests for bot/router_components/conversational_edit.py."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.modality import ImageRef
from bot.router_components import conversational_edit as ce


# ---------------------------------------------------------------------------
# classify_edit_intent
# ---------------------------------------------------------------------------


class TestClassifyEditIntent:
    def test_empty_text_is_not_edit(self) -> None:
        assert ce.classify_edit_intent("").is_edit is False
        assert ce.classify_edit_intent("   ").is_edit is False

    def test_repro_case_from_bug_report_is_edit(self) -> None:
        result = ce.classify_edit_intent("give this man a tasteful Norwood 4")
        assert result.is_edit is True
        assert result.matched_phrase == "give this"

    @pytest.mark.parametrize(
        "text",
        [
            "make it black and white",
            "remove the background",
            "add a hat to him",
            "turn this into a painting",
            "can you edit this photo",
            "put a mustache on him",
        ],
    )
    def test_edit_instructions_detected(self, text: str) -> None:
        assert ce.classify_edit_intent(text).is_edit is True

    @pytest.mark.parametrize(
        "text",
        [
            "what is this",
            "describe this image",
            "who is this guy",
            "is this a cat?",
            "analyze this picture",
            "can you tell me about this",
        ],
    )
    def test_analysis_questions_are_not_edit(self, text: str) -> None:
        assert ce.classify_edit_intent(text).is_edit is False

    def test_ambiguous_question_with_edit_verb_prefers_analysis(self) -> None:
        # Contains "remove" (edit verb) but reads as a question -> analysis wins.
        result = ce.classify_edit_intent("what would this look like if you removed the hat?")
        assert result.is_edit is False

    def test_extra_keywords_merge_with_defaults(self) -> None:
        result = ce.classify_edit_intent("norwoodify this guy", extra_keywords="norwoodify")
        assert result.is_edit is True
        assert result.matched_phrase == "norwoodify"

    def test_missing_policy_file_falls_back_to_defaults(self) -> None:
        ce._load_policy_trigger_phrases.cache_clear()
        result = ce.classify_edit_intent("please remove the logo", policy_path="/nonexistent/vision_policy.json")
        assert result.is_edit is True


# ---------------------------------------------------------------------------
# resolve_edit_source_image
# ---------------------------------------------------------------------------


def _fake_message(*, content="", attachments=True, reference=None):
    msg = SimpleNamespace()
    msg.content = content
    msg.attachments = ["fake"] if attachments else []
    msg.embeds = []
    msg.reference = reference
    msg.channel = SimpleNamespace(fetch_message=AsyncMock())
    return msg


@pytest.mark.asyncio
class TestResolveEditSourceImage:
    async def test_current_message_attachment_wins(self, monkeypatch) -> None:
        current_ref = ImageRef(url="https://cdn.example/current.png", content_type="image/png")
        msg = _fake_message()

        def fake_collect(m):
            return [current_ref] if m is msg else []

        monkeypatch.setattr(ce, "collect_image_urls_from_message", fake_collect)

        async def fake_download(ref, local_path, max_size_mb=25):
            with open(local_path, "wb") as f:
                f.write(b"png-bytes")
            return True

        monkeypatch.setattr(ce, "download_robust_image", fake_download)

        resolved = await ce.resolve_edit_source_image(msg, max_size_mb=25)
        assert resolved is not None
        assert resolved.source == "current"
        assert resolved.data == b"png-bytes"

    async def test_falls_back_to_referenced_message_attachment(self, monkeypatch) -> None:
        ref_msg = SimpleNamespace()
        reply_ref = ImageRef(url="https://cdn.example/reply.png", content_type="image/png")

        msg = _fake_message(attachments=False, reference=SimpleNamespace(message_id=555))
        msg.channel.fetch_message = AsyncMock(return_value=ref_msg)

        def fake_collect(m):
            if m is ref_msg:
                return [reply_ref]
            return []

        monkeypatch.setattr(ce, "collect_image_urls_from_message", fake_collect)

        async def fake_download(ref, local_path, max_size_mb=25):
            with open(local_path, "wb") as f:
                f.write(b"reply-bytes")
            return True

        monkeypatch.setattr(ce, "download_robust_image", fake_download)

        resolved = await ce.resolve_edit_source_image(msg, max_size_mb=25)
        assert resolved is not None
        assert resolved.source == "reply"
        assert resolved.data == b"reply-bytes"

    async def test_falls_back_to_bare_url_in_text(self, monkeypatch) -> None:
        msg = _fake_message(content="check this out https://example.com/pic.jpg", attachments=False)

        monkeypatch.setattr(ce, "collect_image_urls_from_message", lambda m: [])

        async def fake_download(ref, local_path, max_size_mb=25):
            with open(local_path, "wb") as f:
                f.write(b"url-bytes")
            return True

        monkeypatch.setattr(ce, "download_robust_image", fake_download)

        resolved = await ce.resolve_edit_source_image(msg, max_size_mb=25)
        assert resolved is not None
        assert resolved.source == "url"
        assert resolved.data == b"url-bytes"

    async def test_no_image_anywhere_returns_none(self, monkeypatch) -> None:
        msg = _fake_message(content="just some text", attachments=False)
        monkeypatch.setattr(ce, "collect_image_urls_from_message", lambda m: [])

        resolved = await ce.resolve_edit_source_image(msg, max_size_mb=25)
        assert resolved is None

    async def test_size_limit_enforced_via_download_robust_image(self, monkeypatch) -> None:
        """download_robust_image is the size-limit enforcement point; verify we
        pass max_size_mb through and honor a rejection (oversize) result."""
        current_ref = ImageRef(url="https://cdn.example/big.png", content_type="image/png")
        msg = _fake_message()

        monkeypatch.setattr(ce, "collect_image_urls_from_message", lambda m: [current_ref])

        captured = {}

        async def fake_download(ref, local_path, max_size_mb=25):
            captured["max_size_mb"] = max_size_mb
            return False  # simulate oversize rejection

        monkeypatch.setattr(ce, "download_robust_image", fake_download)

        resolved = await ce.resolve_edit_source_image(msg, max_size_mb=7)
        assert resolved is None
        assert captured["max_size_mb"] == 7


class TestEditPhraseWordBoundaries:
    """Trigger phrases must match whole words, not substrings. [IV]"""

    @pytest.mark.parametrize(
        "text",
        [
            "i gave him credit for that",
            "saw it on reddit earlier",
            "the editor approved it",
            "this is the extended edition",
            "nothing changed since the exchange",
            "prefix the filename please",
            "he removed himself from the group",
            "unaltered footage",
            "the fixture is broken",
        ],
    )
    def test_substring_matches_are_not_edits(self, text: str) -> None:
        assert ce.classify_edit_intent(text).is_edit is False


class TestParseEditFlags:
    """parse_edit_flags: -seed/-steps/-strength/-guidance/-negative/-provider/
    -use/-h extraction, bounds-checked against /imgedit's own ranges. [IV]
    """

    def test_no_flags_returns_prompt_unchanged(self) -> None:
        result = ce.parse_edit_flags("make him a superhero")
        assert result.prompt == "make him a superhero"
        assert result.seed is None
        assert result.steps is None
        assert result.strength is None
        assert result.guidance is None
        assert result.negative is None
        assert result.provider is None
        assert result.model is None
        assert result.help_requested is False
        assert result.errors == ()

    def test_all_flags_parsed_and_stripped_from_prompt(self) -> None:
        result = ce.parse_edit_flags(
            "make him a superhero -seed 7 -steps 20 -strength 0.6 -guidance 9.5 -provider together -use some/model"
        )
        assert result.prompt == "make him a superhero"
        assert result.seed == 7
        assert result.steps == 20
        assert result.strength == 0.6
        assert result.guidance == 9.5
        assert result.provider == "together"
        assert result.model == "some/model"
        assert result.errors == ()

    def test_flags_interspersed_with_prompt_text_still_extracted(self) -> None:
        result = ce.parse_edit_flags("make -steps 15 him a superhero")
        assert result.prompt == "make him a superhero"
        assert result.steps == 15

    def test_negative_flag_supports_quoted_multiword_value(self) -> None:
        result = ce.parse_edit_flags('make him a superhero -negative "blurry, low quality"')
        assert result.prompt == "make him a superhero"
        assert result.negative == "blurry, low quality"

    def test_unbalanced_quotes_falls_back_to_whitespace_split(self) -> None:
        # shlex.split would raise ValueError on the stray quote; must not crash
        # or eat the whole prompt.
        result = ce.parse_edit_flags('make him a superhero "quote')
        assert "superhero" in result.prompt

    @pytest.mark.parametrize(
        ("flag_text", "expected_substr"),
        [
            ("-seed notanumber", "-seed"),
            ("-steps 5", "-steps"),  # below STEPS_RANGE minimum (10)
            ("-steps 100", "-steps"),  # above STEPS_RANGE maximum (50)
            ("-strength 0", "-strength"),  # below STRENGTH_RANGE minimum (0.1)
            ("-strength 2.0", "-strength"),  # above STRENGTH_RANGE maximum (1.0)
            ("-guidance 0.5", "-guidance"),  # below GUIDANCE_RANGE minimum (1.0)
            ("-guidance 25", "-guidance"),  # above GUIDANCE_RANGE maximum (20.0)
            ("-provider bogus", "-provider"),
        ],
    )
    def test_out_of_range_or_invalid_values_produce_errors(self, flag_text: str, expected_substr: str) -> None:
        result = ce.parse_edit_flags(f"prompt {flag_text}")
        assert result.errors, f"expected an error for {flag_text!r}"
        assert any(expected_substr in e for e in result.errors)

    def test_flag_missing_value_produces_error(self) -> None:
        result = ce.parse_edit_flags("make him a superhero -steps")
        assert result.errors
        assert "-steps" in result.errors[0]

    @pytest.mark.parametrize("help_text", ["-h", "--help"])
    def test_help_flag_detected(self, help_text: str) -> None:
        result = ce.parse_edit_flags(f"{help_text} make him a superhero")
        assert result.help_requested is True

    def test_steps_at_range_boundaries_accepted(self) -> None:
        assert ce.parse_edit_flags("prompt -steps 10").steps == 10
        assert ce.parse_edit_flags("prompt -steps 50").steps == 50

    def test_strength_at_range_boundaries_accepted(self) -> None:
        assert ce.parse_edit_flags("prompt -strength 0.1").strength == 0.1
        assert ce.parse_edit_flags("prompt -strength 1.0").strength == 1.0

    def test_provider_case_insensitive(self) -> None:
        assert ce.parse_edit_flags("prompt -provider NOVITA").provider == "novita"

    @pytest.mark.parametrize(
        ("text", "phrase"),
        [
            ("edit this to be brighter", "edit"),
            ("please fix the lighting", "fix"),
            ("change the sky to purple", "change"),
            ("extend the canvas on the left", "extend"),
        ],
    )
    def test_whole_word_verbs_still_fire(self, text: str, phrase: str) -> None:
        result = ce.classify_edit_intent(text)
        assert result.is_edit is True
        assert result.matched_phrase == phrase

    def test_longest_phrase_wins_for_reporting(self) -> None:
        result = ce.classify_edit_intent("make it look like a renaissance painting")
        assert result.is_edit is True
        assert result.matched_phrase == "make it look like"
