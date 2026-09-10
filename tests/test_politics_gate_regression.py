"""Regression test for the politics gate fix.

Verifies the assembled v9 prompt implements three-behavior routing
(NORMAL / TOPICAL / REQUESTED_LENS) and does not carry over the old
blanket-apology word-blacklist / 'pure mechanisms only' restrictions
that caused 'what were hitlers politics' to be refused.
"""

from __future__ import annotations

from pathlib import Path

PROMPT_PATH = Path("prompts/prompt-pry-super-chill-v9.txt")


def _load_prompt() -> str:
    text = PROMPT_PATH.read_text()
    assert len(text) > 100, "Prompt file suspiciously short"
    return text


class TestPoliticsGateStructure:
    """The prompt must contain the new gate and not the old one."""

    def test_old_mode_gate_removed(self) -> None:
        text = _load_prompt()
        assert "MODE GATE (HARD)" not in text, "Old MODE GATE still present"
        assert "MODE = " not in text, "Old MODE assignment still present"

    def test_old_forbidden_word_blacklist_removed(self) -> None:
        text = _load_prompt()
        assert "forbidden in normal mode" not in text.lower(), "Old forbidden-word blacklist still present"
        # The old NORMAL section listed specific words as forbidden. Verify
        # that exact blacklist block is gone by checking for the section header
        # + the first entry together.
        assert "forbidden in normal mode (never mention, even joking):" not in text
        assert "landlord(s), working-class, protest" not in text, "Old blacklist word list still present"

    def test_pure_mechanisms_removed(self) -> None:
        text = _load_prompt()
        assert "pure mechanisms only" not in text, "Old 'pure mechanisms only' restriction still present"

    def test_new_gate_present(self) -> None:
        text = _load_prompt()
        assert "POLITICS GATE (INTERNAL ONLY)" in text
        assert "EXPLICIT_LENS_REQUEST" in text
        assert "POLITICS_RELEVANT" in text

    def test_three_behaviors_present(self) -> None:
        text = _load_prompt()
        assert "NORMAL behavior" in text
        assert "TOPICAL behavior" in text
        assert "REQUESTED_LENS behavior" in text

    def test_normal_behavior_relevance_not_prohibition(self) -> None:
        text = _load_prompt()
        # NORMAL behavior must say "do not introduce unsolicited" — not "never mention"
        assert "do not introduce unsolicited political framing" in text, "NORMAL should be a relevance instruction, not a blanket ban"

    def test_topical_answers_directly(self) -> None:
        text = _load_prompt()
        assert "answer political, historical, economic, policy, or ideological questions directly" in text
        assert "do not refuse to answer a political question" in text

    def test_requested_lens_applies_explicitly(self) -> None:
        text = _load_prompt()
        assert "apply that specific perspective" in text
        assert "distinguish the requested interpretation from established facts" in text

    def test_political_reference_access_condition_updated(self) -> None:
        text = _load_prompt()
        # New access condition: use only in TOPICAL or REQUESTED_LENS
        assert "use only in TOPICAL or REQUESTED_LENS behavior" in text
        # Old "READ ONLY IF MODE == POLITICAL" must be gone
        assert 'MODE == "POLITICAL"' not in text, "Old POLITICAL REFERENCE access condition still present"

    def test_fetched_content_rule_preserved(self) -> None:
        text = _load_prompt()
        assert "MANDATORY: if the message you" + "'re replying to includes" in text
        assert "I processed N input(s) from your message" in text

    def test_no_contradictory_apolitical_instructions(self) -> None:
        text = _load_prompt()
        # The old NORMAL said "stay fully apolitical. no politics" — that exact
        # phrase must not exist alongside the new TOPICAL that answers directly.
        assert "stay fully apolitical" not in text

    def test_internal_gate_not_announced(self) -> None:
        text = _load_prompt()
        # The gate must tell the model never to narrate the decision
        assert "never narrate, quote, summarize, or reference this decision procedure" in text
