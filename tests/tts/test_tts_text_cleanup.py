"""TTS text cleanup: stretched words can't produce unsplittable chunks. [REH]"""

import pytest

from bot.tts.interface import MAX_TTS_WORD_CHARS, MAX_WORDS_PER_SENTENCE, TTSManager


@pytest.fixture
def clean():
    return TTSManager._clean_text.__get__(object.__new__(TTSManager))


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("Nnnnnnnnnnnnnnniiiiiiiiiiiiiiiiiiiii", "Nnii"),  # the 23:17 timeout
        ("Pooooooooooooooooooooooooo", "Poo"),
        ("hahahahahahahaha lmaooooo", "haha lmaoo"),
        ("in 2000 it cost 1999", "in 2000 it cost 1999"),  # digits untouched
        ("bookkeeper committee mississippi bananas", "bookkeeper committee mississippi bananas"),
    ],
)
def test_collapses_stretched_words_only(clean, raw, spoken):
    assert clean(raw) == spoken


def test_caps_unsplittable_tokens(clean):
    out = clean("ok " + "x1y2" * 30)
    assert max(len(w) for w in out.split()) <= MAX_TTS_WORD_CHARS


def test_drops_scripts_pocket_cannot_speak(clean):
    # 2026-10-24: Heart Sutra in Chinese -> one 815-token chunk -> timeout -> text reply
    assert clean("觀自在菩薩 行深般若波羅蜜多時 照見五蘊皆空") == ""
    assert clean("here you go: 觀自在菩薩 行深般若") == "here you go:"
    assert clean("café naïve résumé — ok") == "café naïve résumé — ok"  # Latin accents stay


def test_breaks_unpunctuated_runs(clean):
    # pinyin with no full stops was one 867-token chunk
    pinyin = " ".join(["guān zì zài pú sà xíng shēn bō rě"] * 12)
    out = clean(pinyin)
    for sentence in out.split(". "):
        assert len(sentence.split()) <= MAX_WORDS_PER_SENTENCE
    assert out.replace(". ", " ").split() == pinyin.split()  # no words lost
    assert clean("short one. another short one!") == "short one. another short one!"
