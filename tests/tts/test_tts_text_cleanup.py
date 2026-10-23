"""TTS text cleanup: stretched words can't produce unsplittable chunks. [REH]"""

import pytest

from bot.tts.interface import MAX_TTS_WORD_CHARS, TTSManager


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
