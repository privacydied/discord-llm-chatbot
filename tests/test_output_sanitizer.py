from __future__ import annotations

import pytest

from bot.utils.output_sanitizer import strip_leading_mode_preamble


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MODE: NORMAL\n\nhello", "hello"),
        ("MODE: POLITICAL\nhello", "hello"),
        ("mode: normal\nhello", "hello"),
        ("Mode: Political\n\nhello", "hello"),
        ("**MODE: NORMAL**\nhello", "hello"),
        ("`MODE: NORMAL`\nhello", "hello"),
        ("> MODE: NORMAL\nhello", "hello"),
        ("MODE: NORMAL\nMODE: NORMAL\nhello", "hello"),
        ("\ufeffMODE: NORMAL\nhello", "hello"),
        ("normal mode in vim is useful", "normal mode in vim is useful"),
        ("mode: normal is the setting you asked about", "mode: normal is the setting you asked about"),
        ("hello\nMODE: NORMAL", "hello\nMODE: NORMAL"),
        ("the model said MODE: NORMAL yesterday", "the model said MODE: NORMAL yesterday"),
        ("", ""),
    ],
)
def test_strip_leading_mode_preamble(raw: str, expected: str) -> None:
    assert strip_leading_mode_preamble(raw) == expected
