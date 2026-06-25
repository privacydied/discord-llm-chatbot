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
        (
            "A: false\nB: false\nMODE: NORMAL\n\nactual answer",
            "actual answer",
        ),
        (
            "A: false\r\nB: false\r\nMODE: NORMAL\r\n\r\nactual answer",
            "actual answer",
        ),
        (
            "A: false\nB: false\nMODE: POLITICAL\nactual answer",
            "actual answer",
        ),
        (
            "A: false\nB: false\nMODE: CONTRADICTION\n\nactual answer",
            "actual answer",
        ),
        (
            "A: true\nB: false\nMODE: NORMAL\n\nactual answer",
            "actual answer",
        ),
        (
            "a: false\nb: false\nmode: normal\n\nactual answer",
            "actual answer",
        ),
        (
            "**A: false**\n**B: false**\n**MODE: NORMAL**\n\nactual answer",
            "actual answer",
        ),
        (
            "`A: false`\n`B: false`\n`MODE: NORMAL`\n\nactual answer",
            "actual answer",
        ),
        (
            "> A: false\n> B: false\n> MODE: NORMAL\n\nactual answer",
            "actual answer",
        ),
        (
            "A: false B: false MODE: NORMAL\n\nactual answer",
            "actual answer",
        ),
        (
            "A: false B: false MODE: POLITICAL actual answer",
            "actual answer",
        ),
        (
            "A: false\nB: false\nMODE: CONTRADICTION",
            "A: false\nB: false",
        ),
    ],
)
def test_strip_leading_mode_preamble(raw: str, expected: str) -> None:
    assert strip_leading_mode_preamble(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "normal mode in vim is useful",
        "mode: normal is the setting you asked about",
        "hello\nMODE: NORMAL",
        "the model said MODE: NORMAL yesterday",
        "A: false\nB: false\nbecause both statements are false",
        "A: false\nB: false",
        "A: false B: false",
        "MODE: normal is the setting you asked about",
        "A: apples\nB: bananas\nMODE: recipe",
    ],
)
def test_preserve_non_diagnostic_text(raw: str) -> None:
    assert strip_leading_mode_preamble(raw) == raw
