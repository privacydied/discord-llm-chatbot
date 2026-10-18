"""Tests for persona lowercase enforcement [IV][REH]."""

import pytest

from bot import brain
from bot.core.text_case import lowercase_outside_protected


def test_lowercases_plain_text():
    assert lowercase_outside_protected("Kendrick WON the Beef. I Think So.") == "kendrick won the beef. i think so."


def test_preserves_url_case():
    text = "Watch This https://youtu.be/89U_eXacTZ0?si=X-iRWhl_ Now"
    assert lowercase_outside_protected(text) == "watch this https://youtu.be/89U_eXacTZ0?si=X-iRWhl_ now"


def test_preserves_code_blocks_and_inline_code():
    text = "Run `ls -L` Then:\n```py\nPRINT = True\n```\nDone"
    assert lowercase_outside_protected(text) == "run `ls -L` then:\n```py\nPRINT = True\n```\ndone"


def test_preserves_discord_tokens():
    text = "Ayo <@1392310898311303259> <:PepeLaugh:123> <a:Spin:456> <t:1700000000:R> OK"
    expected = "ayo <@1392310898311303259> <:PepeLaugh:123> <a:Spin:456> <t:1700000000:R> ok"
    assert lowercase_outside_protected(text) == expected


def test_empty_text():
    assert lowercase_outside_protected("") == ""


@pytest.mark.parametrize(("flag", "expected"), [(None, "nah, aave"), ("false", "Nah, AAVE")])
async def test_brain_infer_applies_flag(monkeypatch, flag, expected):
    async def fake_generate_response(**_kwargs):
        return {"text": "Nah, AAVE"}

    monkeypatch.setattr(brain, "generate_response", fake_generate_response)
    if flag is None:
        monkeypatch.delenv("FORCE_LOWERCASE_REPLIES", raising=False)
    else:
        monkeypatch.setenv("FORCE_LOWERCASE_REPLIES", flag)
    action = await brain.brain_infer("hi")
    assert action.content == expected
