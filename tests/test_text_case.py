"""Tests for persona lowercase enforcement [IV][REH]."""

import pytest

from bot import brain
from bot.core.text_case import collapse_line_breaks_outside_code, lowercase_outside_protected


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


# ── line-break collapsing ────────────────────────────────────────────────────


def test_collapses_blank_line_after_emoji():
    text = "nigga really out here running a whale operation 💀\n\nthat's a wild tangent"
    assert collapse_line_breaks_outside_code(text) == "nigga really out here running a whale operation 💀 that's a wild tangent"


def test_collapses_single_newlines_and_trailing_spaces():
    assert collapse_line_breaks_outside_code("one  \ntwo\n\n\n three") == "one two three"


def test_keeps_list_items_on_their_own_lines():
    text = "do this:\n\n1. install it\n2. run it\n- or skip\n\nthen done"
    assert collapse_line_breaks_outside_code(text) == "do this:\n1. install it\n2. run it\n- or skip then done"


def test_leaves_code_blocks_untouched():
    text = "run this:\n\n```py\nx = 1\n\ny = 2\n```\n\nthat's it"
    assert collapse_line_breaks_outside_code(text) == "run this:\n```py\nx = 1\n\ny = 2\n```\nthat's it"


def test_no_newline_is_identity():
    text = "just one line"
    assert collapse_line_breaks_outside_code(text) is text


async def test_brain_infer_collapses_by_default(monkeypatch):
    async def fake_generate_response(**_kwargs):
        return {"text": "yo 💀\n\nthat's crazy"}

    monkeypatch.setattr(brain, "generate_response", fake_generate_response)
    monkeypatch.delenv("COLLAPSE_REPLY_LINE_BREAKS", raising=False)
    assert (await brain.brain_infer("hi")).content == "yo 💀 that's crazy"
    monkeypatch.setenv("COLLAPSE_REPLY_LINE_BREAKS", "false")
    assert (await brain.brain_infer("hi")).content == "yo 💀\n\nthat's crazy"
