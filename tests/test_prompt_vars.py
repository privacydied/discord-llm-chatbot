"""Tests for request-time prompt placeholders [IV]."""

from datetime import UTC, datetime

from bot.config.prompts import render_prompt_vars

_NOW = datetime(2026, 10, 19, 1, 30, tzinfo=UTC)


def test_fills_date_and_time():
    text = "* it's {current_date}, {current_time_utc} utc right now."
    assert render_prompt_vars(text, _NOW) == "* it's monday 2026-10-19, 01:30 utc right now."


def test_leaves_other_braces_alone():
    text = "use {name} and {current_date}"
    assert render_prompt_vars(text, _NOW) == "use {name} and monday 2026-10-19"


def test_no_placeholder_is_identity():
    text = "plain prompt {x}"
    assert render_prompt_vars(text, _NOW) is text
