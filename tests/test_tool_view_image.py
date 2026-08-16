"""Tests for the view_image tool — re-reading an image posted earlier.
[CA][IV][REH][SFT].
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from bot.tools import ToolContext, execute_tool
from bot.tools.builtins import vision
from bot.tools.builtins.vision import MAX_IMAGE_LOOKBACK, view_image


class _Author:
    def __init__(self, name="alice"):
        self.display_name = name
        self.name = name


class _Msg:
    def __init__(self, content="", author="alice", minute=0, images=()):
        self.content = content
        self.author = _Author(author)
        self.created_at = datetime(2026, 8, 15, 12, minute, tzinfo=UTC)
        self._images = list(images)


class _Channel:
    def __init__(self, messages, raises=None):
        self._messages = messages
        self._raises = raises
        self.last_limit = None

    def history(self, limit=None, before=None):
        self.last_limit = limit
        raises = self._raises
        messages = self._messages[:limit]

        async def _gen():
            if raises:
                raise raises
            for msg in messages:
                yield msg

        return _gen()


class _Forbidden(Exception):
    pass


_Forbidden.__name__ = "Forbidden"


def _ref(url="https://cdn.discordapp.com/a/b/cat.png", filename="cat.png"):
    return SimpleNamespace(url=url, filename=filename, content_type="image/png", fallback_urls=[])


def _ctx(channel):
    msg = _Msg("current")
    msg.channel = channel
    return ToolContext(message=msg, bot=None, config={})


@pytest.fixture(autouse=True)
def _stub_dependencies(monkeypatch):
    """Harvest images off our fakes, allow all URLs, and stub the VL call."""
    monkeypatch.setattr(vision, "_image_refs", lambda msg: list(getattr(msg, "_images", []) or []))

    async def _describe(url, question, cfg):
        return f"a description of {url} answering '{question}'"

    monkeypatch.setattr(vision, "_describe", _describe)


def _channel_with_image_at(position, depth=10):
    """Channel where only the message `position` posts ago carries an image."""
    messages = []
    for i in range(depth):
        images = [_ref()] if (i + 1) == position else []
        messages.append(_Msg(f"msg {i + 1}", author=f"user{i + 1}", minute=i, images=images))
    return _Channel(messages)


# --------------------------------------------------------------------------
# The goldfish case
# --------------------------------------------------------------------------


async def test_finds_most_recent_image_without_being_told_where():
    """'that image' with no position must locate the latest one."""
    result = await view_image(_ctx(_channel_with_image_at(3)), {})
    assert result.ok
    assert "cat.png" in result.content
    assert "3 posts ago" in result.content


async def test_uses_explicit_position_when_given():
    result = await view_image(_ctx(_channel_with_image_at(5)), {"posts_ago": 5})
    assert result.ok
    assert "5 posts ago" in result.content


async def test_question_is_passed_to_vision():
    result = await view_image(_ctx(_channel_with_image_at(1)), {"question": "what breed is the cat?"})
    assert result.ok
    assert "what breed is the cat?" in result.content


async def test_default_question_used_when_omitted():
    result = await view_image(_ctx(_channel_with_image_at(1)), {})
    assert result.ok
    assert "Describe this image" in result.content


async def test_provenance_included():
    channel = _Channel([_Msg("look", author="bob", minute=4, images=[_ref()])])
    result = await view_image(_ctx(channel), {})
    assert "bob" in result.content
    assert "2026-08-15" in result.content


# --------------------------------------------------------------------------
# Search behaviour
# --------------------------------------------------------------------------


async def test_skips_messages_without_images():
    result = await view_image(_ctx(_channel_with_image_at(7)), {})
    assert result.ok
    assert "7 posts ago" in result.content


async def test_bounded_lookback_when_no_position_given():
    channel = _channel_with_image_at(999, depth=5)  # no image anywhere
    await view_image(_ctx(channel), {})
    assert channel.last_limit == MAX_IMAGE_LOOKBACK


async def test_only_fetches_as_deep_as_requested():
    channel = _channel_with_image_at(4)
    await view_image(_ctx(channel), {"posts_ago": 4})
    assert channel.last_limit == 4


async def test_no_image_anywhere_is_reported():
    channel = _Channel([_Msg(f"m{i}", minute=i) for i in range(6)])
    result = await view_image(_ctx(channel), {})
    assert not result.ok
    assert "no image found" in (result.error or "")


async def test_named_message_has_no_image():
    channel = _channel_with_image_at(2)
    result = await view_image(_ctx(channel), {"posts_ago": 5})
    assert not result.ok
    assert "no image" in (result.error or "").lower()


async def test_history_too_short():
    channel = _Channel([_Msg("only one", images=[_ref()])])
    result = await view_image(_ctx(channel), {"posts_ago": 9})
    assert not result.ok
    assert "does not go back" in (result.error or "")


# --------------------------------------------------------------------------
# Validation and failure modes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -3, MAX_IMAGE_LOOKBACK + 1])
async def test_rejects_out_of_range_position(bad):
    result = await view_image(_ctx(_channel_with_image_at(1)), {"posts_ago": bad})
    assert not result.ok


async def test_rejects_non_integer_position():
    result = await view_image(_ctx(_channel_with_image_at(1)), {"posts_ago": "banana"})
    assert not result.ok
    assert "integer" in (result.error or "")


async def test_accepts_numeric_string_position():
    result = await view_image(_ctx(_channel_with_image_at(2)), {"posts_ago": "2"})
    assert result.ok


async def test_missing_permission_reported():
    channel = _Channel([], raises=_Forbidden("nope"))
    result = await view_image(_ctx(channel), {})
    assert not result.ok
    assert "permission" in (result.error or "").lower()


async def test_history_failure_contained():
    channel = _Channel([], raises=RuntimeError("gateway"))
    result = await view_image(_ctx(channel), {})
    assert not result.ok


async def test_no_channel():
    result = await view_image(ToolContext(), {})
    assert not result.ok


async def test_vision_failure_is_reported_not_raised(monkeypatch):
    async def _fail(url, question, cfg):
        return None

    monkeypatch.setattr(vision, "_describe", _fail)
    result = await view_image(_ctx(_channel_with_image_at(1)), {})
    assert not result.ok
    assert "could not read that image" in (result.error or "")


async def test_ref_without_url_is_rejected(monkeypatch):
    channel = _Channel([_Msg("x", images=[SimpleNamespace(url="", filename="broken.png")])])
    result = await view_image(_ctx(channel), {})
    assert not result.ok


async def test_long_description_truncated(monkeypatch):
    async def _huge(url, question, cfg):
        return "x" * 9000

    monkeypatch.setattr(vision, "_describe", _huge)
    result = await view_image(_ctx(_channel_with_image_at(1)), {})
    assert result.ok
    assert len(result.content) < 9000


async def test_description_is_wrapped_as_untrusted():
    """A VL description of a user-supplied image is untrusted text. [SFT]"""
    result = await view_image(_ctx(_channel_with_image_at(1)), {})
    assert result.ok
    assert "UNVERIFIED" in result.content.upper()


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


async def test_dispatch_via_registry():
    result = await execute_tool("view_image", {}, _ctx(_channel_with_image_at(1)))
    assert result.ok


def test_registered_and_allowlisted():
    from bot.tools.registry import ALLOWED_TOOL_NAMES, get_registry

    assert "view_image" in ALLOWED_TOOL_NAMES
    assert "view_image" in get_registry().names()


def test_has_a_longer_budget_than_the_default():
    """Vision inference cannot finish inside the 10s default. [PA]"""
    from bot.tools.registry import TOOL_TIMEOUT_S, get_registry

    spec = get_registry().get("view_image")
    assert spec.timeout_s > TOOL_TIMEOUT_S


def test_loop_budget_accommodates_the_vision_tool():
    """A tool slower than the whole loop could never complete."""
    from bot.config._base import _build_config
    from bot.tools.builtins.vision import VIEW_IMAGE_TIMEOUT_S

    cfg = _build_config(lambda key, default=None: default)
    assert cfg["TOOLS_TIMEOUT_S"] > VIEW_IMAGE_TIMEOUT_S


def test_position_is_optional_in_schema():
    from bot.tools.builtins.vision import PARAMETERS

    assert PARAMETERS["required"] == []
