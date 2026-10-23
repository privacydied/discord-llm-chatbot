"""Image generation success posts the image only -- no follow-up status line [CA]."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from bot.router import Router


async def test_vision_success_posts_image_and_returns_no_payload(tmp_path, monkeypatch):
    png = tmp_path / "out.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    router = Router.__new__(Router)
    router.logger = MagicMock()
    router._build_vision_status_embed = MagicMock(return_value=MagicMock())
    edited = AsyncMock()
    monkeypatch.setattr("bot.router.safe_edit", edited)
    channel = SimpleNamespace(send=AsyncMock(), permissions_for=lambda me: SimpleNamespace(attach_files=True, send_messages=True))
    original = SimpleNamespace(channel=channel, guild=SimpleNamespace(me=object()), author=SimpleNamespace(id=1))
    job = SimpleNamespace(job_id="abcdef123456", response=SimpleNamespace(artifacts=[png], actual_cost=None), request=SimpleNamespace(prompt="a cat"))

    action = await Router._handle_vision_success(router, job, MagicMock(), original)

    channel.send.assert_awaited_once()
    assert len(channel.send.await_args.kwargs["files"]) == 1
    assert not action.has_payload  # nothing else posted after the image
