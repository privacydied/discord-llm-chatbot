"""Tests for still-frame extraction + VL description (bot/video_frame.py). [PA][REH]"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from bot.video_frame import STILL_VL_PROMPT, describe_video_still, extract_video_still

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg binary required")


@pytest.fixture(scope="module")
def video_5s(tmp_path_factory) -> Path:
    """A 5s silent test video (matching the silent-Twitter-clip scenario)."""
    path = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=5", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
        timeout=60,
    )
    return path


class TestExtractVideoStill:
    async def test_extracts_jpeg_at_midpointish(self, video_5s: Path) -> None:
        frame = await extract_video_still(video_5s, duration_s=5.0)
        assert frame is not None
        try:
            assert frame.suffix == ".jpg"
            assert frame.stat().st_size > 500  # a real image, not an empty stub
        finally:
            frame.unlink(missing_ok=True)

    async def test_short_clip_falls_back_to_start(self, video_5s: Path) -> None:
        # Seek far past the end -> first attempt fails -> retry at t=0 succeeds
        frame = await extract_video_still(video_5s, duration_s=3000.0)
        assert frame is not None
        frame.unlink(missing_ok=True)

    async def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert await extract_video_still(tmp_path / "nope.mp4") is None


class TestDescribeVideoStill:
    async def test_returns_vl_text_and_cleans_frame(self, video_5s: Path, monkeypatch) -> None:
        captured = {}

        async def fake_see_infer(image_path, prompt=None, model_override=None):
            captured["path"] = Path(image_path)
            captured["exists_during_call"] = Path(image_path).exists()
            captured["prompt"] = prompt

            class Action:
                content = "A colorful SMPTE-style test pattern."
                error = False

            return Action()

        monkeypatch.setattr("bot.see.see_infer", fake_see_infer)

        text = await describe_video_still(video_5s, duration_s=5.0)

        assert text == "A colorful SMPTE-style test pattern."
        assert captured["exists_during_call"] is True
        assert captured["prompt"] == STILL_VL_PROMPT
        assert not captured["path"].exists()  # temp frame cleaned up [RM]

    async def test_vl_error_action_returns_none(self, video_5s: Path, monkeypatch) -> None:
        async def fake_see_infer(image_path, prompt=None, model_override=None):
            class Action:
                content = "provider exploded"
                error = True

            return Action()

        monkeypatch.setattr("bot.see.see_infer", fake_see_infer)
        assert await describe_video_still(video_5s) is None

    async def test_vl_exception_returns_none(self, video_5s: Path, monkeypatch) -> None:
        async def fake_see_infer(image_path, prompt=None, model_override=None):
            raise RuntimeError("VL blew up")

        monkeypatch.setattr("bot.see.see_infer", fake_see_infer)
        assert await describe_video_still(video_5s) is None  # never raises [REH]
