"""Silent media must raise NoAudioStreamError, not fall back to fake defaults. [IV][REH]

Regression test for: silent Twitter video → _ffprobe returned default
sr=16000/ch=1 → ffmpeg preprocessing died with the cryptic "Output file does
not contain any stream" → user was told to "try again later" for a video that
can never be transcribed.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from bot.exceptions import InferenceError, NoAudioStreamError
from bot.hear import _ffprobe

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg binary required")


@pytest.fixture(scope="module")
def silent_video(tmp_path_factory) -> str:
    """A 1s h264 video with NO audio track (like a muted Twitter clip)."""
    path = tmp_path_factory.mktemp("media") / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
        timeout=60,
    )
    return str(path)


@pytest.fixture(scope="module")
def audio_file(tmp_path_factory) -> str:
    """A 1s mono WAV with a real audio stream."""
    path = tmp_path_factory.mktemp("media") / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-ar", "16000", "-ac", "1", str(path)],
        check=True,
        timeout=60,
    )
    return str(path)


class TestNoAudioStreamDetection:
    async def test_silent_video_raises_no_audio_error(self, silent_video) -> None:
        from pathlib import Path

        with pytest.raises(NoAudioStreamError) as exc_info:
            await _ffprobe(Path(silent_video))
        # media_path lets the router extract a still frame for VL context [PA]
        assert exc_info.value.media_path == silent_video

    async def test_no_audio_error_is_an_inference_error(self) -> None:
        # Callers catching InferenceError keep working [REH]
        assert issubclass(NoAudioStreamError, InferenceError)

    async def test_real_audio_still_probes_normally(self, audio_file) -> None:
        from pathlib import Path

        duration, sample_rate, channels = await _ffprobe(Path(audio_file))
        assert 0.5 < duration < 2.0
        assert sample_rate == 16000
        assert channels == 1
