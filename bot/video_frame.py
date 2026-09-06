"""Extract a representative still frame from a video and describe it via VL. [PA][REH]

Stitched-context support for the video→STT flow: alongside the audio
transcript, a single frame is pulled with ffmpeg and run through the vision
flow (see_infer), giving the text LLM both what was said AND what is visible.
For silent videos (NoAudioStreamError) the still is the only context available.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from pathlib import Path

from bot.utils.logging import get_logger

logger = get_logger(__name__)

# Timeouts + geometry [CMV][REH]
STILL_EXTRACT_TIMEOUT_S = 15.0
STILL_VL_TIMEOUT_S = 60.0
STILL_MAX_WIDTH = 1280
STILL_SEEK_FRACTION = 0.33  # one-third in: usually past intros/title cards
STILL_FALLBACK_SEEK_S = 1.0
STILL_JPEG_QUALITY = "3"  # ffmpeg -q:v (2-5 = high quality)

STILL_VL_PROMPT = (
    "This is a single still frame captured from a video. Describe what is visibly happening: people, objects, text on screen, setting, and any notable action. Be concise and factual."
)


def _ffmpeg_bin() -> str:
    """Reuse the STT pipeline's probed ffmpeg binary (AAC-capable). [REH]"""
    from bot.hear import _resolve_ffmpeg_bin

    return _resolve_ffmpeg_bin()


async def _run_extract(video_path: Path, seek_s: float, out_path: Path) -> bool:
    cmd = [
        _ffmpeg_bin(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, seek_s):.2f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        STILL_JPEG_QUALITY,
        "-vf",
        f"scale='min({STILL_MAX_WIDTH},iw)':-2",
        "-y",
        str(out_path),
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=STILL_EXTRACT_TIMEOUT_S)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()
        logger.warning("video.still.extract_timeout | path=%s", video_path.name)
        return False
    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        logger.debug("video.still.extract_fail rc=%s err=%s", proc.returncode, (stderr or b"").decode(errors="ignore")[:200])
        return False
    return True


async def extract_video_still(video_path: Path, duration_s: float | None = None) -> Path | None:
    """Grab one frame as a JPEG; returns the temp file path or None. [REH]

    Seeks a third of the way in when duration is known; falls back to t=1.0s,
    then t=0 (for clips shorter than a second). Caller owns cleanup.
    """
    if not video_path.exists():
        return None
    fd, tmp_name = tempfile.mkstemp(prefix="vstill_", suffix=".jpg")
    os.close(fd)
    out_path = Path(tmp_name)
    seek = duration_s * STILL_SEEK_FRACTION if duration_s and duration_s > 0 else STILL_FALLBACK_SEEK_S
    for attempt_seek in (seek, 0.0):
        if await _run_extract(video_path, attempt_seek, out_path):
            logger.info("video.still.extract_ok | seek=%.2fs size=%d", attempt_seek, out_path.stat().st_size)
            return out_path
        if attempt_seek == 0.0:
            break
    with contextlib.suppress(OSError):
        out_path.unlink()
    return None


async def describe_video_still(video_path: Path, duration_s: float | None = None, prompt: str | None = None) -> str | None:
    """Extract a still and describe it through the VL flow. Never raises. [REH]

    Returns the visual description text, or None on any failure (no frame,
    VL error/timeout, empty response) — callers treat it as best-effort
    context, not a hard dependency.
    """
    frame: Path | None = None
    try:
        frame = await extract_video_still(video_path, duration_s=duration_s)
        if frame is None:
            return None
        from bot.see import see_infer

        action = await asyncio.wait_for(
            see_infer(str(frame), prompt=prompt or STILL_VL_PROMPT),
            timeout=STILL_VL_TIMEOUT_S,
        )
        if action is None or getattr(action, "error", False):
            return None
        text = (getattr(action, "content", "") or "").strip()
        return text or None
    except TimeoutError:
        logger.warning("video.still.vl_timeout | path=%s", video_path.name)
        return None
    except Exception as exc:  # noqa: BLE001 - VL backend unbounded; TimeoutError handled above, logged, None fallback
        logger.debug(f"video.still.vl_fail: {exc}")
        return None
    finally:
        if frame is not None:
            with contextlib.suppress(OSError):
                frame.unlink()
