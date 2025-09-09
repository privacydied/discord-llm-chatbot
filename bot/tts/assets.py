"""
Async asset management for Kokoro-ONNX models.

- Removes legacy paths under tts/onnx and tts/voices
- Downloads model and voices concurrently to tts/
- Robust fallback: aiohttp first, requests fallback
- Cached: skips existing files unless force=True

Run at bot startup via TTSManager to prepare assets asynchronously.

[RAT][IV][RM][PA][REH][CMV]
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Tuple

from bot.util.logging import get_logger

logger = get_logger(__name__)

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

MODEL_NAME = "kokoro-v1.0.onnx"
VOICES_NAME = "voices-v1.0.bin"

LEGACY_MODEL = Path("tts/onnx/kokoro-v1.0.onnx")
LEGACY_VOICES = Path("tts/voices/voices-v1.0.bin")


async def _download_aiohttp(url: str, dest: Path, timeout: int = 120) -> None:
    import aiohttp  # type: ignore

    dest.parent.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.get(url, timeout=timeout) as resp:
            with open(dest.with_suffix(dest.suffix + ".tmp"), "wb") as f:
                async for chunk in resp.content.iter_chunked(1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.replace(dest)


def _download_requests(url: str, dest: Path, timeout: int = 120) -> None:
    import requests  # type: ignore

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest.with_suffix(dest.suffix + ".tmp"), "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.replace(dest)


async def _ensure_file(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        logger.info(
            "ℹ Asset present; skip download",
            extra={"subsys": "tts.assets", "event": "skip", "detail": str(dest)},
        )
        return

    # Try aiohttp first
    try:
        await _download_aiohttp(url, dest)
        logger.info(
            "✔ Downloaded asset (aiohttp)",
            extra={"subsys": "tts.assets", "event": "download", "detail": str(dest)},
        )
        return
    except Exception:
        logger.warning(
            "⚠ aiohttp failed; falling back to requests",
            extra={"subsys": "tts.assets"},
            exc_info=True,
        )

    # Fallback to requests in a thread
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _download_requests, url, dest, 120)
    logger.info(
        "✔ Downloaded asset (requests)",
        extra={"subsys": "tts.assets", "event": "download", "detail": str(dest)},
    )


async def ensure_kokoro_assets(
    out_dir: Path = Path("tts"), force: bool = False
) -> Tuple[Path, Path]:
    """Ensure Kokoro-ONNX model and voices exist under out_dir.

    - Deletes legacy paths under tts/onnx and tts/voices.
    - Downloads concurrently if missing.
    - Returns (model_path, voices_path).
    """
    # Remove legacy files if present
    for legacy in (LEGACY_MODEL, LEGACY_VOICES):
        try:
            if legacy.exists():
                legacy.unlink()
                logger.info(
                    "ℹ Removed legacy asset",
                    extra={
                        "subsys": "tts.assets",
                        "event": "removed_legacy",
                        "detail": str(legacy),
                    },
                )
        except Exception:
            logger.warning(
                "⚠ Failed to remove legacy asset",
                extra={"subsys": "tts.assets", "detail": str(legacy)},
                exc_info=True,
            )

    model_path = out_dir / MODEL_NAME
    voices_path = out_dir / VOICES_NAME

    # If both exist and not force, done
    if model_path.exists() and voices_path.exists() and not force:
        return model_path, voices_path

    # Concurrent downloads for missing ones
    tasks = []
    if force or not model_path.exists():
        tasks.append(_ensure_file(MODEL_URL, model_path, force))
    if force or not voices_path.exists():
        tasks.append(_ensure_file(VOICES_URL, voices_path, force))

    if tasks:
        await asyncio.gather(*tasks)

    return model_path, voices_path
