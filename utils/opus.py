from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional


async def transcode_to_ogg_opus(
    wav_path: str | Path,
    out_path: Optional[str | Path] = None,
    *,
    bitrate: str = "32k",
    vbr: str = "on",
    compression_level: int = 10,
) -> Path:
    """
    Transcode a WAV file to Ogg Opus (48 kHz mono) via ffmpeg. [PA][IV][REH]

    Returns the output Path. Raises RuntimeError if ffmpeg fails.
    """
    src = Path(wav_path)
    if not src.exists():
        raise FileNotFoundError(f"Input WAV not found: {src}")

    dst = Path(out_path) if out_path else src.with_suffix(".ogg")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",  # mono
        "-ar",
        "48000",  # 48kHz
        "-c:a",
        "libopus",
        "-b:a",
        bitrate,
        "-vbr",
        vbr,
        "-compression_level",
        str(compression_level),
        str(dst),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg opus transcode failed (code={proc.returncode}). stderr={stderr.decode(errors='ignore')[:4000]}"
        )

    return dst
