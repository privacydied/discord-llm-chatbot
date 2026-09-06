"""Tests for PCM retry-cache preservation across consumer aborts (bot.hear).

Production fault chain: memory-guard abort -> finalize() deleted the temp PCM
unconditionally -> slow-decode downgrade (base->tiny) found "PCM cache
unavailable for STT retry" -> kept a partial transcript, and any outer retry
paid a full re-download + re-preprocess (up to the 45s pre budget).

Fixed lifecycle: the temp file is promoted to the retry cache whenever the
captured audio itself is intact (ffmpeg reached EOF, no preprocessing error),
even when the *consumer* aborted. A truncated capture (producer killed
mid-stream) is still deleted so it can never masquerade as full audio.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import bot.hear as hear_mod
from bot.hear import FFMpegPCMStream, _load_pcm_cache


def _make_stream(tmp_path: Path, *, producer_ok: bool, aborted: bool, samples: int = 16000) -> FFMpegPCMStream:
    stream = FFMpegPCMStream.__new__(FFMpegPCMStream)
    hear_mod.BasePCMStream.__init__(stream, sample_rate=16000, frame_samples=4000)
    temp_path = tmp_path / "ffmpeg_out.pcm"
    temp_path.write_bytes(b"\x00\x01" * samples)
    stream._proc = MagicMock()
    stream._proc.returncode = 0
    stream._temp_path = temp_path
    stream._cache_key = "unittest-pcm-key"
    stream._spans = MagicMock()
    stream._pre_timeout = 5.0
    stream._stderr_task = None
    stream._producer_ok = producer_ok
    stream._aborted = aborted
    stream._total_samples = samples
    stream._error = None
    return stream


@pytest.mark.asyncio
async def test_abort_after_complete_audio_preserves_pcm(tmp_path, monkeypatch) -> None:
    """Consumer abort (memory guard) with ffmpeg at EOF keeps the retry cache,
    so the model-downgrade path can re-run from disk instead of failing."""
    monkeypatch.setattr(hear_mod, "PCM_CACHE_DIR", tmp_path / "pcm")
    stream = _make_stream(tmp_path, producer_ok=True, aborted=True)
    await stream.finalize(False)
    assert not stream._temp_path.exists()
    cached = _load_pcm_cache("unittest-pcm-key")
    assert cached is not None
    pcm_path, meta = cached
    assert pcm_path.exists()
    assert int(meta.get("total_samples")) == 16000


@pytest.mark.asyncio
async def test_truncated_capture_still_deleted(tmp_path, monkeypatch) -> None:
    """Producer killed mid-stream: partial PCM must never enter the cache."""
    monkeypatch.setattr(hear_mod, "PCM_CACHE_DIR", tmp_path / "pcm")
    stream = _make_stream(tmp_path, producer_ok=False, aborted=True)
    await stream.finalize(False)
    assert not stream._temp_path.exists()
    assert _load_pcm_cache("unittest-pcm-key") is None


@pytest.mark.asyncio
async def test_preprocess_error_still_deleted(tmp_path, monkeypatch) -> None:
    """ffmpeg failure: corrupt/errored audio must never enter the cache.

    (Base finalize re-raises the preprocessing error -- pre-existing
    contract relied on by callers -- so nothing is stored.)"""
    monkeypatch.setattr(hear_mod, "PCM_CACHE_DIR", tmp_path / "pcm")
    stream = _make_stream(tmp_path, producer_ok=False, aborted=False)
    stream._error = RuntimeError("ffmpeg boom")
    with pytest.raises(RuntimeError, match="ffmpeg boom"):
        await stream.finalize(False)
    assert _load_pcm_cache("unittest-pcm-key") is None
