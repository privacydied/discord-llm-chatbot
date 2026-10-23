"""Kyutai Pocket TTS engine. [CA][REH][PA]

Pocket TTS (MIT, ~100M params, CPU real-time) runs as its own HTTP server
(``python -m pocket_tts serve``) in a separate venv, so torch never loads into
the bot process and the bot's venv is untouched.

- ``POCKET_TTS_URL`` points at the server (default http://127.0.0.1:8765).
- With ``POCKET_TTS_AUTOSTART`` (default on) the engine launches the server
  from ``POCKET_TTS_PYTHON`` on first use, waits for ``/health``, and stops it
  again after ``MODEL_IDLE_UNLOAD_SECONDS`` without use (frees ~1.2 GB).
- A server already listening on the URL (e.g. run by hand) is used as-is and
  never stopped by the bot.

Measured on the NAS (Ryzen R1600): RTF ~0.9 for ``english`` (100M), ~0.4 for
``english_18m_2026-10``. Request timeouts scale with text length.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import struct
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import aiohttp

from ...utils.logging import get_logger
from ..errors import SynthesisError
from .base import BaseEngine

logger = get_logger(__name__)

# 8765 is taken on the NAS (fmpresence-art), whose /health also answers 200.
DEFAULT_URL = "http://127.0.0.1:8767"
DEFAULT_LANGUAGE = "english"
DEFAULT_VOICE = "alba"
HEALTH_TIMEOUT_S = 2.0
START_POLL_S = 1.0
STOP_GRACE_S = 5.0
LOG_TAIL_BYTES = 1500
# Synthesis budget: fixed overhead plus a per-character allowance. Speech runs
# ~15 chars/s; at RTF ~0.9 that is ~0.06 s/char, so 0.12 leaves 2x headroom.
BUDGET_BASE_S = 10.0
BUDGET_PER_CHAR_S = 0.12


def fix_wav_header(data: bytes) -> bytes:
    """Rewrite RIFF/data sizes of a streamed WAV (written before length was known)."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise SynthesisError("Pocket TTS returned non-WAV audio")
    out = bytearray(data)
    pos = 12
    while pos + 8 <= len(out):
        chunk_id = bytes(out[pos : pos + 4])
        size = struct.unpack_from("<I", out, pos + 4)[0]
        if chunk_id == b"data":
            struct.pack_into("<I", out, pos + 4, len(out) - pos - 8)
            struct.pack_into("<I", out, 4, len(out) - 8)
            return bytes(out)
        pos += 8 + size + (size & 1)
    raise SynthesisError("Pocket TTS WAV has no data chunk")


def _cfg(config: Mapping[str, Any], key: str, default: str) -> str:
    value = config.get(key)
    if value in (None, ""):
        value = os.getenv(key, default)
    return str(value).strip().strip('"')


def _cfg_bool(config: Mapping[str, Any], key: str, default: bool) -> bool:
    return _cfg(config, key, "true" if default else "false").lower() in {"1", "true", "yes", "on"}


class PocketTTSEngine(BaseEngine):
    """HTTP client for a (optionally bot-managed) Pocket TTS server."""

    def __init__(self, config: Mapping[str, Any] | None = None, voice: str | None = None) -> None:
        super().__init__()
        cfg = config or {}
        self.url = _cfg(cfg, "POCKET_TTS_URL", DEFAULT_URL).rstrip("/")
        self.language = _cfg(cfg, "POCKET_TTS_LANGUAGE", DEFAULT_LANGUAGE)
        self.voice = voice or _cfg(cfg, "TTS_VOICE", DEFAULT_VOICE)
        self.autostart = _cfg_bool(cfg, "POCKET_TTS_AUTOSTART", True)
        self.python = _cfg(cfg, "POCKET_TTS_PYTHON", "")
        self.hf_home = _cfg(cfg, "POCKET_TTS_HF_HOME", "")
        self.threads = int(_cfg(cfg, "POCKET_TTS_THREADS", "2"))
        self.start_timeout_s = float(_cfg(cfg, "POCKET_TTS_START_TIMEOUT_S", "180"))
        self.log_path = Path(_cfg(cfg, "POCKET_TTS_LOG", "logs/pocket_tts.log"))
        self._proc: asyncio.subprocess.Process | None = None
        self._start_lock = asyncio.Lock()
        self._last_used = time.monotonic()
        self._busy = 0

    # -- budgets --------------------------------------------------------------------
    def timeout_budget(self, text: str) -> float:
        """Seconds a request for ``text`` may take, including a cold server start."""
        budget = BUDGET_BASE_S + BUDGET_PER_CHAR_S * len(text)
        if self._proc is None or self._proc.returncode is not None:
            budget += self.start_timeout_s
        return budget

    # -- server lifecycle -----------------------------------------------------------
    async def _healthy(self, session: aiohttp.ClientSession) -> bool:
        try:
            async with session.get(f"{self.url}/health", timeout=aiohttp.ClientTimeout(total=HEALTH_TIMEOUT_S)) as resp:
                # Exact Pocket reply: another service on the port must not pass as Pocket.
                return resp.status == 200 and (await resp.json(content_type=None)).get("status") == "healthy"
        except (aiohttp.ClientError, TimeoutError, OSError, ValueError, AttributeError):
            return False

    def _serve_command(self) -> list[str]:
        host_port = self.url.split("://", 1)[-1]
        host, _, port = host_port.partition(":")
        return [self.python, "-m", "pocket_tts", "serve", "--host", host or "127.0.0.1", "--port", port or "8767", "--language", self.language, "--default-voice", self.voice]

    def _log_tail(self) -> str:
        with contextlib.suppress(OSError):
            data = self.log_path.read_bytes()[-LOG_TAIL_BYTES:]
            # Typer prints errors inside a box; keep the lines with real text.
            lines = [ln.strip(" │╭╮╰╯─\t") for ln in data.decode("utf-8", "replace").splitlines()]
            return " | ".join(ln for ln in lines if ln)[-300:]
        return ""

    async def _spawn(self) -> None:
        if not self.python or not Path(self.python).exists():
            msg = f"Pocket TTS server is down and POCKET_TTS_PYTHON is not a valid interpreter: {self.python!r}"
            raise SynthesisError(msg)
        env = dict(os.environ, OMP_NUM_THREADS=str(self.threads), MKL_NUM_THREADS=str(self.threads))
        if self.hf_home:
            env["HF_HOME"] = self.hf_home
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("ab") as log:
            # Fixed argv from operator config (POCKET_TTS_PYTHON), no shell.
            self._proc = await asyncio.create_subprocess_exec(*self._serve_command(), stdout=log, stderr=log, env=env)
        logger.info(f"tts.pocket.spawned pid={self._proc.pid} language={self.language} voice={self.voice}")

    async def _ensure_server(self, session: aiohttp.ClientSession) -> None:
        if await self._healthy(session):
            return
        async with self._start_lock:
            if await self._healthy(session):
                return
            if not self.autostart:
                raise SynthesisError(f"Pocket TTS server not reachable at {self.url}")
            if self._proc is None or self._proc.returncode is not None:
                await self._spawn()
            started = time.monotonic()
            deadline = started + self.start_timeout_s
            while time.monotonic() < deadline:
                if self._proc is not None and self._proc.returncode is not None:
                    raise SynthesisError(f"Pocket TTS server exited ({self._proc.returncode}): {self._log_tail()}")
                if await self._healthy(session):
                    logger.info(f"tts.pocket.ready after={time.monotonic() - started:.1f}s")
                    return
                await asyncio.sleep(START_POLL_S)
            await self._stop()
            raise SynthesisError(f"Pocket TTS server not healthy after {self.start_timeout_s:.0f}s")

    async def _stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=STOP_GRACE_S)
        except TimeoutError:
            proc.kill()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=STOP_GRACE_S)
        logger.info(f"tts.pocket.stopped pid={proc.pid}")

    def idle_stop_due(self, ttl_s: float) -> bool:
        """True when a bot-owned server has been idle for ``ttl_s`` (sync check)."""
        if self._proc is None or self._proc.returncode is not None or self._busy:
            return False
        return time.monotonic() - self._last_used >= ttl_s

    async def close(self) -> None:
        await self._stop()

    # -- synthesis ------------------------------------------------------------------
    async def synthesize(self, text: str) -> bytes:
        self._busy += 1
        try:
            budget = aiohttp.ClientTimeout(total=self.timeout_budget(text))
            async with aiohttp.ClientSession(timeout=budget) as session:
                await self._ensure_server(session)
                form = aiohttp.FormData({"text": text, "voice_url": self.voice})
                started = time.monotonic()
                async with session.post(f"{self.url}/tts", data=form) as resp:
                    if resp.status != 200:
                        detail = (await resp.text())[:200]
                        raise SynthesisError(f"Pocket TTS HTTP {resp.status}: {detail}")
                    audio = fix_wav_header(await resp.read())
            logger.info(f"tts.pocket.synth chars={len(text)} bytes={len(audio)} took={time.monotonic() - started:.2f}s voice={self.voice}")
            return audio
        except aiohttp.ClientError as exc:
            raise SynthesisError(f"Pocket TTS request failed: {exc}") from exc
        finally:
            self._busy -= 1
            self._last_used = time.monotonic()
