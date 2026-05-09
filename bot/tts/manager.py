from __future__ import annotations

import logging
import asyncio
from pathlib import Path
from typing import Optional

from bot.config import load_config
from bot.tts.engines.kokoro import KokoroTTS, KokoroConfig
from bot.tts.errors import SynthesisError
from bot.memory import get_profile

logger = logging.getLogger(__name__)

class TTSManager:
    def __init__(self) -> None:
        cfg = load_config()
        self.enabled = bool(cfg.get("TTS_ENABLE", False))
        self.engine: KokoroTTS | None = None
        self.max_concurrent_jobs = int(cfg.get("TTS_MAX_CONCURRENT", 2))
        self.max_text_length = int(cfg.get("TTS_MAX_TEXT_LENGTH", 500))
        self.synthesis_timeout = float(cfg.get("TTS_SYNTHESIS_TIMEOUT", 30.0))
        self._current_jobs: dict[str, asyncio.Task] = {}
        self._job_locks: dict[str, asyncio.Lock] = {}
        
        if self.enabled:
            try:
                # Initialize Kokoro TTS engine
                self.engine = KokoroTTS.from_config(cfg)
                logger.info("TTS engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize TTS engine: {e}", exc_info=True)
                self.enabled = False

    async def generate_tts(self, text: str, user_id: str, out_path: str) -> str:
        if not self.enabled or not self.engine:
            raise RuntimeError("TTS is not enabled")
        
        # Check max text length
        if len(text) > self.max_text_length:
            raise ValueError(f"Text length exceeds maximum ({self.max_text_length} characters)")
        
        # Per-user gating: check user's profile preferences
        profile = get_profile(user_id)
        user_pref = profile.get("preferences", {}).get("memory_enabled", True)
        if not user_pref:
            raise RuntimeError("TTS not available due to user memory preference")
        
        # Check concurrent jobs for this user (simple per-user gating)
        if user_id in self._current_jobs:
            raise RuntimeError("User already has a pending TTS job")
        
        # Create a lock for this user
        if user_id not in self._job_locks:
            self._job_locks[user_id] = asyncio.Lock()
        
        # Acquire the lock to prevent concurrent jobs for the same user
        async with self._job_locks[user_id]:
            # Create a task for this job
            task = asyncio.create_task(self._synthesize_with_timeout(text, out_path))
            self._current_jobs[user_id] = task
            
            try:
                result = await task
                return result
            except Exception as e:
                raise e
            finally:
                # Remove the job from current jobs
                if user_id in self._current_jobs and self._current_jobs[user_id] is task:
                    del self._current_jobs[user_id]

    async def _synthesize_with_timeout(self, text: str, out_path: str) -> str:
        try:
            # Synthesize with timeout
            await asyncio.wait_for(
                self.engine.synthesize(text, out_path),
                timeout=self.synthesis_timeout
            )
            return out_path
        except asyncio.TimeoutError:
            raise SynthesisError(f"TTS synthesis timed out after {self.synthesis_timeout} seconds")

    def get_status(self) -> dict:
        if not self.enabled or not self.engine:
            return {"enabled": False, "status": "disabled"}
        return {
            "enabled": True,
            "status": "ready",
            "engine": "kokoro",
            "max_chars": self.max_text_length,
            "sample_rate": self.engine.sample_rate if hasattr(self.engine, "sample_rate") else "unknown",
            "max_concurrent": self.max_concurrent_jobs,
        }

    async def shutdown(self) -> None:
        # Cancel any ongoing jobs
        for task in self._current_jobs.values():
            task.cancel()
        if self.engine:
            await self.engine.shutdown()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
