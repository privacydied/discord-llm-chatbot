"""Load the whisper model while the media is still downloading. [PA][REH]

The model choice depends on the audio duration (long audio downgrades
base -> tiny), and that used to be known only after ffprobe, so the cold load
(8.6 s for tiny on the NAS) ran strictly after the download. The ingest layer
now reports the duration as soon as it knows it (yt-dlp metadata, or the
ffmpeg demux header) and this prefetcher starts ``ensure_model`` for the
predicted spec in the background. The later ``ensure_model`` call in the
whisper stage then waits on the same per-spec lock and gets a cache hit.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .spec_select import predict_model_spec

# Strong references: the event loop only keeps weak ones to running tasks.
_PENDING: set[asyncio.Task] = set()


class ModelPrefetcher:
    """Starts at most one background model load per STT job."""

    def __init__(self, *, manager: Any, downgrade_threshold_s: float, logger: Any) -> None:
        self._manager = manager
        self._threshold = downgrade_threshold_s
        self._logger = logger
        self.spec: Any = None

    def on_duration(self, duration_s: float) -> None:
        """Duration hint from the ingest layer. Safe to call more than once."""
        if self.spec is not None or not duration_s or duration_s <= 0:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self.spec = predict_model_spec(manager=self._manager, duration_in_s=duration_s, downgrade_threshold_s=self._threshold)
        task = loop.create_task(self._load(self.spec, duration_s))
        _PENDING.add(task)
        task.add_done_callback(_PENDING.discard)

    async def _load(self, spec: Any, duration_s: float) -> None:
        started = time.monotonic()
        try:
            await self._manager.ensure_model(spec)
        except Exception as exc:  # noqa: BLE001 - best effort; the whisper stage loads (and reports) again
            self._logger.warning("stt.model_prefetch.fail size=%s error=%s", getattr(spec, "size", spec), str(exc)[:120])
            return
        self._logger.info(
            "stt.model_prefetch.done size=%s dur_hint=%.1f s=%.1f",
            getattr(spec, "size", spec),
            duration_s,
            time.monotonic() - started,
        )
