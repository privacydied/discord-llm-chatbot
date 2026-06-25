"""Background distiller that mines raw server archive history into curated memories."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from bot.config import load_config
from bot.server_archive.service import get_server_archive_service

from .curator import CuratedMemoryCurator, MemoryCandidate
from .service import get_memory_service

logger = logging.getLogger(__name__)

_DURABLE_HINTS = (
    "i prefer",
    "my preference is",
    "i'd prefer",
    "i would prefer",
    "call me",
    "i like replies",
    "i like reply",
    "always",
    "never",
    "from now on",
    "going forward",
    "for discord-bot",
    "for the discord-bot",
    "in this project",
    "the bot should",
    "the rule is",
    "we decided",
    "final decision",
    "the plan is",
    "you were wrong about",
    "actually",
    "don't do that again",
)

_BLOCKED_HINTS = (
    "password",
    "passphrase",
    "api key",
    "secret",
    "token",
    "authorization",
    "bearer",
    "private key",
    "chain of thought",
    "think step by step",
    "internal prompt",
    "tool trace",
    "function call",
    "hidden reasoning",
    "system prompt",
)


@dataclass(slots=True)
class DistillerWindow:
    messages: list[dict[str, Any]]


class MemoryArchiveDistiller:
    def __init__(self, bot: Any | None = None) -> None:
        self.bot = bot
        self._start_lock = asyncio.Lock()
        self._started = False
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._dry_run_override: bool | None = None
        self.archive_service = None
        self.memory_service = None
        self.last_run: dict[str, Any] | None = None
        self.refresh_config()
        self.curator = CuratedMemoryCurator(
            default_ttl_days=int(self.config.get("PERSISTENT_MEMORY_DEFAULT_TTL_DAYS", 180)),
            temp_ttl_days=int(self.config.get("PERSISTENT_MEMORY_TEMP_TTL_DAYS", 14)),
            min_importance=float(self.config.get("PERSISTENT_MEMORY_MIN_IMPORTANCE", 0.55)),
        )

    def refresh_config(self) -> None:
        cfg = load_config()
        self.config = cfg
        self.enabled = bool(cfg.get("MEMORY_DISTILLER_ENABLED", False))
        if self._dry_run_override is None:
            self.dry_run = bool(cfg.get("MEMORY_DISTILLER_DRY_RUN", True))
        else:
            self.dry_run = self._dry_run_override
        self.batch_size = max(1, int(cfg.get("MEMORY_DISTILLER_BATCH_SIZE", 200)))
        self.interval_seconds = max(1, int(cfg.get("MEMORY_DISTILLER_INTERVAL_SECONDS", 900)))
        self.window_messages = max(1, int(cfg.get("MEMORY_DISTILLER_WINDOW_MESSAGES", 25)))
        self.min_confidence = float(cfg.get("MEMORY_DISTILLER_MIN_CONFIDENCE", 0.85))
        self.max_memories_per_window = max(1, int(cfg.get("MEMORY_DISTILLER_MAX_MEMORIES_PER_WINDOW", 3)))
        self.exclude_bot_messages = bool(cfg.get("MEMORY_DISTILLER_EXCLUDE_BOT_MESSAGES", True))

    def set_dry_run(self, enabled: bool) -> None:
        self._dry_run_override = bool(enabled)
        self.dry_run = self._dry_run_override

    async def start(self, bot: Any | None = None) -> MemoryArchiveDistiller:
        if bot is not None:
            self.bot = bot
        self.refresh_config()
        if not self.enabled:
            logger.info(
                "Memory distiller disabled",
                extra={"subsys": "memory", "event": "memory_distiller_disabled"},
            )
            return self

        async with self._start_lock:
            if self._started:
                return self

            self.archive_service = await get_server_archive_service(self.bot)
            self.memory_service = await get_memory_service(self.bot)
            if not getattr(self.archive_service, "enabled", False):
                logger.info(
                    "Memory distiller skipped because archive is disabled",
                    extra={
                        "subsys": "memory",
                        "event": "memory_distiller_archive_disabled",
                    },
                )
                return self
            if not getattr(self.memory_service, "enabled", False):
                logger.info(
                    "Memory distiller skipped because curated memory is disabled",
                    extra={
                        "subsys": "memory",
                        "event": "memory_distiller_memory_disabled",
                    },
                )
                return self

            await self.archive_service.store.initialize()
            await self.memory_service.store.initialize()
            await self.memory_service.semantic_store.initialize()
            self._stop_event.clear()
            self._task = asyncio.create_task(self._loop(), name="memory-distiller-loop")
            self._started = True
            logger.info(
                "Memory distiller started",
                extra={
                    "subsys": "memory",
                    "event": "memory_distiller_started",
                    "detail": {
                        "dry_run": self.dry_run,
                        "batch_size": self.batch_size,
                        "interval_seconds": self.interval_seconds,
                    },
                },
            )
            return self

    async def stop(self) -> None:
        self._stop_event.set()
        task = self._task
        self._task = None
        self._started = False
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.gather(task, return_exceptions=True)
            except (asyncio.CancelledError, RuntimeError):
                logger.debug("Memory distiller task cancellation raised", exc_info=True)

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except (AttributeError, TypeError, ValueError, RuntimeError, OSError):
                logger.exception(
                    "Memory distiller loop failed",
                    extra={"subsys": "memory", "event": "memory_distiller_loop_failed"},
                )
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                continue

    async def run_once(self, *, batch_size: int | None = None) -> dict[str, Any]:
        self.refresh_config()
        batch_size = max(1, int(batch_size)) if batch_size is not None else self.batch_size

        started_at = self._utc_now()
        run_id = str(uuid4())
        summary = {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": None,
            "scanned_count": 0,
            "candidate_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "merged_count": 0,
            "dry_run": self.dry_run,
            "skipped_reason": None,
            "error": None,
        }

        self.archive_service = self.archive_service or await get_server_archive_service(self.bot)
        self.memory_service = self.memory_service or await get_memory_service(self.bot)
        if not self.enabled:
            summary["skipped_reason"] = "disabled"
            return summary
        if not getattr(self.archive_service, "enabled", False):
            summary["skipped_reason"] = "archive_disabled"
            return summary
        if not getattr(self.memory_service, "enabled", False):
            summary["skipped_reason"] = "memory_disabled"
            return summary

        store = self.archive_service.store
        await store.start_distiller_run(run_id, started_at=started_at)

        try:
            scopes = await store.list_distiller_scopes(limit=batch_size, guild_id=None)
            remaining = batch_size
            for scope in scopes:
                if remaining <= 0:
                    break
                messages = await self._fetch_scope_batch(store, scope, remaining)
                if not messages:
                    continue
                for window in self._group_windows(messages):
                    if remaining <= 0:
                        break
                    window_size = len(window.messages)
                    remaining -= window_size
                    summary["scanned_count"] += window_size
                    try:
                        distilled = self._distill_window(window.messages)
                    except (AttributeError, TypeError, ValueError, RuntimeError, OSError) as exc:
                        logger.warning(
                            "Skipping bad archive window during distillation",
                            extra={
                                "subsys": "memory",
                                "event": "memory_distiller_window_error",
                                "detail": {
                                    "guild_id": scope.get("guild_id"),
                                    "channel_id": scope.get("channel_id"),
                                    "thread_id": scope.get("thread_id"),
                                    "author_id": scope.get("author_id"),
                                },
                            },
                            exc_info=True,
                        )
                        summary["error"] = summary["error"] or str(exc)
                        continue

                    summary["candidate_count"] += len(distilled)
                    accepted: list[MemoryCandidate] = []
                    window_candidates = distilled[: self.max_memories_per_window]
                    summary["rejected_count"] += max(0, len(distilled) - len(window_candidates))
                    for candidate in window_candidates:
                        curated = self._curate_candidate(candidate)
                        if curated is None:
                            summary["rejected_count"] += 1
                            continue
                        if curated.confidence < self.min_confidence:
                            summary["rejected_count"] += 1
                            continue
                        accepted.append(curated)

                    summary["accepted_count"] += len(accepted)
                    if accepted and not self.dry_run:
                        persist_stats = await self.memory_service._persist_batch(accepted)
                        if isinstance(persist_stats, dict):
                            summary["merged_count"] += int(persist_stats.get("merged", 0))
                    elif not accepted:
                        pass

                    last = window.messages[-1]
                    await store.upsert_distiller_state(
                        guild_id=str(last["guild_id"]),
                        channel_id=str(last["channel_id"]) if last.get("channel_id") is not None else None,
                        thread_id=str(last["thread_id"]) if last.get("thread_id") is not None else None,
                        author_id=str(last["author_id"]) if last.get("author_id") is not None else None,
                        last_processed_message_id=str(last["message_id"]),
                        last_processed_created_at=str(last["created_at"]),
                        error=None,
                    )
        except Exception as exc:
            summary["error"] = str(exc)
            logger.exception(
                "Memory distiller run failed",
                extra={"subsys": "memory", "event": "memory_distiller_run_failed"},
            )
        finally:
            summary["finished_at"] = self._utc_now()
            await store.finish_distiller_run(
                run_id,
                finished_at=summary["finished_at"],
                scanned_count=int(summary["scanned_count"]),
                candidate_count=int(summary["candidate_count"]),
                accepted_count=int(summary["accepted_count"]),
                rejected_count=int(summary["rejected_count"]),
                merged_count=int(summary["merged_count"]),
                error=summary["error"],
            )
            self.last_run = summary
        return summary

    async def get_status(self, *, guild_id: str | None = None) -> dict[str, Any]:
        self.refresh_config()
        archive_enabled = bool(getattr(self.archive_service, "enabled", False)) if self.archive_service else False
        memory_enabled = bool(getattr(self.memory_service, "enabled", False)) if self.memory_service else False
        backlog = None
        if self.archive_service is not None:
            try:
                backlog = await self.archive_service.store.count_distiller_backlog(guild_id=guild_id)
            except (AttributeError, TypeError, ValueError, RuntimeError, OSError):
                logger.debug("Failed to calculate distiller backlog", exc_info=True)
        latest_run = None
        if self.archive_service is not None:
            try:
                latest_run = await self.archive_service.store.latest_distiller_run()
            except (AttributeError, TypeError, ValueError, RuntimeError, OSError):
                logger.debug("Failed to fetch latest distiller run", exc_info=True)
        return {
            "enabled": self.enabled,
            "started": self._started,
            "dry_run": self.dry_run,
            "archive_enabled": archive_enabled,
            "memory_enabled": memory_enabled,
            "batch_size": self.batch_size,
            "interval_seconds": self.interval_seconds,
            "window_messages": self.window_messages,
            "min_confidence": self.min_confidence,
            "max_memories_per_window": self.max_memories_per_window,
            "exclude_bot_messages": self.exclude_bot_messages,
            "backlog": backlog,
            "last_run": self.last_run or latest_run,
        }

    async def _fetch_scope_batch(self, store: Any, scope: dict[str, Any], remaining: int) -> list[dict[str, Any]]:
        state = await store.get_distiller_state(
            guild_id=str(scope["guild_id"]),
            channel_id=str(scope["channel_id"]) if scope.get("channel_id") is not None else None,
            thread_id=str(scope["thread_id"]) if scope.get("thread_id") is not None else None,
            author_id=str(scope["author_id"]) if scope.get("author_id") is not None else None,
        )
        return await store.fetch_distiller_messages(
            guild_id=str(scope["guild_id"]),
            channel_id=str(scope["channel_id"]) if scope.get("channel_id") is not None else None,
            thread_id=str(scope["thread_id"]) if scope.get("thread_id") is not None else None,
            author_id=str(scope["author_id"]),
            after_created_at=str(state["last_processed_created_at"]) if state and state.get("last_processed_created_at") else None,
            after_message_id=str(state["last_processed_message_id"]) if state and state.get("last_processed_message_id") else None,
            limit=min(self.window_messages, remaining),
        )

    def _group_windows(self, messages: list[dict[str, Any]]) -> list[DistillerWindow]:
        if not messages:
            return []
        ordered = sorted(
            messages,
            key=lambda item: (
                str(item.get("created_at") or ""),
                str(item.get("message_id") or ""),
            ),
        )
        windows: list[DistillerWindow] = []
        current: list[dict[str, Any]] = []
        previous_at: datetime | None = None
        for message in ordered:
            created_at = self._parse_datetime(message.get("created_at"))
            if current:
                gap = (created_at - previous_at).total_seconds() if created_at and previous_at else 0
                if len(current) >= self.window_messages or gap > 15 * 60:
                    windows.append(DistillerWindow(messages=current))
                    current = []
            current.append(message)
            previous_at = created_at
        if current:
            windows.append(DistillerWindow(messages=current))
        return windows

    def _distill_window(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        distilled: list[dict[str, Any]] = []
        seen: set[str] = set()
        for message in messages:
            if self.exclude_bot_messages and self._as_bool(message.get("author_bot")):
                continue
            text = self._normalize_text(message.get("clean_content") or message.get("content") or "")
            if not text or len(text) < 16:
                continue
            if self._looks_blocked(text):
                continue
            if not self._looks_durable(text):
                continue
            normalized = text.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            distilled.append(message | {"distilled_text": text})
        return distilled

    def _curate_candidate(self, message: dict[str, Any]) -> MemoryCandidate | None:
        text = str(message.get("distilled_text") or "").strip()
        if not text:
            return None
        metadata = {
            "archive_message_id": message.get("message_id"),
            "archive_created_at": message.get("created_at"),
            "distilled_from_archive": True,
            "distiller_source": "server_archive",
        }
        return self.curator.curate_inferred_candidate(
            user_id=str(message.get("author_id") or ""),
            text=text,
            guild_id=str(message.get("guild_id") or "") or None,
            channel_id=str(message.get("channel_id") or "") or None,
            thread_id=str(message.get("thread_id") or "") or None,
            source_message_id=str(message.get("message_id") or "") or None,
            metadata=metadata,
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text or "").split()).strip()

    @staticmethod
    def _looks_blocked(text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in _BLOCKED_HINTS)

    @staticmethod
    def _looks_durable(text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in _DURABLE_HINTS)

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if value in (None, ""):
            return None
        text = str(value)
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError:
            return None

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return value in (1, "1", "true", "True", "yes", "on")

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat()


_service: MemoryArchiveDistiller | None = None
_service_lock = asyncio.Lock()


async def get_memory_distiller(bot: Any | None = None) -> MemoryArchiveDistiller:
    global _service
    async with _service_lock:
        if _service is None:
            _service = MemoryArchiveDistiller(bot=bot)
        elif bot is not None:
            _service.bot = bot
        return _service


async def start_memory_distiller(bot: Any | None = None) -> MemoryArchiveDistiller:
    service = await get_memory_distiller(bot)
    await service.start(bot)
    return service


async def stop_memory_distiller() -> None:
    global _service
    if _service is None:
        return
    await _service.stop()


async def run_memory_distiller_once(*, batch_size: int | None = None) -> dict[str, Any]:
    service = await get_memory_distiller()
    return await service.run_once(batch_size=batch_size)


async def get_memory_distiller_status(*, guild_id: str | None = None) -> dict[str, Any]:
    service = await get_memory_distiller()
    return await service.get_status(guild_id=guild_id)
