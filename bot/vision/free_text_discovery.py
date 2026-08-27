"""Auto-discovery of free text models on OpenRouter. [REH][CMV][PA]

Why this exists: the text ladder in ``TEXT_FALLBACK_MODELS`` goes stale the same
way the vision one does -- ``:free`` slugs get delisted, renamed, or start 503ing
without warning. This module polls OpenRouter's public model catalogue, keeps the
currently-live free text models on disk, and feeds them into the text fallback
ladder so the bot self-heals.

Public surface:
- ``discover_free_text_models()``  -- async fetch + cache refresh
- ``get_cached_free_text_models()`` -- sync, non-blocking read for ladder build
- ``start_discovery_refresh()`` / ``stop_discovery_refresh()`` -- background loop
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import httpx

from bot.utils.logging import get_logger
from bot.vision.model_ranking import param_tier

logger = get_logger(__name__)

# --- Constants [CMV] -------------------------------------------------------

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DISCOVERY_QUERY: dict[str, str] = {
    "max_price": "0",
    "output_modalities": "text",
    "variant": "free",
}
FREE_VARIANT_SUFFIX = ":free"
DEFAULT_CACHE_PATH = "vision_data/free_text_models.json"
DEFAULT_TTL_S = 6 * 60 * 60  # 6h
DEFAULT_REFRESH_INTERVAL_S = 6 * 60 * 60
DEFAULT_MAX_MODELS = 6
PROBE_CANDIDATE_MULTIPLIER = 3
DEFAULT_TIMEOUT_S = 10.0
MAX_FETCH_ATTEMPTS = 3
RETRY_BASE_DELAY_S = 1.5
RETRY_MAX_DELAY_S = 8.0
FAILURE_RETRY_INTERVAL_S = 900.0
MIN_CONTEXT_LENGTH = 8000

# Models that accept text but are useless as a chat rung.
_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "content-safety",
    "guard",
    "moderation",
    "embed",
    "rerank",
    "lyria",
    "tts",
    "whisper",
    "stable-diffusion",
    "flux",
    "text-to-",
    "speech",
    "audio",
    "voice",
)

# Preferred families, ranked. Matching models sort first (stable within a tier).
_PREFERRED_SUBSTRINGS: tuple[str, ...] = (
    "deepseek",
    "qwen",
    "gemma",
    "llama",
    "mistral",
    "gemini",
    "gpt",
    "claude",
    "phi",
    "nemotron",
    "ling",
    "glm",
    "kimi",
    "grok",
    "aya",
)

_ENV_ENABLED = "TEXT_AUTO_DISCOVERY"
_ENV_CACHE_PATH = "TEXT_DISCOVERY_CACHE_PATH"
_ENV_TTL = "TEXT_DISCOVERY_TTL_S"
_ENV_INTERVAL = "TEXT_DISCOVERY_REFRESH_S"
_ENV_MAX_MODELS = "TEXT_DISCOVERY_MAX_MODELS"
_ENV_TIMEOUT = "TEXT_DISCOVERY_TIMEOUT_S"

# In-process cache: (models, fetched_at)
_memory_cache: tuple[list[str], float] | None = None
_refresh_task: asyncio.Task | None = None
_fetch_lock = asyncio.Lock()


# --- Env helpers -----------------------------------------------------------


def is_enabled() -> bool:
    """Auto-discovery on unless explicitly disabled."""
    return os.getenv(_ENV_ENABLED, "1").strip().lower() not in ("0", "false", "no", "off")


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, "").strip() or default)
    except (ValueError, TypeError):
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    try:
        value = int(float(os.getenv(name, "").strip() or default))
    except (ValueError, TypeError):
        return default
    return value if value > 0 else default


def cache_path() -> Path:
    return Path(os.getenv(_ENV_CACHE_PATH, "").strip() or DEFAULT_CACHE_PATH)


# --- Filtering / ranking ---------------------------------------------------


def is_free_slug(model_id: str) -> bool:
    """True only for OpenRouter ``:free`` variants. [SFT]"""
    return bool(model_id) and model_id.strip().lower().endswith(FREE_VARIANT_SUFFIX)


def _is_free(pricing: dict[str, Any], model_id: str = "") -> bool:
    """Free means: ``:free`` variant AND every priced dimension is exactly zero."""
    if model_id and not is_free_slug(model_id):
        return False
    if not isinstance(pricing, dict):
        return False
    for raw in pricing.values():
        if raw in (None, ""):
            continue
        try:
            if float(raw) != 0.0:
                return False
        except (ValueError, TypeError):
            return False
    return True


def _is_usable_text_model(entry: dict[str, Any]) -> bool:
    """Client-side re-validation of the catalogue filters. [IV]"""
    model_id = str(entry.get("id") or "").strip()
    if not model_id:
        return False
    lowered = model_id.lower()
    if any(bad in lowered for bad in _EXCLUDE_SUBSTRINGS):
        return False

    arch = entry.get("architecture") or {}
    inputs = {str(m).lower() for m in (arch.get("input_modalities") or [])}
    outputs = {str(m).lower() for m in (arch.get("output_modalities") or [])}
    # Text models: input must include text (or be empty = default), output must be text.
    if inputs and "text" not in inputs:
        return False
    if outputs and outputs != {"text"}:
        return False

    if not _is_free(entry.get("pricing") or {}, model_id):
        return False

    try:
        if int(entry.get("context_length") or 0) < MIN_CONTEXT_LENGTH:
            return False
    except (ValueError, TypeError):
        return False
    return True


def _rank_key(entry: dict[str, Any]) -> tuple[int, int, int, int]:
    """Sort: bigger parameter count first, then preferred families, then bigger context, then newer."""
    lowered = str(entry.get("id", "")).lower()
    size_tier = param_tier(lowered)
    family_tier = next((i for i, kw in enumerate(_PREFERRED_SUBSTRINGS) if kw in lowered), len(_PREFERRED_SUBSTRINGS))
    try:
        context = int(entry.get("context_length") or 0)
    except (ValueError, TypeError):
        context = 0
    try:
        created = int(entry.get("created") or 0)
    except (ValueError, TypeError):
        created = 0
    return (size_tier, family_tier, -context, -created)


def _select_models(payload: dict[str, Any], limit: int) -> list[str]:
    """Rank and cap the usable free models. Non-``:free`` slugs can never survive."""
    entries = [e for e in (payload.get("data") or []) if isinstance(e, dict) and _is_usable_text_model(e)]
    entries.sort(key=_rank_key)
    seen: set[str] = set()
    models: list[str] = []
    for entry in entries:
        model_id = str(entry["id"]).strip()
        if model_id in seen or not is_free_slug(model_id):
            continue
        seen.add(model_id)
        models.append(model_id)
        if len(models) >= limit:
            break
    return models


# --- Cache I/O -------------------------------------------------------------


def _read_cache_file() -> tuple[list[str], float]:
    path = cache_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], 0.0
    models = [str(m).strip() for m in (raw.get("models") or []) if is_free_slug(str(m))]
    try:
        fetched_at = float(raw.get("fetched_at") or 0.0)
    except (ValueError, TypeError):
        fetched_at = 0.0
    return models, fetched_at


async def _write_cache_file(models: list[str]) -> None:
    from bot.atomic_json import write_json_atomic

    payload = {"models": models, "fetched_at": time.time(), "source": OPENROUTER_MODELS_URL}
    try:
        await write_json_atomic(cache_path(), payload)
    except (OSError, ValueError) as exc:
        logger.warning("text.discovery.cache_write_failed error=%s", exc)


def get_cached_free_text_models() -> list[str]:
    """Sync, non-blocking read of the discovered ladder (memory then disk)."""
    global _memory_cache

    if not is_enabled():
        return []
    if _memory_cache is not None:
        return list(_memory_cache[0])
    models, fetched_at = _read_cache_file()
    if models:
        _memory_cache = (models, fetched_at)
    return list(models)


def cache_is_fresh() -> bool:
    if _memory_cache is None:
        _, fetched_at = _read_cache_file()
    else:
        _, fetched_at = _memory_cache
    return (time.time() - fetched_at) < _env_float(_ENV_TTL, DEFAULT_TTL_S)


# --- Fetch -----------------------------------------------------------------


async def _fetch_catalogue(timeout_s: float) -> dict[str, Any]:
    """GET the model catalogue with bounded retries + jittered backoff. [REH]"""
    last_exc: Exception | None = None
    headers = {"Accept": "application/json", "User-Agent": "discord-llm-chatbot/text-discovery"}

    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
                response = await client.get(OPENROUTER_MODELS_URL, params=DISCOVERY_QUERY, headers=headers)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            last_exc = exc
            logger.warning("text.discovery.fetch_attempt_failed attempt=%d/%d error=%s", attempt, MAX_FETCH_ATTEMPTS, exc)
            if attempt < MAX_FETCH_ATTEMPTS:
                delay = min(RETRY_MAX_DELAY_S, RETRY_BASE_DELAY_S * (2 ** (attempt - 1)))
                await asyncio.sleep(delay * (0.5 + random.random()))  # nosec B311 - jitter only
    raise RuntimeError(f"OpenRouter model discovery failed after {MAX_FETCH_ATTEMPTS} attempts: {last_exc}")


async def _verify_candidates(candidates: list[str], limit: int) -> list[str]:
    """Liveness-probe candidates with a trivial text prompt; return the top `limit`."""
    try:
        from bot.vision.free_model_probe import (
            filter_quarantined,
            probe_text_models,
            quarantine_text_models,
        )

        try:
            fresh = filter_quarantined(candidates, modality="text")
        except TypeError:
            # Older signature without modality kwarg
            fresh = filter_quarantined(candidates)
        if not fresh:
            logger.warning("text.discovery.all_candidates_quarantined (probing anyway)")
            fresh = list(candidates)
        # Also vet operator-configured text rungs in the same round.
        text_env_candidates = []
        for raw in (os.getenv("TEXT_FALLBACK_MODELS") or "").split(","):
            model = raw.split("|", 1)[-1].strip()
            if model.lower().endswith(":free") and model not in fresh:
                text_env_candidates.append(model)
        extras = [m for m in filter_quarantined(text_env_candidates, modality="text") if m not in fresh]
        report = await probe_text_models([*fresh, *extras])
        await quarantine_text_models(report.dead)
        discovered = set(fresh)
        verified_good = set(report.good)
        usable = [m for m in report.usable if m in discovered or m in verified_good]
        return usable[:limit]
    except (ImportError, OSError, ValueError, RuntimeError, asyncio.TimeoutError) as exc:
        logger.warning("text.discovery.probe_failed error=%s (using unprobed candidates)", exc)
        return candidates[:limit]


async def _commit(models: list[str]) -> list[str]:
    """Persist the verified ladder to memory + disk and log what changed."""
    global _memory_cache

    previous = get_cached_free_text_models()
    _memory_cache = (models, time.time())
    await _write_cache_file(models)
    if models != previous:
        logger.info("text.discovery.updated models=%s previous=%s", models, previous)
    else:
        logger.debug("text.discovery.unchanged models=%s", models)
    return list(models)


async def discover_free_text_models(*, force: bool = False) -> list[str]:
    """Refresh the free text model list. Returns the cached list on failure."""
    if not is_enabled():
        return []
    if not force and cache_is_fresh():
        return get_cached_free_text_models()

    async with _fetch_lock:
        if not force and cache_is_fresh():
            return get_cached_free_text_models()
        try:
            payload = await _fetch_catalogue(_env_float(_ENV_TIMEOUT, DEFAULT_TIMEOUT_S))
        except (RuntimeError, asyncio.TimeoutError) as exc:
            logger.error("text.discovery.failed error=%s (keeping previous ladder)", exc)
            return get_cached_free_text_models()

        limit = _env_int(_ENV_MAX_MODELS, DEFAULT_MAX_MODELS)
        candidates = _select_models(payload, limit * PROBE_CANDIDATE_MULTIPLIER)
        if not candidates:
            logger.warning("text.discovery.empty_result (keeping previous ladder)")
            return get_cached_free_text_models()

        models = await _verify_candidates(candidates, limit)
        if not models:
            logger.warning("text.discovery.no_live_models (keeping previous ladder)")
            return get_cached_free_text_models()

        return await _commit(models)


async def refresh_and_apply(*, force: bool = False) -> list[str]:
    """Discover models and rebuild the retry manager's text ladder."""
    models = await discover_free_text_models(force=force)
    if not models:
        return []
    try:
        from bot.enhanced_retry import get_retry_manager

        summary = get_retry_manager().refresh_from_env()
        logger.info("text.discovery.ladder_applied text=%s", summary.get("text"))
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        logger.warning("text.discovery.ladder_apply_failed error=%s", exc)
    return models


async def _refresh_loop() -> None:
    interval = _env_float(_ENV_INTERVAL, DEFAULT_REFRESH_INTERVAL_S)
    while True:
        try:
            models = await refresh_and_apply()
            sleep_for = interval if models else FAILURE_RETRY_INTERVAL_S
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # defensive: never kill the loop [REH]
            logger.error("text.discovery.loop_error error=%s", exc, exc_info=True)
            sleep_for = FAILURE_RETRY_INTERVAL_S
        await asyncio.sleep(sleep_for)


async def start_discovery_refresh() -> None:
    """Start the periodic discovery loop (idempotent)."""
    global _refresh_task

    if not is_enabled():
        logger.info("text.discovery.disabled")
        return
    if _refresh_task is not None and not _refresh_task.done():
        return
    _refresh_task = asyncio.create_task(_refresh_loop(), name="text_model_discovery")
    logger.info("text.discovery.started interval_s=%.0f", _env_float(_ENV_INTERVAL, DEFAULT_REFRESH_INTERVAL_S))


async def stop_discovery_refresh() -> None:
    """Cancel the periodic discovery loop. [RM]"""
    global _refresh_task

    if _refresh_task is None:
        return
    _refresh_task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await _refresh_task
    _refresh_task = None
    logger.info("text.discovery.stopped")


def _reset_for_tests() -> None:
    global _memory_cache, _refresh_task
    _memory_cache = None
    _refresh_task = None
