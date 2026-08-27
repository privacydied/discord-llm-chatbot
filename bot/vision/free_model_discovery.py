"""Auto-discovery of free image-capable models on OpenRouter. [REH][CMV][PA]

Why this exists: free VL models on OpenRouter get delisted, renamed or 404'd
without warning, which silently takes vision down until someone hand-edits
``VISION_FALLBACK_MODELS`` in ``.env``. This module polls OpenRouter's public
model catalogue, keeps the currently-live free image->text models on disk, and
feeds them into the vision fallback ladder so the bot self-heals.

Public surface:
- ``discover_free_vision_models()``  -- async fetch + cache refresh
- ``get_cached_free_vision_models()`` -- sync, non-blocking read for ladder build
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

logger = get_logger(__name__)

# --- Constants [CMV] -------------------------------------------------------

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DISCOVERY_QUERY: dict[str, str] = {
    "max_price": "0",
    "input_modalities": "image",
    "output_modalities": "text",
}
DEFAULT_CACHE_PATH = "vision_data/free_vision_models.json"
DEFAULT_TTL_S = 6 * 60 * 60  # 6h
DEFAULT_REFRESH_INTERVAL_S = 6 * 60 * 60
DEFAULT_MAX_MODELS = 6
DEFAULT_TIMEOUT_S = 10.0
MAX_FETCH_ATTEMPTS = 3
RETRY_BASE_DELAY_S = 1.5
RETRY_MAX_DELAY_S = 8.0
FAILURE_RETRY_INTERVAL_S = 900.0  # retry sooner after a failed refresh
MIN_CONTEXT_LENGTH = 8000

# Models that technically accept images but are useless as a VL rung:
# safety/moderation classifiers, embeddings, audio/music generators, etc.
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
)

# Preferred families, ranked. Matching models sort first (stable within a tier).
_PREFERRED_SUBSTRINGS: tuple[str, ...] = (
    "-vl",
    "vision",
    "pixtral",
    "qwen",
    "gemma",
    "llama",
    "mistral",
    "gemini",
)

_ENV_ENABLED = "VISION_AUTO_DISCOVERY"
_ENV_CACHE_PATH = "VISION_DISCOVERY_CACHE_PATH"
_ENV_TTL = "VISION_DISCOVERY_TTL_S"
_ENV_INTERVAL = "VISION_DISCOVERY_REFRESH_S"
_ENV_MAX_MODELS = "VISION_DISCOVERY_MAX_MODELS"
_ENV_TIMEOUT = "VISION_DISCOVERY_TIMEOUT_S"

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


def _is_free(pricing: dict[str, Any]) -> bool:
    """All priced dimensions must be zero (OpenRouter returns strings)."""
    for key in ("prompt", "completion", "request", "image", "input_cache_read", "input_cache_write"):
        raw = pricing.get(key)
        if raw in (None, ""):
            continue
        try:
            if float(raw) > 0:
                return False
        except (ValueError, TypeError):
            return False
    return True


def _is_usable_vision_model(entry: dict[str, Any]) -> bool:
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
    if "image" not in inputs or outputs != {"text"}:
        return False

    if not _is_free(entry.get("pricing") or {}):
        return False

    try:
        if int(entry.get("context_length") or 0) < MIN_CONTEXT_LENGTH:
            return False
    except (ValueError, TypeError):
        return False
    return True


def _rank_key(entry: dict[str, Any]) -> tuple[int, int, int]:
    """Sort: preferred VL families first, then bigger context, then newer."""
    lowered = str(entry.get("id", "")).lower()
    tier = next((i for i, kw in enumerate(_PREFERRED_SUBSTRINGS) if kw in lowered), len(_PREFERRED_SUBSTRINGS))
    try:
        context = int(entry.get("context_length") or 0)
    except (ValueError, TypeError):
        context = 0
    try:
        created = int(entry.get("created") or 0)
    except (ValueError, TypeError):
        created = 0
    return (tier, -context, -created)


def _select_models(payload: dict[str, Any], limit: int) -> list[str]:
    entries = [e for e in (payload.get("data") or []) if isinstance(e, dict) and _is_usable_vision_model(e)]
    entries.sort(key=_rank_key)
    seen: set[str] = set()
    models: list[str] = []
    for entry in entries:
        model_id = str(entry["id"]).strip()
        if model_id in seen:
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
    models = [str(m) for m in (raw.get("models") or []) if str(m).strip()]
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
        logger.warning("vision.discovery.cache_write_failed error=%s", exc)


def get_cached_free_vision_models() -> list[str]:
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
    headers = {"Accept": "application/json", "User-Agent": "discord-llm-chatbot/vision-discovery"}
    api_key = (os.getenv("OPENROUTER_API_KEY") or os.getenv("VISION_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
                response = await client.get(OPENROUTER_MODELS_URL, params=DISCOVERY_QUERY, headers=headers)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            last_exc = exc
            logger.warning("vision.discovery.fetch_attempt_failed attempt=%d/%d error=%s", attempt, MAX_FETCH_ATTEMPTS, exc)
            if attempt < MAX_FETCH_ATTEMPTS:
                delay = min(RETRY_MAX_DELAY_S, RETRY_BASE_DELAY_S * (2 ** (attempt - 1)))
                await asyncio.sleep(delay * (0.5 + random.random()))  # noqa: S311  # nosec B311 - jitter only
    raise RuntimeError(f"OpenRouter model discovery failed after {MAX_FETCH_ATTEMPTS} attempts: {last_exc}")


async def discover_free_vision_models(*, force: bool = False) -> list[str]:
    """Refresh the free image-capable model list. Returns the cached list on failure."""
    global _memory_cache

    if not is_enabled():
        return []
    if not force and cache_is_fresh():
        return get_cached_free_vision_models()

    async with _fetch_lock:
        if not force and cache_is_fresh():
            return get_cached_free_vision_models()
        try:
            payload = await _fetch_catalogue(_env_float(_ENV_TIMEOUT, DEFAULT_TIMEOUT_S))
        except (RuntimeError, asyncio.TimeoutError) as exc:
            logger.error("vision.discovery.failed error=%s (keeping previous ladder)", exc)
            return get_cached_free_vision_models()

        models = _select_models(payload, _env_int(_ENV_MAX_MODELS, DEFAULT_MAX_MODELS))
        if not models:
            logger.warning("vision.discovery.empty_result (keeping previous ladder)")
            return get_cached_free_vision_models()

        previous = get_cached_free_vision_models()
        _memory_cache = (models, time.time())
        await _write_cache_file(models)
        if models != previous:
            logger.info("vision.discovery.updated models=%s previous=%s", models, previous)
        else:
            logger.debug("vision.discovery.unchanged models=%s", models)
        return list(models)


async def refresh_and_apply(*, force: bool = False) -> list[str]:
    """Discover models and rebuild the retry manager's vision ladder."""
    models = await discover_free_vision_models(force=force)
    if not models:
        return []
    try:
        from bot.enhanced_retry import get_retry_manager

        summary = get_retry_manager().refresh_from_env()
        logger.info("vision.discovery.ladder_applied vision=%s", summary.get("vision"))
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        logger.warning("vision.discovery.ladder_apply_failed error=%s", exc)
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
            logger.error("vision.discovery.loop_error error=%s", exc, exc_info=True)
            sleep_for = FAILURE_RETRY_INTERVAL_S
        await asyncio.sleep(sleep_for)


async def start_discovery_refresh() -> None:
    """Start the periodic discovery loop (idempotent)."""
    global _refresh_task

    if not is_enabled():
        logger.info("vision.discovery.disabled")
        return
    if _refresh_task is not None and not _refresh_task.done():
        return
    _refresh_task = asyncio.create_task(_refresh_loop(), name="vision_model_discovery")
    logger.info("vision.discovery.started interval_s=%.0f", _env_float(_ENV_INTERVAL, DEFAULT_REFRESH_INTERVAL_S))


async def stop_discovery_refresh() -> None:
    """Cancel the periodic discovery loop. [RM]"""
    global _refresh_task

    if _refresh_task is None:
        return
    _refresh_task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await _refresh_task
    _refresh_task = None
    logger.info("vision.discovery.stopped")


def _reset_for_tests() -> None:
    global _memory_cache, _refresh_task
    _memory_cache = None
    _refresh_task = None
