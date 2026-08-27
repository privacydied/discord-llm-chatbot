"""Liveness probing + quarantine for free OpenRouter vision models. [REH][SFT][PA]

Being listed in OpenRouter's catalogue does not mean a model will answer an
image request. Observed in the wild on ``:free`` VL slugs:

- ``404 No endpoints found`` -- slug listed, no provider serving it
- ``403 ... only available on agentic harnesses`` -- not callable via the API
- ``429 ... temporarily rate-limited upstream`` -- transient, model is fine

So each discovery round probes candidates with a 1x1 PNG. Hard failures are
quarantined (they will not come back on their own); transient failures stay in
the ladder because the circuit breaker already handles them.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from bot.utils.logging import get_logger

logger = get_logger(__name__)

# --- Constants [CMV] -------------------------------------------------------

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
# 1x1 PNG - the smallest valid image input, so a probe costs ~nothing.
PROBE_IMAGE_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
PROBE_PROMPT = "Reply with one word."
PROBE_MAX_TOKENS = 8
DEFAULT_PROBE_TIMEOUT_S = 20.0
DEFAULT_PROBE_CONCURRENCY = 4
DEFAULT_QUARANTINE_TTL_S = 24 * 60 * 60  # 24h
DEFAULT_QUARANTINE_PATH = "vision_data/vision_model_quarantine.json"

# Statuses that mean "this model will not serve image requests" (not a blip).
HARD_FAIL_STATUSES: frozenset[int] = frozenset({400, 403, 404, 410, 422})
# Statuses that mean "our account/key is the problem" -> abort probing entirely
# rather than quarantining perfectly good models. [REH]
ACCOUNT_FAIL_STATUSES: frozenset[int] = frozenset({401, 402})

_ENV_PROBE_ENABLED = "VISION_DISCOVERY_PROBE"
_ENV_PROBE_TIMEOUT = "VISION_DISCOVERY_PROBE_TIMEOUT_S"
_ENV_PROBE_CONCURRENCY = "VISION_DISCOVERY_PROBE_CONCURRENCY"
_ENV_QUARANTINE_TTL = "VISION_DISCOVERY_QUARANTINE_TTL_S"
_ENV_QUARANTINE_PATH = "VISION_DISCOVERY_QUARANTINE_PATH"


@dataclass
class ProbeReport:
    """Outcome of probing a batch of candidate models."""

    good: list[str] = field(default_factory=list)
    transient: list[str] = field(default_factory=list)
    dead: dict[str, str] = field(default_factory=dict)  # model -> reason
    skipped: bool = False  # True when probing could not run (no key / disabled)

    @property
    def usable(self) -> list[str]:
        """Verified-live models first, then ones that merely blipped."""
        return [*self.good, *self.transient]


# --- Env helpers -----------------------------------------------------------


def probing_enabled() -> bool:
    return os.getenv(_ENV_PROBE_ENABLED, "1").strip().lower() not in ("0", "false", "no", "off")


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


def quarantine_path() -> Path:
    return Path(os.getenv(_ENV_QUARANTINE_PATH, "").strip() or DEFAULT_QUARANTINE_PATH)


def resolve_openrouter_key() -> str:
    """Find a usable OpenRouter key without assuming one env var name. [REH]

    Deployments differ: some set OPENROUTER_API_KEY, others point OPENAI_API_BASE
    at OpenRouter and put the key in OPENAI_API_KEY. A key belonging to a
    different provider is never returned -- probing with it would 401 and quarantine
    every model.
    """
    direct = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if direct:
        return direct
    try:
        from bot.config import load_config

        config = load_config()
    except (ImportError, AttributeError, ValueError, OSError):
        config = {}

    direct = str(config.get("OPENROUTER_API_KEY") or "").strip()
    if direct:
        return direct
    for key_name, base_name in (("OPENAI_API_KEY", "OPENAI_API_BASE"), ("VISION_API_KEY", "VISION_API_BASE")):
        base = str(config.get(base_name) or os.getenv(base_name) or "").lower()
        candidate = str(config.get(key_name) or os.getenv(key_name) or "").strip()
        if candidate and "openrouter" in base:
            return candidate
    return ""


# --- Quarantine store ------------------------------------------------------


def load_quarantine() -> dict[str, dict[str, Any]]:
    """Read the quarantine map, dropping entries whose ban has expired."""
    from bot.atomic_json import read_json_safe

    raw = read_json_safe(quarantine_path(), default={}) or {}
    entries = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(entries, dict):
        return {}
    now = time.time()
    live: dict[str, dict[str, Any]] = {}
    for model, meta in entries.items():
        if not isinstance(meta, dict):
            continue
        try:
            until = float(meta.get("until") or 0.0)
        except (ValueError, TypeError):
            continue
        if until > now:
            live[str(model)] = meta
    return live


async def save_quarantine(entries: dict[str, dict[str, Any]]) -> None:
    from bot.atomic_json import write_json_atomic

    try:
        await write_json_atomic(quarantine_path(), {"models": entries, "updated_at": time.time()})
    except (OSError, ValueError) as exc:
        logger.warning("vision.probe.quarantine_write_failed error=%s", exc)


async def quarantine_models(dead: dict[str, str]) -> None:
    """Bench hard-failed models for the quarantine TTL. [REH]"""
    if not dead:
        return
    ttl = _env_float(_ENV_QUARANTINE_TTL, DEFAULT_QUARANTINE_TTL_S)
    entries = load_quarantine()
    now = time.time()
    for model, reason in dead.items():
        entries[model] = {"reason": reason[:200], "since": now, "until": now + ttl}
    await save_quarantine(entries)
    logger.info("vision.probe.quarantined models=%s ttl_s=%.0f", sorted(dead), ttl)


def filter_quarantined(models: list[str]) -> list[str]:
    """Drop models currently serving a quarantine ban."""
    banned = load_quarantine()
    if not banned:
        return list(models)
    kept = [m for m in models if m not in banned]
    dropped = [m for m in models if m in banned]
    if dropped:
        logger.debug("vision.probe.skipping_quarantined models=%s", dropped)
    return kept


# --- Probing ---------------------------------------------------------------


def _probe_body(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": PROBE_MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROBE_PROMPT},
                    {"type": "image_url", "image_url": {"url": PROBE_IMAGE_DATA_URL}},
                ],
            },
        ],
    }


def _classify(status: int, body: str) -> tuple[str, str]:
    """Map an HTTP response to (verdict, reason): good | transient | dead | account."""
    if status == 200:
        return ("good", "")
    if status in ACCOUNT_FAIL_STATUSES:
        return ("account", f"HTTP {status}: {body[:120]}")
    if status in HARD_FAIL_STATUSES:
        return ("dead", f"HTTP {status}: {body[:160]}")
    return ("transient", f"HTTP {status}: {body[:120]}")


async def _probe_one(client: httpx.AsyncClient, model: str, key: str) -> tuple[str, str, str]:
    """Probe one model. Returns (model, verdict, reason)."""
    try:
        response = await client.post(
            OPENROUTER_CHAT_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=_probe_body(model),
        )
    except httpx.HTTPError as exc:
        # Network trouble is ours, not the model's -> never quarantine on it.
        return (model, "transient", f"{type(exc).__name__}: {exc}")

    verdict, reason = _classify(response.status_code, response.text)
    if verdict == "good" and not _has_choices(response):
        return (model, "transient", "200 with no choices in response")
    return (model, verdict, reason)


def _has_choices(response: httpx.Response) -> bool:
    try:
        payload = response.json()
    except ValueError:
        return False
    choices = payload.get("choices") if isinstance(payload, dict) else None
    return bool(choices)


def _unprobed(models: list[str]) -> ProbeReport:
    """Report used when probing cannot run: keep every candidate, quarantine none."""
    return ProbeReport(good=[], transient=list(models), dead={}, skipped=True)


async def _run_probes(models: list[str], key: str) -> list[Any]:
    """Probe every model concurrently under a bounded semaphore. [PA][RM]"""
    timeout = _env_float(_ENV_PROBE_TIMEOUT, DEFAULT_PROBE_TIMEOUT_S)
    semaphore = asyncio.Semaphore(_env_int(_ENV_PROBE_CONCURRENCY, DEFAULT_PROBE_CONCURRENCY))

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def guarded(model: str) -> tuple[str, str, str]:
            async with semaphore:
                return await _probe_one(client, model, key)

        return await asyncio.gather(*(guarded(m) for m in models), return_exceptions=True)


def _collect(results: list[Any]) -> tuple[ProbeReport, str]:
    """Fold probe results into a report plus any account-level failure reason."""
    report = ProbeReport()
    account_failure = ""
    for item in results:
        if isinstance(item, BaseException):
            logger.debug("vision.probe.unexpected_error error=%s", item)
            continue
        model, verdict, reason = item
        if verdict == "good":
            report.good.append(model)
        elif verdict == "dead":
            report.dead[model] = reason
        elif verdict == "account":
            account_failure = reason
        else:
            report.transient.append(model)
    return report, account_failure


async def probe_models(models: list[str]) -> ProbeReport:
    """Probe candidates concurrently with a 1x1 image. Never raises. [REH][PA]"""
    if not models:
        return ProbeReport()
    if not probing_enabled():
        return _unprobed(models)

    key = resolve_openrouter_key()
    if not key:
        logger.info("vision.probe.no_key (skipping liveness probe)")
        return _unprobed(models)

    report, account_failure = _collect(await _run_probes(models, key))
    if account_failure:
        # Bad/exhausted key: the models are innocent, so keep them all and bail out.
        logger.warning("vision.probe.account_error error=%s (keeping unprobed ladder)", account_failure)
        return _unprobed(models)

    logger.info("vision.probe.done good=%s transient=%s dead=%s", report.good, report.transient, sorted(report.dead))
    return report


# --- Env-configured candidates ---------------------------------------------

_ENV_LADDER_VARS: tuple[str, ...] = ("VL_MODEL", "VISION_FALLBACK_MODELS")


def env_ladder_candidates() -> list[str]:
    """Free models the operator configured by hand (VL_MODEL / VISION_FALLBACK_MODELS).

    They are probed alongside the discovered ones so a slug that has gone 404 in
    ``.env`` gets quarantined and drops out of the ladder, instead of being tried
    on every image until someone notices. Entries may carry a ``provider|model``
    prefix; non-``:free`` entries are ignored (this module only vets free models).
    """
    candidates: list[str] = []
    for var in _ENV_LADDER_VARS:
        for raw in (os.getenv(var) or "").split(","):
            model = raw.split("|", 1)[-1].strip()
            if model.lower().endswith(":free") and model not in candidates:
                candidates.append(model)
    return candidates
