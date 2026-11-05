"""Tier D for web extraction: load a walled page in camofox and clear a Cloudflare Turnstile.

Tiers A-C never see past a bot-wall interstitial. When a camofox access key is
configured, this opens the URL in the self-hosted stealth browser, checks the
page for a Turnstile widget (``utils/turnstile.py present()``) and, only if one
is there, runs ``utils/turnstile.py`` (trusted click + wait for the
token) before reading the page text. Everything here degrades to ``None``:
no key, no ``cfx`` helpers, a failed solve, or a page that is still walled all
mean "no text", never an exception. [REH]

``utils/turnstile.py`` drives camofox through the ``cfx`` helpers from the
job-application repo (``CFX_SCRIPTS``), which are configured by process-wide
env vars (CFX_KEY / CFX_TAB / CFX_USER / CFX_URL). Those are set under a lock
for the duration of one solve, so only one page is solved at a time.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

import httpx

from .utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CAMOFOX_URL = "http://127.0.0.1:9377"
CAMOFOX_USER_ID = "ronin-bot"
SESSION_KEY = "webex-turnstile"
OVERALL_TIMEOUT_S = float(os.getenv("WEBEX_TIER_D_TIMEOUT_S", "90"))
MIN_TEXT_CHARS = 200
_LOAD_WAIT_S = 15.0
_AFTER_SOLVE_WAIT_S = 15.0
_WIDGET_WAIT_S = 6.0  # Turnstile renders async; give it a moment to appear
_POLL_S = 1.5
_TEXT_JS = "(document.body && document.body.innerText) || ''"
_READY_JS = "document.readyState"

_LOCK = asyncio.Lock()  # CFX_* env is process-global: one solve at a time


def enabled() -> bool:
    off = {"0", "false", "no"}
    return os.getenv("WEBEX_ENABLE_TIER_D", "1").strip().lower() not in off and bool(_key())


def _key() -> str:
    return (os.getenv("CAMOFOX_ACCESS_KEY") or "").strip().strip('"')


def _base() -> str:
    return (os.getenv("CAMOFOX_URL") or DEFAULT_CAMOFOX_URL).strip().strip('"').rstrip("/")


@contextlib.contextmanager
def _cfx_env(key: str, tab: str, base: str) -> Iterator[None]:
    """Point the cfx helpers at this tab for one solve, then restore the env."""
    wanted = {"CFX_KEY": key, "CFX_TAB": tab, "CFX_USER": CAMOFOX_USER_ID, "CFX_URL": base}
    saved = {name: os.environ.get(name) for name in wanted}
    os.environ.update(wanted)
    try:
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def _load_turnstile() -> Any | None:
    """Import utils.turnstile (needs cfx from CFX_SCRIPTS); None when unavailable."""
    try:
        return importlib.import_module("utils.turnstile")
    except Exception as exc:  # noqa: BLE001 - missing job-application checkout, bad env, ...
        logger.info(f"webex.tier_d.turnstile_unavailable | error={exc!r}")
        return None


def _solve_sync(key: str, tab: str, base: str) -> str:
    """Run utils/turnstile.py on *tab*. -> 'PASS' | 'PASS:already-solved' | 'FAIL:<reason>' | 'NONE'."""
    with _cfx_env(key, tab, base):
        module = _load_turnstile()
        if module is None:
            return "FAIL:turnstile helper unavailable"
        if not module.present():
            return "NONE"
        return str(module.click())


class _Camofox:
    """Minimal async REST client for one camofox tab."""

    def __init__(self, http: httpx.AsyncClient, base: str, key: str) -> None:
        self._http, self._base, self._headers = http, base, {"Authorization": f"Bearer {key}"}
        self.tab: str | None = None

    async def open(self, url: str) -> bool:
        body = {"userId": CAMOFOX_USER_ID, "sessionKey": SESSION_KEY, "url": url}
        resp = await self._http.post(f"{self._base}/tabs", json=body, headers=self._headers)
        if resp.status_code != 200:
            logger.info(f"webex.tier_d.open_failed | status={resp.status_code}")
            return False
        data = resp.json()
        self.tab = str(data.get("tabId") or data.get("id") or "") or None
        return self.tab is not None

    async def evaluate(self, expression: str) -> Any:
        body = {"userId": CAMOFOX_USER_ID, "expression": expression}
        resp = await self._http.post(f"{self._base}/tabs/{quote(str(self.tab))}/evaluate", json=body, headers=self._headers)
        return resp.json().get("result") if resp.status_code == 200 else None

    async def close(self) -> None:
        if self.tab is None:
            return
        with contextlib.suppress(Exception):
            await self._http.delete(f"{self._base}/tabs/{quote(self.tab)}", params={"userId": CAMOFOX_USER_ID}, headers=self._headers)


async def _read_text(page: _Camofox, wait_s: float, *, until_clear: bool) -> str:
    """Poll the page text until it loads (and, if *until_clear*, stops looking like a wall)."""
    from .web_extraction_service import is_bot_wall

    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_s
    text = ""
    while True:
        with contextlib.suppress(Exception):
            ready = await page.evaluate(_READY_JS)
            text = str(await page.evaluate(_TEXT_JS) or "")
            if ready == "complete" and text.strip() and not (until_clear and is_bot_wall(text)):
                return text
        if loop.time() >= deadline:
            return text
        await asyncio.sleep(_POLL_S)


async def _detect_and_solve(page: _Camofox, key: str, base: str) -> str:
    """Wait briefly for a Turnstile widget to render, then solve it. 'NONE' when there isn't one."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _WIDGET_WAIT_S
    while True:
        verdict = await asyncio.to_thread(_solve_sync, key, str(page.tab), base)
        if verdict != "NONE" or loop.time() >= deadline:
            return verdict
        await asyncio.sleep(_POLL_S)


async def _fetch(url: str, key: str, base: str) -> str | None:
    from .web_extraction_service import is_bot_wall

    timeout = httpx.Timeout(25.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as http:
        page = _Camofox(http, base, key)
        try:
            if not await page.open(url):
                return None
            text = await _read_text(page, _LOAD_WAIT_S, until_clear=False)
            # Only a Turnstile that is actually on the page counts: a paywall teaser or a
            # plain 404 loaded fine in camofox and must not be reported as "recovered".
            verdict = await _detect_and_solve(page, key, base)
            logger.info(f"webex.tier_d.turnstile | url={url} result={verdict}")
            if not verdict.startswith("PASS"):
                return None
            text = await _read_text(page, _AFTER_SOLVE_WAIT_S, until_clear=True)
            if is_bot_wall(text) or len(text.strip()) < MIN_TEXT_CHARS:
                return None
            return text.strip()
        finally:
            await page.close()


async def fetch_text(url: str) -> str | None:
    """Page text for *url* via camofox (solving Turnstile if needed), or None. Never raises."""
    if not enabled():
        return None
    try:
        async with _LOCK:
            return await asyncio.wait_for(_fetch(url, _key(), _base()), timeout=OVERALL_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - tier D is best-effort
        logger.info(f"webex.tier_d.failed | url={url} error={type(exc).__name__}: {exc}")
        return None
