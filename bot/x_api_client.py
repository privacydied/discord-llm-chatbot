from __future__ import annotations

import re
from typing import Any, Self
from urllib.parse import parse_qs, urlparse

import httpx

from .exceptions import APIError
from .retry_utils import API_RETRY_CONFIG, with_retry
from .utils.logging import get_logger

logger = get_logger(__name__)

# Constants [CMV]
_X_API_BASE_URL = "https://api.twitter.com/2"  # Official v2 endpoint remains twitter.com domain
_TWEET_ID_RE = re.compile(r"^\d{8,20}$")
_X_URL_PATH_ID_RE = re.compile(
    r"/(?:i/(?:web/)?status|i/status|[^/]+/status)/(\d{8,20})(?:\D|$)",
    re.IGNORECASE,
)
_X_URL_QUERY_ID_KEYS = ("id", "tweet_id", "status", "status_id")


class XApiClient:
    """Async client for Twitter/X v2 API: Get Tweet by ID with hydrated expansions.

    Security: never logs bearer token, only presence via has_token. [SFT]
    """

    def __init__(
        self,
        bearer_token: str | None,
        timeout_ms: int = 8000,
        default_tweet_fields: list[str] | None = None,
        default_expansions: list[str] | None = None,
        default_media_fields: list[str] | None = None,
        default_user_fields: list[str] | None = None,
        default_poll_fields: list[str] | None = None,
        default_place_fields: list[str] | None = None,
        base_url: str = _X_API_BASE_URL,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_ms / 1000.0)
        self._client = httpx.AsyncClient(timeout=self._timeout, headers=self._build_headers(bearer_token))
        self._has_token = bool(bearer_token)

        # Defaults for field hydration [CMV]
        self._tweet_fields = default_tweet_fields or []
        self._expansions = default_expansions or []
        self._media_fields = default_media_fields or []
        self._user_fields = default_user_fields or []
        self._poll_fields = default_poll_fields or []
        self._place_fields = default_place_fields or []

        logger.debug(
            "Initialized XApiClient",
            extra={
                "detail": {
                    "base_url": self._base_url,
                    "timeout_ms": timeout_ms,
                    "has_token": self._has_token,
                    "defaults": {
                        "tweet_fields": self._tweet_fields,
                        "expansions": self._expansions,
                        "media_fields": self._media_fields,
                        "user_fields": self._user_fields,
                        "poll_fields": self._poll_fields,
                        "place_fields": self._place_fields,
                    },
                },
            },
        )

    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception as e:
            logger.debug(f"XApiClient close error: {e}")

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    @staticmethod
    def extract_tweet_id(value: str) -> str | None:
        """Extract tweet ID from raw ID or URL. Returns None if not found. [IV]."""
        if not value or not isinstance(value, str):
            return None
        value = value.strip()
        if _TWEET_ID_RE.match(value):
            return value
        # Robust URL parsing: handle multiple hosts + URL shapes (status path, i/web/status, query params). [IV]
        try:
            parsed = urlparse(value)
            # 1) Query params (syndication/embed style URLs)
            qs = parse_qs(parsed.query or "")
            for k in _X_URL_QUERY_ID_KEYS:
                vals = qs.get(k)
                if vals:
                    cand = (vals[0] or "").strip()
                    if _TWEET_ID_RE.match(cand):
                        return cand

            # 2) Path-based extraction
            m = _X_URL_PATH_ID_RE.search(parsed.path or "")
            if m:
                return m.group(1)
        except Exception as exc:
            # Fall through to raw regex search
            logger.debug(f"tweet ID URL parse failed: {exc}")

        try:
            m2 = _X_URL_PATH_ID_RE.search(value)
            if m2:
                return m2.group(1)
        except Exception as exc:
            logger.debug(f"tweet ID raw regex failed: {exc}")
        return None

    def _build_headers(self, bearer_token: str | None) -> dict[str, str]:
        headers = {"User-Agent": "discord-bot-x-integration/1.0"}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        return headers

    def _build_params(
        self,
        tweet_fields: list[str] | None = None,
        expansions: list[str] | None = None,
        media_fields: list[str] | None = None,
        user_fields: list[str] | None = None,
        poll_fields: list[str] | None = None,
        place_fields: list[str] | None = None,
    ) -> dict[str, str]:
        def _csv(v: list[str] | None) -> str | None:
            if not v:
                return None
            return ",".join(sorted({s.strip() for s in v if s and s.strip()}))

        params: dict[str, str] = {}
        tf = _csv(tweet_fields or self._tweet_fields)
        ex = _csv(expansions or self._expansions)
        mf = _csv(media_fields or self._media_fields)
        uf = _csv(user_fields or self._user_fields)
        pf = _csv(poll_fields or self._poll_fields)
        plf = _csv(place_fields or self._place_fields)
        if tf:
            params["tweet.fields"] = tf
        if ex:
            params["expansions"] = ex
        if mf:
            params["media.fields"] = mf
        if uf:
            params["user.fields"] = uf
        if pf:
            params["poll.fields"] = pf
        if plf:
            params["place.fields"] = plf
        return params

    @with_retry(API_RETRY_CONFIG)
    async def get_tweet_by_id(
        self,
        tweet_id: str,
        *,
        tweet_fields: list[str] | None = None,
        expansions: list[str] | None = None,
        media_fields: list[str] | None = None,
        user_fields: list[str] | None = None,
        poll_fields: list[str] | None = None,
        place_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch a tweet by ID with field hydration.
        Raises APIError with status-specific messages. [REH].
        """
        params = self._build_params(
            tweet_fields,
            expansions,
            media_fields,
            user_fields,
            poll_fields,
            place_fields,
        )
        url = f"{self._base_url}/tweets/{tweet_id}"

        try:
            resp = await self._client.get(url, params=params)
            # Raise for status to unify handling
            if resp.status_code >= 400:
                await self._raise_for_status(resp)
            return resp.json()
        except httpx.HTTPStatusError as he:
            # Note: with_retry checks status text too; re-raise as APIError for consistency
            status = he.response.status_code if he.response is not None else None
            msg = f"HTTP error from X API: {status}"
            raise APIError(msg) from he
        except (httpx.TimeoutException, httpx.TransportError) as te:
            msg = f"Transport error from X API: {te}"
            raise APIError(msg) from te
        except Exception as e:
            msg = f"Unexpected X API error: {e}"
            raise APIError(msg) from e

    async def _raise_for_status(self, resp: httpx.Response) -> None:
        status = resp.status_code
        detail = None
        try:
            detail = resp.json()
        except Exception as exc:
            logger.debug(f"Response JSON parse failed: {exc}")
            detail = {"text": resp.text[:2000]}
        extra = {"detail": {"status": status, "has_token": self._has_token, "body": detail}}

        # Strict mapping per spec [REH][SFT]
        if status in (401, 403):
            logger.info("X API access denied", extra=extra)
            msg = f"X API access denied ({status})"
            raise APIError(msg)
        if status in (404, 410):
            logger.info("X API post not found or deleted", extra=extra)
            msg = f"X API post not found or deleted ({status})"
            raise APIError(msg)
        if status == 429:
            retry_after = resp.headers.get("retry-after")
            # Best-effort parse Retry-After seconds (Twitter typically returns seconds)
            retry_after_secs = None
            if retry_after:
                try:
                    retry_after_secs = float(retry_after)
                except Exception as exc:
                    logger.debug(f"Retry-After parse failed: {exc}")
                    retry_after_secs = None
            logger.warning(
                "X API rate limited",
                extra={"detail": {**extra["detail"], "retry_after": retry_after}},
            )
            # Attach advisory delay for the retry logic to respect [REH]
            err = APIError("429 Too Many Requests")
            try:
                if retry_after_secs and retry_after_secs > 0:
                    err.retry_after_seconds = float(retry_after_secs)
            except Exception as exc:
                logger.debug(f"retry_after parse failed: {exc}")
            # Allow retries via decorator
            raise err
        if 500 <= status <= 599:
            logger.warning("X API server error", extra=extra)
            msg = f"X API server error ({status})"
            raise APIError(msg)

        # Fallback generic
        logger.error("X API unexpected status", extra=extra)
        msg = f"X API unexpected status: {status}"
        raise APIError(msg)


# Convenience helpers [CA]


def parse_csv_env(value: str | None, default: list[str]) -> list[str]:
    if value is None or not value.strip():
        return default
    return [s.strip() for s in value.split(",") if s.strip()]
