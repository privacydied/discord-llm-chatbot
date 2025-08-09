from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import httpx

from .util.logging import get_logger
from .exceptions import APIError
from .retry_utils import with_retry, API_RETRY_CONFIG

logger = get_logger(__name__)

# Constants [CMV]
_X_API_BASE_URL = "https://api.twitter.com/2"  # Official v2 endpoint remains twitter.com domain
_TWEET_ID_RE = re.compile(r"^\d{8,20}$")
_X_URL_ID_RE = re.compile(
    r"https?://(?:www\.)?(?:twitter|x|vxtwitter|fxtwitter)\.com/[^/]+/status/(\d{8,20})(?:\D.*)?",
    re.IGNORECASE,
)


class XApiClient:
    """
    Async client for Twitter/X v2 API: Get Tweet by ID with hydrated expansions.

    Security: never logs bearer token, only presence via has_token. [SFT]
    """

    def __init__(
        self,
        bearer_token: Optional[str],
        timeout_ms: int = 8000,
        default_tweet_fields: Optional[List[str]] = None,
        default_expansions: Optional[List[str]] = None,
        default_media_fields: Optional[List[str]] = None,
        default_user_fields: Optional[List[str]] = None,
        default_poll_fields: Optional[List[str]] = None,
        default_place_fields: Optional[List[str]] = None,
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
                }
            },
        )

    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception as e:
            logger.debug(f"XApiClient close error: {e}")

    async def __aenter__(self) -> "XApiClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    @staticmethod
    def extract_tweet_id(value: str) -> Optional[str]:
        """Extract tweet ID from raw ID or URL. Returns None if not found. [IV]"""
        if not value or not isinstance(value, str):
            return None
        value = value.strip()
        if _TWEET_ID_RE.match(value):
            return value
        m = _X_URL_ID_RE.match(value)
        if m:
            return m.group(1)
        return None

    def _build_headers(self, bearer_token: Optional[str]) -> Dict[str, str]:
        headers = {"User-Agent": "discord-bot-x-integration/1.0"}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        return headers

    def _build_params(
        self,
        tweet_fields: Optional[List[str]] = None,
        expansions: Optional[List[str]] = None,
        media_fields: Optional[List[str]] = None,
        user_fields: Optional[List[str]] = None,
        poll_fields: Optional[List[str]] = None,
        place_fields: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        def _csv(v: Optional[List[str]]) -> Optional[str]:
            if not v:
                return None
            return ",".join(sorted({s.strip() for s in v if s and s.strip()}))

        params: Dict[str, str] = {}
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
        tweet_fields: Optional[List[str]] = None,
        expansions: Optional[List[str]] = None,
        media_fields: Optional[List[str]] = None,
        user_fields: Optional[List[str]] = None,
        poll_fields: Optional[List[str]] = None,
        place_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fetch a tweet by ID with field hydration.
        Raises APIError with status-specific messages. [REH]
        """
        params = self._build_params(
            tweet_fields, expansions, media_fields, user_fields, poll_fields, place_fields
        )
        url = f"{self._base_url}/tweets/{tweet_id}"

        try:
            resp = await self._client.get(url, params=params)
            # Raise for status to unify handling
            if resp.status_code >= 400:
                await self._raise_for_status(resp)
            data = resp.json()
            return data
        except httpx.HTTPStatusError as he:
            # Note: with_retry checks status text too; re-raise as APIError for consistency
            status = he.response.status_code if he.response is not None else None
            raise APIError(f"HTTP error from X API: {status}") from he
        except (httpx.TimeoutException, httpx.TransportError) as te:
            raise APIError(f"Transport error from X API: {te}") from te
        except Exception as e:
            raise APIError(f"Unexpected X API error: {e}") from e

    async def _raise_for_status(self, resp: httpx.Response) -> None:
        status = resp.status_code
        detail = None
        try:
            detail = resp.json()
        except Exception:
            detail = {"text": resp.text[:2000]}
        extra = {"detail": {"status": status, "has_token": self._has_token, "body": detail}}

        # Strict mapping per spec [REH][SFT]
        if status in (401, 403):
            logger.info("X API access denied", extra=extra)
            raise APIError(f"X API access denied ({status})")
        if status in (404, 410):
            logger.info("X API post not found or deleted", extra=extra)
            raise APIError(f"X API post not found or deleted ({status})")
        if status == 429:
            retry_after = resp.headers.get("retry-after")
            logger.warning("X API rate limited", extra={"detail": {**extra["detail"], "retry_after": retry_after}})
            # Allow retries via decorator
            raise APIError("429 Too Many Requests")
        if 500 <= status <= 599:
            logger.warning("X API server error", extra=extra)
            raise APIError(f"X API server error ({status})")

        # Fallback generic
        logger.error("X API unexpected status", extra=extra)
        raise APIError(f"X API unexpected status: {status}")


# Convenience helpers [CA]

def parse_csv_env(value: Optional[str], default: List[str]) -> List[str]:
    if value is None or not value.strip():
        return default
    return [s.strip() for s in value.split(',') if s.strip()]
