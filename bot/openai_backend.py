"""OpenAI/OpenRouter Backend - Handles OpenAI API calls including OpenRouter."""

import asyncio
import atexit
import base64
import inspect
import os
import ssl
import time
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import aiohttp
import certifi
import httpx

try:
    import openai as _openai
except ModuleNotFoundError:
    _openai = None
    openai = SimpleNamespace(
        AsyncOpenAI=None,
        AuthenticationError=Exception,
        RateLimitError=Exception,
        APIError=Exception,
        _base_client=SimpleNamespace(AsyncHttpxClientWrapper=None),
    )
else:
    openai = _openai

import contextlib

from bot.config import get_vl_model_ladder, load_config
from bot.enhanced_retry import get_retry_manager
from bot.exceptions import APIError
from bot.memory import get_profile, get_server_profile
from bot.retry_utils import (
    API_RETRY_CONFIG,
    with_retry,
)
from bot.utils.logging import get_logger

logger = get_logger(__name__)

# --- Shared TLS context + client reuse (Fix 5) [PA][RM] ---
# Building the SSL context parses the full certifi CA bundle from disk; do it
# ONCE at import time and reuse the same context for every httpx client.
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# Cache of AsyncOpenAI clients keyed by (base_url, api_key, timeout, max_retries).
# Reusing clients keeps httpx keep-alive / HTTP-2 connection pools warm across
# requests and across the fallback ladder instead of a fresh TLS handshake each call.
_OPENAI_CLIENT_CACHE: dict[tuple[Any, ...], Any] = {}

# Cache of local OllamaClient instances keyed by base_url so the aiohttp session
# (and its keep-alive pool) is reused across ladder attempts. [PA][RM]
_OLLAMA_CLIENT_CACHE: dict[str, Any] = {}

_HTTP2_SUPPORTED: bool | None = None

# --- Prompt-file cache (Fix 7): re-read only when mtime changes. [PA] ---
_PROMPT_CACHE: dict[str, tuple[float, str]] = {}

# --- Shared image-download session (low-value cleanup) [PA][RM] ---
_IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 10
_image_session: aiohttp.ClientSession | None = None

if _openai is not None:
    OpenAIAuthenticationError = _openai.AuthenticationError
    OpenAIRateLimitError = _openai.RateLimitError
    OpenAIAPIError = _openai.APIError
else:
    OpenAIAuthenticationError = OpenAIRateLimitError = OpenAIAPIError = Exception


def _embedded_provider_error_detail(response_obj: Any) -> str:
    """Extract the provider error OpenRouter embeds in 200-responses. [REH]

    Free models return HTTP 200 with an `error` object and no usable choices
    when rate-limited/saturated upstream. Returns a ' (provider error: ...)'
    suffix for exception messages (so logs show the real cause and embedded
    status codes like 429/502 feed the retry classifier), or '' if absent.
    """
    embedded = getattr(response_obj, "error", None)
    if embedded is None:
        extra = getattr(response_obj, "model_extra", None)
        if isinstance(extra, dict):
            embedded = extra.get("error")
    return f" (provider error: {str(embedded)[:300]})" if embedded else ""


def _require_openai() -> Any:
    if _openai is None:
        msg = "OpenAI backend is unavailable because the 'openai' package is not installed"
        raise APIError(msg)
    return _openai


def _patch_openai_httpx_wrapper_destructor() -> None:
    """Prevent OpenAI's wrapper __del__ from scheduling a broken close task.

    On Synology the default SSL cafile can be missing. If AsyncOpenAI creation
    fails during httpx transport init, OpenAI may leave a partially initialized
    AsyncHttpxClientWrapper whose __del__ schedules aclose(); httpx then raises
    AttributeError('_transport') in a background task. [REH]
    """
    try:
        openai = _require_openai()
        wrapper_cls = openai._base_client.AsyncHttpxClientWrapper
    except Exception:
        return

    if getattr(wrapper_cls, "_hermes_safe_del", False):
        return

    def _safe_del(self) -> None:
        try:
            if not hasattr(self, "_transport"):
                return
            if self.is_closed:
                return
            import asyncio

            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:
            return

    wrapper_cls.__del__ = _safe_del
    wrapper_cls._hermes_safe_del = True


_patch_openai_httpx_wrapper_destructor()


def _http2_supported() -> bool:
    """Return True when the optional 'h2' dependency is importable. [REH]"""
    global _HTTP2_SUPPORTED
    if _HTTP2_SUPPORTED is None:
        try:
            import h2  # type: ignore  # noqa: F401

            _HTTP2_SUPPORTED = True
        except ImportError:
            _HTTP2_SUPPORTED = False
    return _HTTP2_SUPPORTED


def _timeout_cache_key(timeout: httpx.Timeout) -> tuple[Any, ...]:
    """Hashable key for an httpx.Timeout so clients can be cached by timeout."""
    return (
        getattr(timeout, "connect", None),
        getattr(timeout, "read", None),
        getattr(timeout, "write", None),
        getattr(timeout, "pool", None),
    )


def _schedule_http_client_close(http_client: Any) -> None:
    """Best-effort async close of a throwaway httpx client. [RM]"""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(http_client.aclose())
    except Exception as e:
        logger.debug(f"Failed to schedule http_client close: {e}")


def _build_openai_async_client(
    *,
    api_key: str | None,
    base_url: str | None,
    timeout: httpx.Timeout,
    max_retries: int,
) -> Any:
    """Construct a fresh AsyncOpenAI backed by a keep-alive httpx client.

    Passing our own httpx.AsyncClient with the shared certifi-backed SSL context
    avoids httpx consulting broken platform cafile paths such as
    /etc/ssl/cert.pem on Synology. [REH]
    """
    http_client = httpx.AsyncClient(
        timeout=timeout,
        verify=_SSL_CONTEXT,
        trust_env=False,
        http2=_http2_supported(),
    )
    try:
        openai = _require_openai()
        return openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
        )
    except Exception:
        _schedule_http_client_close(http_client)
        raise


def _make_openai_async_client(
    *,
    api_key: str | None,
    base_url: str | None,
    timeout: httpx.Timeout,
    max_retries: int = 0,
) -> Any:
    """Return a cached AsyncOpenAI client, building one on first use. [PA]

    Reusing the client keeps the httpx connection pool (keep-alive / HTTP-2)
    warm instead of a new TLS handshake per request. Cached clients must NOT be
    closed per request; see _safe_aclose_openai_client / aclose_all_clients.

    The AsyncOpenAI factory identity is part of the key: in production it is the
    stable class (so clients are reused), while a patched factory (e.g. tests)
    yields a distinct key and thus a freshly built client.
    """
    factory = getattr(_openai, "AsyncOpenAI", None) if _openai is not None else None
    cache_key = (id(factory), base_url or "", api_key or "", _timeout_cache_key(timeout), max_retries)
    client = _OPENAI_CLIENT_CACHE.get(cache_key)
    if client is None:
        client = _build_openai_async_client(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        _OPENAI_CLIENT_CACHE[cache_key] = client
    return client


def _is_cached_openai_client(client: Any) -> bool:
    """True when `client` is a shared cached client that must stay open."""
    return any(client is cached for cached in _OPENAI_CLIENT_CACHE.values())


def _load_prompt_cached(path: str) -> str:
    """Return prompt file contents, re-reading only when its mtime changes. [PA]

    Raises FileNotFoundError/OSError to callers exactly like open() so existing
    error handling continues to work.
    """
    mtime = os.path.getmtime(path)
    cached = _PROMPT_CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    with open(path, encoding="utf-8") as pf:
        contents = pf.read().strip()
    _PROMPT_CACHE[path] = (mtime, contents)
    return contents


def _get_image_session() -> aiohttp.ClientSession:
    """Return a shared aiohttp session for image downloads, bound to this loop.

    Recreates the session if it was never created, was closed, or belongs to a
    different (e.g. restarted) event loop. Keeps the 10s total timeout. [RM]
    """
    global _image_session
    loop = asyncio.get_running_loop()
    session = _image_session
    if session is not None and not session.closed and getattr(session, "_loop", None) is loop:
        return session
    timeout = aiohttp.ClientTimeout(total=_IMAGE_DOWNLOAD_TIMEOUT_SECONDS)
    _image_session = aiohttp.ClientSession(timeout=timeout)
    return _image_session


async def _aclose_one(obj: Any) -> None:
    """Await-or-call close on a client/session, swallowing errors. [RM]"""
    closer = getattr(obj, "aclose", None) or getattr(obj, "close", None)
    if closer is None:
        return
    try:
        result = closer()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        logger.debug(f"aclose failed | {type(exc).__name__}: {exc}")


async def aclose_all_clients() -> None:
    """Close every cached OpenAI/Ollama client + image session and clear caches.

    Call on shutdown to release pooled connections cleanly. [RM]
    """
    openai_clients = list(_OPENAI_CLIENT_CACHE.values())
    ollama_clients = list(_OLLAMA_CLIENT_CACHE.values())
    _OPENAI_CLIENT_CACHE.clear()
    _OLLAMA_CLIENT_CACHE.clear()
    for client in (*openai_clients, *ollama_clients):
        await _aclose_one(client)
    global _image_session
    if _image_session is not None:
        await _aclose_one(_image_session)
        _image_session = None


def _atexit_close_clients() -> None:
    """Best-effort synchronous cleanup of cached clients at interpreter exit."""
    if not (_OPENAI_CLIENT_CACHE or _OLLAMA_CLIENT_CACHE or _image_session):
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        with contextlib.suppress(Exception):
            asyncio.run(aclose_all_clients())


atexit.register(_atexit_close_clients)


def _resolve_openai_compatible_endpoint(provider_name: str | None, config: dict[str, Any]) -> tuple[str | None, str | None, str]:
    """Resolve API key/base URL for OpenAI-compatible ladders."""
    provider = str(provider_name or "openrouter").strip().lower()
    if provider == "nvidia":
        return (
            config.get("NVIDIA_NIM_API_KEY") or config.get("OPENAI_API_KEY"),
            config.get("NVIDIA_NIM_API_BASE") or config.get("OPENAI_API_BASE") or "https://integrate.api.nvidia.com/v1",
            "nvidia",
        )
    if provider == "openrouter":
        return (
            config.get("OPENROUTER_API_KEY") or config.get("OPENAI_API_KEY") or config.get("VISION_API_KEY"),
            "https://openrouter.ai/api/v1",
            "openrouter",
        )
    return (
        config.get("OPENAI_API_KEY"),
        config.get("OPENAI_API_BASE") or "https://api.openai.com/v1",
        provider,
    )


def _ollama_coro_factory(
    config: dict[str, Any],
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int | None,
    kwargs: dict,
    provider_config,
    presence_penalty: float | None = None,
):
    """Create a coroutine that calls the local Ollama API.

    Converts the OpenAI-style messages to a single prompt for Ollama,
    then calls _make_openai_compatible_request to normalize the response
    to the same shape expected by downstream code.
    """

    async def _run():
        base_url = config.get("OLLAMA_HOST") or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        ollama_client = _get_cached_ollama_client(base_url)

        # Convert OpenAI messages to a single prompt for Ollama
        user_parts = []
        system_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                user_parts.append(content)
            elif role == "assistant":
                user_parts.append(f"[Assistant previous response]: {content}")

        prompt = "\n".join(user_parts) if user_parts else ""
        if system_parts:
            system_text = "\n".join(system_parts)
            prompt = f"System instructions: {system_text}\n\n{prompt}"

        # Do NOT close the cached client here — the aiohttp session is shared
        # across ladder attempts for keep-alive pooling; closed on shutdown. [RM]
        result = await ollama_client.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty if presence_penalty is not None else config.get("PRESENCE_PENALTY", 0.0),
            **{k: v for k, v in kwargs.items() if k != "stream"},
        )
        text = result.get("response", "")
        if not text:
            msg_0 = f"Ollama returned empty response for model {model}"
            raise APIError(msg_0)
        return text

    return _run


def _get_cached_ollama_client(base_url: str) -> Any:
    """Return a cached OllamaClient for `base_url`, creating it on first use. [PA]"""
    client = _OLLAMA_CLIENT_CACHE.get(base_url)
    if client is None:
        from .ollama import OllamaClient

        client = OllamaClient(base_url=base_url)
        _OLLAMA_CLIENT_CACHE[base_url] = client
    return client


async def _safe_aclose_openai_client(client: Any) -> None:
    """Best-effort close for OpenAI/httpx async clients without leaking __del__ tasks.

    OpenAI's AsyncHttpxClientWrapper schedules `aclose()` from `__del__` when a
    client is GC'd open. On some partially-initialized wrapper instances that
    close path raises AttributeError("... has no attribute '_transport'"), which
    appears as an unretrieved task exception. Explicitly closing clients on our
    awaited path prevents the destructor path; this helper also suppresses that
    known close-only AttributeError if the wrapper is already malformed. [REH]
    """
    if client is None:
        return
    # Never tear down a shared/cached client — closing it would destroy the
    # keep-alive connection pool that subsequent requests reuse. [PA][RM]
    if _is_cached_openai_client(client):
        return
    closer = getattr(client, "aclose", None) or getattr(client, "close", None)
    if closer is None:
        return
    try:
        result = closer()
        if inspect.isawaitable(result):
            await result
    except AttributeError as exc:
        if "_transport" in str(exc) and "AsyncHttpxClientWrapper" in str(exc):
            logger.debug("openai.client.close_ignored missing_transport")
            return
        raise
    except Exception as exc:
        logger.debug(f"openai.client.close_failed | {type(exc).__name__}: {exc}")


@with_retry(API_RETRY_CONFIG)
async def generate_openai_response(
    prompt: str,
    context: str = "",
    system_prompt: str | None = None,
    user_id: str | None = None,
    guild_id: str | None = None,
    temperature: float | None = None,
    presence_penalty: float | None = None,
    max_tokens: int | None = None,
    stream: bool = False,
    **kwargs,
) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
    """Generate a response using OpenAI API (including OpenRouter).

    Args:
        prompt: The user's input prompt
        context: Optional context to include in the prompt
        user_id: Optional user ID for personalization
        guild_id: Optional guild ID for server-specific context
        temperature: Controls randomness (0.0 to 1.0)
        presence_penalty: Penalizes tokens that have already appeared, encouraging
            new topics (config default: PRESENCE_PENALTY, 0.0)
        max_tokens: Maximum number of tokens to generate
        stream: Whether to stream the response
        **kwargs: Additional parameters to pass to the API

    Returns:
        Dictionary with the generated text and metadata

    """
    try:
        config = load_config()

        # Configure OpenAI-compatible client with configurable timeout
        import httpx

        backend = str(config.get("TEXT_BACKEND", "openai") or "openai").lower()
        if backend == "nvidia":
            api_key = config.get("NVIDIA_NIM_API_KEY") or config.get("OPENAI_API_KEY")
            api_base = config.get("NVIDIA_NIM_API_BASE") or config.get("OPENAI_API_BASE") or "https://integrate.api.nvidia.com/v1"
            configured_model = config.get("NVIDIA_NIM_TEXT_MODEL") or config.get("OPENAI_TEXT_MODEL") or "meta/llama3-70b-instruct"
        else:
            api_key = config.get("OPENAI_API_KEY")
            api_base = config.get("OPENAI_API_BASE", "https://api.openai.com/v1")
            configured_model = config.get("OPENAI_TEXT_MODEL", "gpt-4")

        # Prefer TEXTGEN_TIMEOUT_SECONDS; fall back to TEXT_REQUEST_TIMEOUT; default 45s
        try:
            timeout_seconds = float(config.get("TEXTGEN_TIMEOUT_SECONDS", config.get("TEXT_REQUEST_TIMEOUT", "45")))
        except Exception:
            timeout_seconds = 45.0

        client = _make_openai_async_client(
            api_key=api_key,
            base_url=api_base,
            timeout=httpx.Timeout(timeout_seconds),
            max_retries=0,  # Let our retry system handle retries
        )

        # Get model configuration
        model = configured_model

        # Get user preferences if user_id is provided
        if user_id:
            profile = get_profile(str(user_id))
            user_prefs = profile.get("preferences", {}) if profile else {}

            # Apply user preferences if not overridden
            if temperature is None:
                temperature = user_prefs.get("temperature", config.get("TEMPERATURE", 0.7))
            if presence_penalty is None:
                presence_penalty = user_prefs.get("presence_penalty", config.get("PRESENCE_PENALTY", 0.0))
        else:
            temperature = temperature or config.get("TEMPERATURE", 0.7)
            presence_penalty = config.get("PRESENCE_PENALTY", 0.0) if presence_penalty is None else presence_penalty

        # Get server context if guild_id is provided
        server_context = ""
        if guild_id:
            server_profile = get_server_profile(str(guild_id))
            if server_profile:
                server_context = server_profile.get("context_notes", "")

        # Prepare the messages payload
        messages = []

        # If a specific system_prompt is provided, use it.
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # Otherwise, construct the default system prompt.
        else:
            prompt_file_path = config.get("PROMPT_FILE")
            if not prompt_file_path:
                msg = "PROMPT_FILE not configured in environment variables"
                raise APIError(msg)
            try:
                base_system_prompt = _load_prompt_cached(prompt_file_path)
            except FileNotFoundError:
                msg = f"Prompt file not found: {prompt_file_path}"
                raise APIError(msg)
            except Exception as e:
                msg = f"Error reading prompt file {prompt_file_path}: {e}"
                raise APIError(msg)

            # Combine base prompt, server context, and conversation history
            full_system_prompt = f"""{base_system_prompt}

Server Context: {server_context}"""
            messages.append({"role": "system", "content": full_system_prompt})

        # Add conversation history if it exists and a specific system prompt was used
        if system_prompt and context:
            messages.append(
                {
                    "role": "system",
                    "content": f"PREVIOUS_CONVERSATION_HISTORY:\n{context}",
                },
            )

        # Finally, add the user's actual prompt
        messages.append({"role": "user", "content": prompt})

        # Set default max_tokens if not provided
        if max_tokens is None:
            max_tokens = config.get("MAX_RESPONSE_TOKENS", 1000)

        logger.info(f"Configured OpenAI-compatible text model: {model}")
        logger.debug(f"[OpenAI] Request params: temp={temperature}, presence_penalty={presence_penalty}, max_tokens={max_tokens}, stream={stream}")

        # Helper to normalize a completion response into our result dict
        def _normalize_nonstream_response(response_obj, used_model: str) -> dict[str, Any]:
            try:
                if not getattr(response_obj, "choices", None):
                    # OpenRouter (esp. free models) returns HTTP 200 with an
                    # embedded error object and no choices when rate-limited
                    # upstream — surface it so logs show the real cause and the
                    # retry classifier can see status codes like 429. [REH]
                    msg_0 = f"No choices returned in OpenAI response{_embedded_provider_error_detail(response_obj)}"
                    raise APIError(msg_0)
                first = response_obj.choices[0]
                content = getattr(getattr(first, "message", None), "content", "") or ""
                try:
                    from .vl.postprocess import sanitize_model_output

                    usable_content = sanitize_model_output(content)
                except Exception:
                    usable_content = content
                if not str(usable_content or "").strip():
                    msg_0 = f"Empty text response from model {used_model}"
                    raise APIError(msg_0)
                usage = getattr(response_obj, "usage", None)
                usage_info_local = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }
                return {
                    "text": content,
                    "model": used_model,
                    "usage": usage_info_local,
                    "backend": "openai",
                }
            except Exception as norm_error:
                msg_0 = f"Response normalization failed: {type(norm_error).__name__}: {norm_error}"
                raise APIError(msg_0)

        if stream:
            logger.debug("[OpenAI] 🔄 Sending request to API (streaming)...")
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )
            logger.debug("[OpenAI] ✅ Received response from API (streaming)")
            logger.debug("[OpenAI] 🌊 Processing streaming response...")

            async def stream_generator():
                chunk_count = 0
                try:
                    async for chunk in response:
                        chunk_count += 1
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield {
                                "text": chunk.choices[0].delta.content,
                                "finished": False,
                            }
                    logger.debug(f"[OpenAI] ✅ Streaming complete, processed {chunk_count} chunks")
                    yield {"text": "", "finished": True}
                finally:
                    await _safe_aclose_openai_client(client)

            return stream_generator()

        # Non-streaming: if using OpenRouter or NVIDIA Build/NIM base, engage provider/model fallback ladder
        base_url = str(api_base or "https://api.openai.com/v1")
        base_url_l = base_url.lower()
        use_text_fallback = "openrouter" in base_url_l or "nvidia.com" in base_url_l

        if use_text_fallback:
            ladder_label = "NVIDIA" if "nvidia.com" in base_url_l else "OpenRouter"
            logger.info(f"[OpenAI] Using EnhancedRetryManager text fallback ladder ({ladder_label})")
            await _safe_aclose_openai_client(client)
            retry_mgr = get_retry_manager()

            def _coro_factory(provider_config):
                selected_model = provider_config.model

                # Ollama provider: route through local Ollama instance
                if provider_config.name == "ollama":
                    return _ollama_coro_factory(
                        config,
                        selected_model,
                        messages,
                        temperature,
                        max_tokens,
                        kwargs,
                        provider_config,
                        presence_penalty=presence_penalty,
                    )

                attempt_api_key, attempt_base_url, attempt_provider = _resolve_openai_compatible_endpoint(provider_config.name, config)

                async def _run():
                    attempt_client = _make_openai_async_client(
                        api_key=attempt_api_key,
                        base_url=attempt_base_url,
                        timeout=httpx.Timeout(provider_config.timeout),
                        max_retries=0,
                    )
                    try:
                        logger.debug(f"[OpenAI] 🔄 Sending request to {attempt_provider} with model: {selected_model}")
                        t0 = time.monotonic()
                        resp = await attempt_client.chat.completions.create(
                            model=selected_model,
                            messages=messages,
                            temperature=temperature,
                            presence_penalty=presence_penalty,
                            max_tokens=max_tokens,
                            stream=False,
                            **kwargs,
                        )
                        logger.debug("[OpenAI] ✅ Received response from API")
                        return _normalize_nonstream_response(resp, selected_model)
                    except httpx.HTTPStatusError as he:
                        status_code = he.response.status_code if getattr(he, "response", None) is not None else "unknown"
                        retry_after = None
                        try:
                            # Prefer standard Retry-After; fall back to common rate-limit reset hints
                            if getattr(he, "response", None) is not None:
                                hdrs = he.response.headers
                                ra = hdrs.get("retry-after") or hdrs.get("Retry-After")
                                if ra is not None:
                                    retry_after = float(ra)
                                else:
                                    # OpenRouter-style reset headers (may be a unix epoch or delta seconds)
                                    reset = hdrs.get("x-ratelimit-reset") or hdrs.get("X-RateLimit-Reset") or hdrs.get("rate-limit-reset")
                                    if reset is not None:
                                        try:
                                            val = float(reset)
                                            now = time.time()
                                            retry_after = val - now if val > now + 1 else val
                                            if retry_after < 0:
                                                retry_after = 0.0
                                        except Exception:
                                            retry_after = None
                        except Exception:
                            retry_after = None
                        extra = f" (retry-after={retry_after}s)" if retry_after is not None else ""
                        logger.warning(f"OpenAI HTTP error during fallback attempt: {status_code} {he}{extra}")
                        err = APIError(f"HTTP {status_code}: {he!s}{extra}")
                        # Propagate Retry-After to outer retry harness
                        try:
                            if retry_after is not None:
                                err.retry_after_seconds = retry_after
                        except Exception as e:
                            logger.debug(f"Failed to set retry_after_seconds: {e}")
                        raise err
                    except Exception as e:
                        # Normalize timeouts and log a compact line per attempt [REH]
                        elapsed_ms = int((time.monotonic() - t0) * 1000)
                        etype = type(e).__name__
                        if "timeout" in str(e).lower() or "timeout" in etype.lower():
                            logger.warning(f"TextGen timeout (model {selected_model}): {elapsed_ms}ms")
                            msg_0 = f"Request timeout after ~{elapsed_ms}ms for model {selected_model}"
                            raise APIError(msg_0)
                        raise
                    finally:
                        await _safe_aclose_openai_client(attempt_client)

                return _run

            per_item_budget = float(config.get("TEXT_PER_ITEM_BUDGET", 120.0))
            # Clamp to the ambient request deadline (multimodal total budget):
            # earlier stages (video/STT/VL) may have eaten most of it, and a
            # full fixed ladder budget here would overrun the outer wait_for and
            # surface as a user-facing total_timeout. Keep a dispatch reserve so
            # the reply can still be delivered, and a small floor so at least
            # one fast attempt runs. [PA][REH]
            from bot.time_budget import remaining_seconds

            _DEADLINE_DISPATCH_RESERVE_S = 10.0  # [CMV]
            _LADDER_MIN_BUDGET_S = 8.0  # [CMV]
            ambient_remaining = remaining_seconds()
            if ambient_remaining is not None:
                clamped = max(_LADDER_MIN_BUDGET_S, ambient_remaining - _DEADLINE_DISPATCH_RESERVE_S)
                if clamped < per_item_budget:
                    logger.info(f"[OpenAI] text.budget clamped to deadline: {per_item_budget:.0f}s -> {clamped:.0f}s (ambient_remaining={ambient_remaining:.0f}s)")
                    per_item_budget = clamped
            with contextlib.suppress(Exception):
                logger.info(f"[OpenAI] text.budget seconds={per_item_budget}")
            try:
                rr = await retry_mgr.run_with_fallback("text", _coro_factory, per_item_budget=per_item_budget)
            finally:
                await _safe_aclose_openai_client(client)
            if not rr.success:
                # Enrich error with ladder context for better observability [REH]
                if rr.error:
                    base_err = rr.error
                    try:
                        prov = getattr(rr, "provider_used", None) or getattr(base_err, "provider_key", None) or "unknown"
                        attempts = getattr(rr, "attempts", 0)
                        total_time = getattr(rr, "total_time", 0.0)
                        err_str = str(base_err).lower()
                        _is_moderation = (
                            "flagged" in err_str
                            or "moderation" in err_str
                            or "self-harm" in err_str
                            or "self_harm" in err_str
                            or "content_filter" in err_str
                            or "content filter" in err_str
                            or "requires moderation" in err_str
                        ) and ("403" in err_str or "400" in err_str)
                        _is_no_endpoints = "no endpoints found" in err_str or ("404" in err_str and "endpoint" in err_str)
                        _is_auth = ("401" in err_str or "403" in err_str) and (
                            "authentication" in err_str or "unauthorized" in err_str or "forbidden" in err_str or "user not found" in err_str or "invalid api key" in err_str
                        )
                        if _is_no_endpoints:
                            msg = (
                                f"Text providers unavailable via OpenRouter (404 / no endpoints) after {attempts} attempt(s) in {total_time:.2f}s (last_provider={prov}). Last error: {type(base_err).__name__}: {base_err}"
                            )
                        elif _is_moderation:
                            msg = f"Content blocked by moderation (last_provider={prov}): input was flagged. Last error: {type(base_err).__name__}: {base_err}"
                        elif _is_auth:
                            msg = f"Text provider authentication failed (last_provider={prov}). Check OPENAI_API_KEY / OpenRouter account. Last error: {type(base_err).__name__}: {base_err}"
                        elif "timeout" in err_str:
                            msg = f"Text generation timeout after {total_time:.2f}s across {attempts} attempt(s) (last_provider={prov}). Last error: {type(base_err).__name__}: {base_err}"
                        else:
                            msg = f"Text fallback ladder failed after {attempts} attempt(s) in {total_time:.2f}s (last_provider={prov}): {type(base_err).__name__}: {base_err}"
                        api_err = APIError(msg)
                        try:
                            # Ladder exhaustion is ALWAYS non-retryable at this level:
                            # run_with_fallback has already walked every provider with
                            # per-provider retries and backoff. Letting the outer
                            # @with_retry re-run the whole ladder (~90s of known-bad
                            # timeouts) after a ~1s delay just burns the caller's total
                            # budget and turns one failure into a user-facing timeout. [REH][PA]
                            api_err.retryable = False
                            if _is_moderation:
                                api_err.content_moderation = True
                        except Exception as e:
                            logger.debug(f"Failed to classify error retryability: {e}")
                    except Exception:
                        # Fallback to original error if wrapping fails
                        raise base_err
                    # Normal path: propagate a clean APIError up to the caller so it can be
                    # handled without dumping a full traceback in logs. [REH]
                    raise api_err
                msg_0 = "All text providers exhausted"
                exhausted_err = APIError(msg_0)
                exhausted_err.retryable = False  # same rationale as above [REH]
                raise exhausted_err
            return rr.result

        # Non-streaming single-provider path (OpenAI base or when ladder disabled)
        logger.debug("[OpenAI] 🔄 Sending request to API...")

        try:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    presence_penalty=presence_penalty,
                    max_tokens=max_tokens,
                    stream=False,
                    **kwargs,
                )
            except Exception as e:
                if "timeout" in str(e).lower() or "TimeoutError" in str(type(e).__name__):
                    logger.warning(f"[OpenAI] ⏰ Request timeout after {config.get('TEXT_REQUEST_TIMEOUT', 30)}s: {e}")
                    msg_0 = f"Request timeout: {e!s}"
                    raise APIError(msg_0)
                logger.exception(f"[OpenAI] ❌ API request failed: {e}")
                raise
            logger.debug("[OpenAI] ✅ Received response from API")
            # Handle non-streaming response
            return _normalize_nonstream_response(response, model)
        finally:
            await _safe_aclose_openai_client(client)

    except OpenAIAuthenticationError as e:
        logger.exception(f"OpenAI authentication failed: {e}")
        msg_0 = f"OpenAI authentication failed - check API key: {e!s}"
        raise APIError(msg_0)
    except OpenAIRateLimitError as e:
        logger.warning(f"OpenAI rate limit exceeded: {e}")
        retry_after = None
        try:
            # Try common headers if available on the error
            resp = getattr(e, "response", None)
            if resp is not None and getattr(resp, "headers", None) is not None:
                ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                if ra is not None:
                    retry_after = float(ra)
        except Exception as exc:
            logger.debug(f"Failed to parse retry-after header: {exc}")
            retry_after = None
        err = APIError(f"OpenAI rate limit exceeded: {e!s}" + (f" (retry-after={retry_after}s)" if retry_after is not None else ""))
        try:
            if retry_after is not None:
                err.retry_after_seconds = retry_after
        except Exception as exc:
            logger.debug(f"Failed to set retry_after_seconds: {exc}")
        raise err
    except OpenAIAPIError as e:
        _es = str(e).lower()
        _mod = ("flagged" in _es or "moderation" in _es or "self-harm" in _es or "self_harm" in _es or "requires moderation" in _es or "content_filter" in _es) and ("403" in _es or "400" in _es)
        _no_ep = "no endpoints found" in _es or ("404" in _es and "endpoint" in _es)
        if _mod:
            logger.warning(f"[OpenAI] Content moderation block (not retried): {e}")
            _err = APIError(f"Content blocked by moderation: {e!s}")
            _err.retryable = False
            _err.content_moderation = True
            raise _err
        elif _no_ep:
            logger.warning(f"[OpenAI] Model/endpoint not found: {e}")
            _err = APIError(f"Model not found (no endpoints): {e!s}")
            _err.retryable = False
            raise _err
        else:
            logger.exception(f"OpenAI API error: {e}")
            msg_0 = f"OpenAI API error: {e!s}"
            raise APIError(msg_0)
    except httpx.HTTPStatusError as e:
        # Surface HTTP errors (e.g., 429 Too Many Requests from OpenRouter) as retriable APIError
        status = e.response.status_code if e.response is not None else "unknown"
        retry_after = None
        try:
            if getattr(e, "response", None) is not None:
                hdrs = e.response.headers
                ra = hdrs.get("retry-after") or hdrs.get("Retry-After")
                if ra is not None:
                    retry_after = float(ra)
                else:
                    # OpenRouter-style reset headers (epoch or delta seconds)
                    reset = hdrs.get("x-ratelimit-reset") or hdrs.get("X-RateLimit-Reset") or hdrs.get("rate-limit-reset")
                    if reset is not None:
                        try:
                            val = float(reset)
                            now = time.time()
                            retry_after = val - now if val > now + 1 else val
                            if retry_after < 0:
                                retry_after = 0.0
                        except Exception:
                            retry_after = None
        except Exception:
            retry_after = None
        extra = f" (retry-after={retry_after}s)" if retry_after is not None else ""
        logger.warning(f"OpenAI HTTP error: {status} {e}{extra}")
        err = APIError(f"HTTP {status}: {e!s}{extra}")
        try:
            if retry_after is not None:
                err.retry_after_seconds = retry_after
        except Exception as exc:
            logger.debug(f"Failed to set retry_after_seconds: {exc}")
        raise err
    except APIError as e:
        # Already normalized, don't double-wrap or spam error-level logs
        logger.warning(f"[OpenAI] Retriable APIError: {e}")
        raise

    except Exception as e:
        # Get detailed error information
        error_type = type(e).__name__
        error_msg = str(e) or "No error message"
        error_details = f"{error_type}: {error_msg}"

        logger.exception(f"Unexpected error in generate_openai_response: {error_details}")
        logger.debug(
            "Traceback for unexpected error in generate_openai_response",
            exc_info=True,
        )
        msg_0 = f"Failed to generate OpenAI response: {error_details}"
        raise APIError(msg_0)


async def get_base64_image(image_url: str) -> str:
    """Process image from URL or file path and convert to base64 data URI.
    CHANGE: Enhanced to handle both URLs and file paths.
    """
    logger.debug(f"📥 Processing image from: {image_url}")

    # Handle file paths (file:// protocol or direct file path)
    if image_url.startswith("file://") or os.path.exists(image_url):
        try:
            # Extract actual path from file:// URL if needed
            file_path = image_url.removeprefix("file://")

            # Verify file exists
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                logger.error(error_msg)
                raise APIError(error_msg)

            # Read file and encode to base64
            with open(file_path, "rb") as f:
                data = f.read()

            # Determine content type from file extension
            ext = os.path.splitext(file_path)[1].lower()
            content_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/webp" if ext == ".webp" else "image/gif" if ext == ".gif" else "image/png"  # Default to PNG

            base64_data = base64.b64encode(data).decode("utf-8")
            logger.debug(f"✅ Image loaded from file: size={len(data)} bytes, type={content_type}")
            return f"data:{content_type};base64,{base64_data}"

        except Exception as e:
            error_msg = f"Failed to process image file: {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)

    # Handle HTTP/HTTPS URLs
    elif image_url.startswith(("http://", "https://")):
        try:
            session = _get_image_session()
            async with session.get(image_url) as response:
                if response.status == 200:
                    data = await response.read()
                    base64_data = base64.b64encode(data).decode("utf-8")
                    content_type = response.headers.get("Content-Type", "image/png")
                    logger.debug(f"✅ Image downloaded: size={len(data)} bytes, type={content_type}")
                    return f"data:{content_type};base64,{base64_data}"
                error_msg = f"Failed to download image: status={response.status}"
                logger.error(error_msg)
                raise APIError(error_msg)
        except Exception as e:
            error_msg = f"Failed to download image from URL: {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)

    # Already a data URL
    elif image_url.startswith("data:"):
        logger.debug("✅ Image already in data URL format")
        return image_url

    # Unsupported format
    else:
        error_msg = f"Unsupported image source format: {image_url[:30]}..."
        logger.error(error_msg)
        raise APIError(error_msg)


async def _generate_vl_response_with_retry(
    image_url: str,
    user_prompt: str = "",
    user_id: str | None = None,
    guild_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Attempt VL inference using EnhancedRetryManager vision ladder with fallback."""
    config = load_config()

    # Handle model_override from kwargs (for backward compatibility with see_infer)
    model_override = kwargs.pop("model_override", None)

    base_url = str(config.get("OPENAI_API_BASE", "https://api.openai.com/v1") or "")
    stream_flag = bool(kwargs.get("stream", False))

    import httpx

    vl_prompt_file_path = config.get("VL_PROMPT_FILE")
    if not vl_prompt_file_path:
        msg = "VL_PROMPT_FILE not configured in environment variables"
        raise APIError(msg)

    try:
        vl_system_prompt = _load_prompt_cached(vl_prompt_file_path)
    except FileNotFoundError as exc:
        msg = f"VL prompt file not found: {vl_prompt_file_path}"
        raise APIError(msg) from exc
    except Exception as exc:
        msg = f"Error reading VL prompt file {vl_prompt_file_path}: {exc}"
        raise APIError(msg) from exc

    if user_id:
        profile = get_profile(str(user_id))
        user_prefs = profile.get("preferences", {}) if profile else {}
        if temperature is None:
            temperature = user_prefs.get("temperature", config.get("TEMPERATURE", 0.7))
    else:
        temperature = temperature or config.get("TEMPERATURE", 0.7)

    system_prompt = vl_system_prompt
    user_message_text = user_prompt or "Analyze this image."

    if isinstance(image_url, str) and image_url.startswith("data:"):
        image_content = image_url
    else:
        try:
            image_content = await get_base64_image(image_url)
        except Exception as exc:
            msg = f"Image processing failed: {exc}"
            raise APIError(msg) from exc

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_message_text},
                {"type": "image_url", "image_url": {"url": image_content}},
            ],
        },
    ]

    if max_tokens is None:
        max_tokens = config.get("MAX_RESPONSE_TOKENS", 1000)

    OPENROUTER_ALLOWED_PARAMS = {
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "stream",
    }

    # Use EnhancedRetryManager for vision ladder fallback (mirrors text flow) [CA][REH]
    # Skip ladder if model_override is explicitly provided (caller wants specific model)
    use_provider_fallback = ("openrouter" in base_url.lower() or "nvidia.com" in base_url.lower()) and not model_override

    if use_provider_fallback:
        ladder_label = "NVIDIA" if "nvidia.com" in base_url.lower() else "OpenRouter"
        logger.info(f"[VL] Using EnhancedRetryManager vision fallback ladder ({ladder_label})")
        retry_mgr = get_retry_manager()

        def _coro_factory(provider_config):
            selected_model = provider_config.model
            attempt_api_key, attempt_base_url, attempt_provider = _resolve_openai_compatible_endpoint(provider_config.name, config)

            async def _run():
                # Create client with per-provider timeout from ladder config
                client = _make_openai_async_client(
                    api_key=attempt_api_key,
                    base_url=attempt_base_url,
                    timeout=httpx.Timeout(provider_config.timeout),
                    max_retries=0,
                )

                api_params = {
                    "model": selected_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                # Filter to allowed params
                api_params = {k: v for k, v in api_params.items() if k in OPENROUTER_ALLOWED_PARAMS}

                t0 = time.monotonic()
                try:
                    logger.debug(f"[VL] 🔄 Sending request to {attempt_provider} with model: {selected_model}")
                    response = await client.chat.completions.create(**api_params)

                    if response is None:
                        msg_1 = "VL API returned None response"
                        raise APIError(msg_1)
                    if not (hasattr(response, "choices") and response.choices and hasattr(response.choices[0], "message") and response.choices[0].message):
                        # Surface the embedded provider error (OpenRouter returns
                        # HTTP 200 + error body when free VL models are saturated)
                        # so logs show the cause and embedded status codes like
                        # 429/502 make the failure correctly retryable. [REH]
                        msg_1 = f"Invalid VL API response structure{_embedded_provider_error_detail(response)}"
                        raise APIError(msg_1)

                    logger.debug("[VL] ✅ Received response from API")

                    # Extract response content
                    try:
                        response_text = response.choices[0].message.content or ""
                    except Exception:
                        response_text = ""

                    # Empty completion is a soft failure — retry with next model [REH]
                    if not response_text.strip():
                        msg_1 = f"VL model {selected_model} returned empty completion"
                        raise APIError(msg_1)
                    usage = getattr(response, "usage", None)
                    usage_info = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage, "completion_tokens", 0),
                        "total_tokens": getattr(usage, "total_tokens", 0),
                    }

                    return {
                        "text": response_text,
                        "model": selected_model,
                        "usage": usage_info,
                        "backend": attempt_provider,
                        "telemetry": {"provider_base": attempt_base_url},
                    }

                except httpx.HTTPStatusError as he:
                    status_code = he.response.status_code if getattr(he, "response", None) is not None else "unknown"
                    retry_after = None
                    try:
                        if getattr(he, "response", None) is not None:
                            hdrs = he.response.headers
                            ra = hdrs.get("retry-after") or hdrs.get("Retry-After")
                            if ra is not None:
                                retry_after = float(ra)
                    except Exception as exc:
                        logger.debug(f"Failed to parse retry-after header: {exc}")
                        retry_after = None
                    extra = f" (retry-after={retry_after}s)" if retry_after is not None else ""
                    logger.warning(f"VL HTTP error during fallback attempt: {status_code} {he}{extra}")
                    err = APIError(f"HTTP {status_code}: {he!s}{extra}")
                    try:
                        if retry_after is not None:
                            err.retry_after_seconds = retry_after
                    except Exception as exc:
                        logger.debug(f"Failed to set retry_after_seconds: {exc}")
                    raise err
                except Exception as e:
                    elapsed_ms = int((time.monotonic() - t0) * 1000)
                    etype = type(e).__name__
                    if "timeout" in str(e).lower() or "timeout" in etype.lower():
                        logger.warning(f"VL timeout (model {selected_model}): {elapsed_ms}ms")
                        msg_1 = f"Request timeout after ~{elapsed_ms}ms for model {selected_model}"
                        raise APIError(msg_1)
                    raise
                finally:
                    await _safe_aclose_openai_client(client)

            return _run

        # Use configurable VL budget; reply-image perception has its own outer timeout,
        # so this must be long enough for large images and slow free OpenRouter VL models. [REH]
        try:
            per_item_budget = float(config.get("VISION_PER_ITEM_BUDGET", 45.0))
        except Exception as exc:
            logger.debug(f"Failed to parse VISION_PER_ITEM_BUDGET: {exc}")
            per_item_budget = 45.0
        with contextlib.suppress(Exception):
            logger.info(f"[VL] vision.budget seconds={per_item_budget}")

        rr = await retry_mgr.run_with_fallback("vision", _coro_factory, per_item_budget=per_item_budget)

        if not rr.success:
            # Enrich error with ladder context for better observability [REH]
            if rr.error:
                base_err = rr.error
                try:
                    prov = getattr(rr, "provider_used", None) or getattr(base_err, "provider_key", None) or "unknown"
                    attempts = getattr(rr, "attempts", 0)
                    total_time = getattr(rr, "total_time", 0.0)
                    err_str = str(base_err).lower()
                    _is_mod = ("flagged" in err_str or "moderation" in err_str or "self-harm" in err_str or "self_harm" in err_str or "requires moderation" in err_str or "content_filter" in err_str) and (
                        "403" in err_str or "400" in err_str
                    )
                    _is_no_ep = "no endpoints found" in err_str or ("404" in err_str and "endpoint" in err_str)
                    _is_auth = ("401" in err_str or "403" in err_str) and (
                        "authentication" in err_str or "unauthorized" in err_str or "forbidden" in err_str or "user not found" in err_str or "invalid api key" in err_str
                    )

                    if _is_no_ep:
                        msg = f"Vision providers unavailable via OpenRouter (404 / no endpoints) after {attempts} attempt(s) in {total_time:.2f}s (last_provider={prov}). Last error: {type(base_err).__name__}: {base_err}"
                    elif _is_mod:
                        msg = f"Vision content blocked by moderation (last_provider={prov}): input was flagged. Last error: {type(base_err).__name__}: {base_err}"
                    elif _is_auth:
                        msg = f"Vision provider authentication failed (last_provider={prov}). Check OPENAI_API_KEY / OpenRouter account. Last error: {type(base_err).__name__}: {base_err}"
                    elif "timeout" in err_str:
                        msg = f"Vision generation timeout after {total_time:.2f}s across {attempts} attempt(s) (last_provider={prov}). Last error: {type(base_err).__name__}: {base_err}"
                    else:
                        msg = f"Vision fallback ladder failed after {attempts} attempt(s) in {total_time:.2f}s (last_provider={prov}): {type(base_err).__name__}: {base_err}"

                    api_err = APIError(msg)
                    api_err.vl_exhausted = True
                    api_err.vl_ladder_summary = f"attempts={attempts},time={total_time:.2f}s,last={prov}"
                    api_err.vl_attempts = attempts
                    api_err.vl_provider_base = base_url
                    try:
                        if _is_no_ep or _is_auth or _is_mod:
                            api_err.retryable = False
                        if _is_mod:
                            api_err.content_moderation = True
                    except Exception as exc:
                        logger.debug(f"Failed to classify error retryability: {exc}")
                except Exception:
                    raise base_err
                raise api_err
            exhaustion_error = APIError("All vision providers exhausted")
            exhaustion_error.vl_exhausted = True
            raise exhaustion_error

        # Success - add telemetry
        result = rr.result
        if isinstance(result, dict):
            result["telemetry"] = result.get("telemetry", {})
            result["telemetry"]["ladder_attempts"] = rr.attempts
            result["telemetry"]["fallback_occurred"] = rr.fallback_occurred
            result["telemetry"]["provider_used"] = rr.provider_used
        return result

    # Fallback: single-provider path (non-OpenRouter base or model_override specified)
    # Use model_override if provided, else first model from VL_MODEL ladder or default
    if model_override:
        model_name = model_override.strip()
        logger.info(f"[VL] Using model_override={model_name} (single-provider mode)")
    else:
        ladder = get_vl_model_ladder()
        if not ladder:
            msg_0 = "No VL models configured"
            raise APIError(msg_0)
        model_name = ladder[0]

    try:
        timeout_seconds = float(config.get("VL_REQUEST_TIMEOUT", "30"))
    except Exception:
        timeout_seconds = 30.0

    single_api_key, single_base_url, single_provider = _resolve_openai_compatible_endpoint(
        "nvidia" if "nvidia.com" in base_url.lower() else "openrouter" if "openrouter" in base_url.lower() else "openai",
        config,
    )
    client = _make_openai_async_client(
        api_key=single_api_key,
        base_url=single_base_url,
        timeout=httpx.Timeout(timeout_seconds),
        max_retries=0,
    )

    logger.info(
        "vl.attempt model=%s provider_base=%s stream=%s (single-provider mode)",
        model_name,
        single_base_url,
        stream_flag,
    )

    api_params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    api_params = {k: v for k, v in api_params.items() if k in OPENROUTER_ALLOWED_PARAMS}

    start = time.monotonic()
    try:
        try:
            response = await client.chat.completions.create(**api_params)
            if response is None:
                msg_0 = "VL API returned None response"
                raise APIError(msg_0)
            if not (hasattr(response, "choices") and response.choices and hasattr(response.choices[0], "message") and response.choices[0].message):
                msg_0 = "Invalid VL API response structure"
                raise APIError(msg_0)
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.warning(
                "vl.fail model=%s latency_ms=%d error=%s",
                model_name,
                elapsed_ms,
                str(exc)[:200],
            )
            exhaustion_error = APIError("Vision request failed")
            exhaustion_error.vl_exhausted = True
            exhaustion_error.vl_ladder_summary = f"{model_name}:fail"
            exhaustion_error.vl_attempts = 1
            exhaustion_error.vl_provider_base = single_base_url
            raise exhaustion_error from exc

        elapsed_ms = int((time.monotonic() - start) * 1000)
        usage = getattr(response, "usage", None)
        total_tokens = getattr(usage, "total_tokens", None)

        logger.info(
            "vl.ok model=%s ms=%d tokens_total=%s",
            model_name,
            elapsed_ms,
            total_tokens if total_tokens is not None else "na",
        )

        try:
            response_text = response.choices[0].message.content or ""
        except Exception:
            response_text = ""

        # Empty completion is a soft failure — retry or fail [REH]
        if not response_text.strip():
            empty_err = APIError(f"VL model {model_name} returned empty completion")
            empty_err.vl_exhausted = True
            empty_err.vl_ladder_summary = f"{model_name}:empty_completion"
            empty_err.vl_attempts = 1
            empty_err.vl_provider_base = single_base_url
            raise empty_err
        usage_info = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": total_tokens or 0,
        }

        return {
            "text": response_text,
            "model": model_name,
            "usage": usage_info,
            "backend": single_provider,
            "telemetry": {"provider_base": single_base_url, "ladder_attempts": 1},
        }
    finally:
        await _safe_aclose_openai_client(client)


async def generate_vl_response(
    image_url: str,
    user_prompt: str = "",
    user_id: str | None = None,
    guild_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Generate a VL response with laddered fallback and user-friendly failures."""
    logger.info("🎨 Starting VL response generation with ladder fallback")
    logger.debug(f"🎨 User: {user_id}, Guild: {guild_id}")

    try:
        result = await _generate_vl_response_with_retry(
            image_url=image_url,
            user_prompt=user_prompt,
            user_id=user_id,
            guild_id=guild_id,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        telemetry = result.get("telemetry", {}) or {}
        logger.info(
            "vl.final status=ok exhausted=false model=%s idx=%s attempts=%s provider_base=%s",
            result.get("model"),
            telemetry.get("ladder_idx"),
            telemetry.get("ladder_attempts"),
            telemetry.get("provider_base"),
        )
        return result

    except APIError as api_error:
        if getattr(api_error, "vl_exhausted", False):
            ladder_summary = getattr(api_error, "vl_ladder_summary", "none")
            attempts = getattr(api_error, "vl_attempts", 0)
            provider_base = getattr(api_error, "vl_provider_base", "unknown")
            logger.warning(
                "vl.final status=error exhausted=true ladder=%s attempts=%s provider_base=%s",
                ladder_summary,
                attempts,
                provider_base,
            )
            friendly_message = "🔧 The vision service is temporarily unavailable. Please try again in a few minutes."
            return {
                "text": friendly_message,
                "model": None,
                "usage": None,
                "backend": "openai",
                "ladder_exhausted": True,
                "telemetry": {
                    "ladder_summary": ladder_summary,
                    "ladder_attempts": attempts,
                    "provider_base": provider_base,
                },
            }
        raise
