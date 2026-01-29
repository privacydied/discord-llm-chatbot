"""
OpenAI/OpenRouter Backend - Handles OpenAI API calls including OpenRouter.
"""

from typing import Any, AsyncGenerator, Dict, Union
import base64
import os
import time

import aiohttp
import openai

from bot.config import load_config, get_vl_model_ladder
from bot.exceptions import APIError
from bot.memory import get_profile, get_server_profile
from bot.retry_utils import (
    API_RETRY_CONFIG,
    with_retry,
)
from bot.utils.logging import get_logger
from bot.enhanced_retry import get_retry_manager

logger = get_logger(__name__)


@with_retry(API_RETRY_CONFIG)
async def generate_openai_response(
    prompt: str,
    context: str = "",
    system_prompt: str = None,
    user_id: str = None,
    guild_id: str = None,
    temperature: float = None,
    max_tokens: int = None,
    stream: bool = False,
    **kwargs,
) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
    """
    Generate a response using OpenAI API (including OpenRouter).

    Args:
        prompt: The user's input prompt
        context: Optional context to include in the prompt
        user_id: Optional user ID for personalization
        guild_id: Optional guild ID for server-specific context
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        stream: Whether to stream the response
        **kwargs: Additional parameters to pass to the API

    Returns:
        Dictionary with the generated text and metadata
    """
    try:
        config = load_config()

        # Configure OpenAI client with configurable timeout
        import httpx

        # Prefer TEXTGEN_TIMEOUT_SECONDS; fall back to TEXT_REQUEST_TIMEOUT; default 45s
        try:
            timeout_seconds = float(
                config.get(
                    "TEXTGEN_TIMEOUT_SECONDS", config.get("TEXT_REQUEST_TIMEOUT", "45")
                )
            )
        except Exception:
            timeout_seconds = 45.0

        client = openai.AsyncOpenAI(
            api_key=config.get("OPENAI_API_KEY"),
            base_url=config.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            timeout=httpx.Timeout(timeout_seconds),
            max_retries=0,  # Let our retry system handle retries
        )

        # Get model configuration
        model = config.get("OPENAI_TEXT_MODEL", "gpt-4")

        # Get user preferences if user_id is provided
        if user_id:
            profile = get_profile(str(user_id))
            user_prefs = profile.get("preferences", {}) if profile else {}

            # Apply user preferences if not overridden
            if temperature is None:
                temperature = user_prefs.get(
                    "temperature", config.get("TEMPERATURE", 0.7)
                )
        else:
            temperature = temperature or config.get("TEMPERATURE", 0.7)

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
                raise APIError("PROMPT_FILE not configured in environment variables")
            try:
                with open(prompt_file_path, "r", encoding="utf-8") as pf:
                    base_system_prompt = pf.read().strip()
            except FileNotFoundError:
                raise APIError(f"Prompt file not found: {prompt_file_path}")
            except Exception as e:
                raise APIError(f"Error reading prompt file {prompt_file_path}: {e}")

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
                }
            )

        # Finally, add the user's actual prompt
        messages.append({"role": "user", "content": prompt})

        # Set default max_tokens if not provided
        if max_tokens is None:
            max_tokens = config.get("MAX_RESPONSE_TOKENS", 1000)

        logger.info(f"Generating OpenAI response with model: {model}")
        logger.debug(
            f"[OpenAI] Request params: temp={temperature}, max_tokens={max_tokens}, stream={stream}"
        )

        # Helper to normalize a completion response into our result dict
        def _normalize_nonstream_response(
            response_obj, used_model: str
        ) -> Dict[str, Any]:
            try:
                if not getattr(response_obj, "choices", None):
                    raise APIError("No choices returned in OpenAI response")
                first = response_obj.choices[0]
                content = getattr(getattr(first, "message", None), "content", "") or ""
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
                raise APIError(
                    f"Response normalization failed: {type(norm_error).__name__}: {norm_error}"
                )

        if stream:
            logger.debug("[OpenAI] 🔄 Sending request to API (streaming)...")
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )
            logger.debug("[OpenAI] ✅ Received response from API (streaming)")
            logger.debug("[OpenAI] 🌊 Processing streaming response...")

            async def stream_generator():
                chunk_count = 0
                async for chunk in response:
                    chunk_count += 1
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield {
                            "text": chunk.choices[0].delta.content,
                            "finished": False,
                        }
                logger.debug(
                    f"[OpenAI] ✅ Streaming complete, processed {chunk_count} chunks"
                )
                yield {"text": "", "finished": True}

            return stream_generator()

        # Non-streaming: if using OpenRouter base, engage provider/model fallback ladder
        base_url = str(config.get("OPENAI_API_BASE", "https://api.openai.com/v1") or "")
        use_openrouter_fallback = "openrouter" in base_url.lower()

        if use_openrouter_fallback:
            logger.info(
                "[OpenAI] Using EnhancedRetryManager text fallback ladder (OpenRouter)"
            )
            retry_mgr = get_retry_manager()

            def _coro_factory(provider_config):
                selected_model = provider_config.model

                async def _run():
                    try:
                        logger.debug(
                            f"[OpenAI] 🔄 Sending request to API with model: {selected_model}"
                        )
                        t0 = time.monotonic()
                        resp = await client.chat.completions.create(
                            model=selected_model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=False,
                            **kwargs,
                        )
                        logger.debug("[OpenAI] ✅ Received response from API")
                        return _normalize_nonstream_response(resp, selected_model)
                    except httpx.HTTPStatusError as he:
                        status_code = (
                            he.response.status_code
                            if getattr(he, "response", None) is not None
                            else "unknown"
                        )
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
                                    reset = (
                                        hdrs.get("x-ratelimit-reset")
                                        or hdrs.get("X-RateLimit-Reset")
                                        or hdrs.get("rate-limit-reset")
                                    )
                                    if reset is not None:
                                        try:
                                            val = float(reset)
                                            now = time.time()
                                            retry_after = (
                                                val - now if val > now + 1 else val
                                            )
                                            if retry_after < 0:
                                                retry_after = 0.0
                                        except Exception:
                                            retry_after = None
                        except Exception:
                            retry_after = None
                        extra = (
                            f" (retry-after={retry_after}s)"
                            if retry_after is not None
                            else ""
                        )
                        logger.warning(
                            f"OpenAI HTTP error during fallback attempt: {status_code} {he}{extra}"
                        )
                        err = APIError(f"HTTP {status_code}: {str(he)}{extra}")
                        # Propagate Retry-After to outer retry harness
                        try:
                            if retry_after is not None:
                                setattr(err, "retry_after_seconds", retry_after)
                        except Exception:
                            pass
                        raise err
                    except Exception as e:
                        # Normalize timeouts and log a compact line per attempt [REH]
                        elapsed_ms = int((time.monotonic() - t0) * 1000)
                        etype = type(e).__name__
                        if "timeout" in str(e).lower() or "timeout" in etype.lower():
                            logger.warning(
                                f"TextGen timeout (model {selected_model}): {elapsed_ms}ms"
                            )
                            raise APIError(
                                f"Request timeout after ~{elapsed_ms}ms for model {selected_model}"
                            )
                        raise

                return _run

            per_item_budget = float(config.get("TEXT_PER_ITEM_BUDGET", 120.0))
            try:
                logger.info(f"[OpenAI] text.budget seconds={per_item_budget}")
            except Exception:
                pass
            rr = await retry_mgr.run_with_fallback(
                "text", _coro_factory, per_item_budget=per_item_budget
            )
            if not rr.success:
                # Enrich error with ladder context for better observability [REH]
                if rr.error:
                    base_err = rr.error
                    try:
                        prov = (
                            getattr(rr, "provider_used", None)
                            or getattr(base_err, "provider_key", None)
                            or "unknown"
                        )
                        attempts = getattr(rr, "attempts", 0)
                        total_time = getattr(rr, "total_time", 0.0)
                        err_str = str(base_err).lower()
                        if "no endpoints found" in err_str or (
                            "404" in err_str and "endpoint" in err_str
                        ):
                            msg = (
                                "Text providers unavailable via OpenRouter (404 / no endpoints) "
                                f"after {attempts} attempt(s) in {total_time:.2f}s "
                                f"(last_provider={prov}). Last error: {type(base_err).__name__}: {base_err}"
                            )
                        elif "timeout" in err_str:
                            msg = (
                                f"Text generation timeout after {total_time:.2f}s across "
                                f"{attempts} attempt(s) (last_provider={prov}). "
                                f"Last error: {type(base_err).__name__}: {base_err}"
                            )
                        else:
                            msg = (
                                f"Text fallback ladder failed after {attempts} attempt(s) in "
                                f"{total_time:.2f}s (last_provider={prov}): {type(base_err).__name__}: {base_err}"
                            )
                        api_err = APIError(msg)
                        try:
                            if "no endpoints found" in err_str or (
                                "404" in err_str and "endpoint" in err_str
                            ):
                                setattr(api_err, "retryable", False)
                        except Exception:
                            pass
                    except Exception:
                        # Fallback to original error if wrapping fails
                        raise base_err
                    # Normal path: propagate a clean APIError up to the caller so it can be
                    # handled without dumping a full traceback in logs. [REH]
                    raise api_err
                raise APIError("All text providers exhausted")
            return rr.result

        # Non-streaming single-provider path (OpenAI base or when ladder disabled)
        logger.debug("[OpenAI] 🔄 Sending request to API...")

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                **kwargs,
            )
        except Exception as e:
            if "timeout" in str(e).lower() or "TimeoutError" in str(type(e).__name__):
                logger.warning(
                    f"[OpenAI] ⏰ Request timeout after {config.get('TEXT_REQUEST_TIMEOUT', 30)}s: {e}"
                )
                raise APIError(f"Request timeout: {str(e)}")
            else:
                logger.error(f"[OpenAI] ❌ API request failed: {e}")
                raise
        logger.debug("[OpenAI] ✅ Received response from API")
        # Handle non-streaming response
        result = _normalize_nonstream_response(response, model)
        return result

    except openai.AuthenticationError as e:
        logger.error(f"OpenAI authentication failed: {e}")
        raise APIError(f"OpenAI authentication failed - check API key: {str(e)}")
    except openai.RateLimitError as e:
        logger.warning(f"OpenAI rate limit exceeded: {e}")
        retry_after = None
        try:
            # Try common headers if available on the error
            resp = getattr(e, "response", None)
            if resp is not None and getattr(resp, "headers", None) is not None:
                ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                if ra is not None:
                    retry_after = float(ra)
        except Exception:
            retry_after = None
        err = APIError(
            f"OpenAI rate limit exceeded: {str(e)}"
            + (f" (retry-after={retry_after}s)" if retry_after is not None else "")
        )
        try:
            if retry_after is not None:
                setattr(err, "retry_after_seconds", retry_after)
        except Exception:
            pass
        raise err
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise APIError(f"OpenAI API error: {str(e)}")
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
                    reset = (
                        hdrs.get("x-ratelimit-reset")
                        or hdrs.get("X-RateLimit-Reset")
                        or hdrs.get("rate-limit-reset")
                    )
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
        err = APIError(f"HTTP {status}: {str(e)}{extra}")
        try:
            if retry_after is not None:
                setattr(err, "retry_after_seconds", retry_after)
        except Exception:
            pass
        raise err
    except APIError as e:
        # Already normalized, don't double-wrap or spam error-level logs
        logger.warning(f"[OpenAI] Retriable APIError: {e}")
        raise

    except Exception as e:
        # Get detailed error information
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else "No error message"
        error_details = f"{error_type}: {error_msg}"

        logger.error(f"Unexpected error in generate_openai_response: {error_details}")
        logger.debug(
            "Traceback for unexpected error in generate_openai_response",
            exc_info=True,
        )
        raise APIError(f"Failed to generate OpenAI response: {error_details}")


async def get_base64_image(image_url: str) -> str:
    """
    Process image from URL or file path and convert to base64 data URI.
    CHANGE: Enhanced to handle both URLs and file paths.
    """
    logger.debug(f"📥 Processing image from: {image_url}")

    # Handle file paths (file:// protocol or direct file path)
    if image_url.startswith("file://") or os.path.exists(image_url):
        try:
            # Extract actual path from file:// URL if needed
            file_path = image_url[7:] if image_url.startswith("file://") else image_url

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
            content_type = (
                "image/jpeg"
                if ext in (".jpg", ".jpeg")
                else "image/png"
                if ext == ".png"
                else "image/webp"
                if ext == ".webp"
                else "image/gif"
                if ext == ".gif"
                else "image/png"
            )  # Default to PNG

            base64_data = base64.b64encode(data).decode("utf-8")
            logger.debug(
                f"✅ Image loaded from file: size={len(data)} bytes, type={content_type}"
            )
            return f"data:{content_type};base64,{base64_data}"

        except Exception as e:
            error_msg = f"Failed to process image file: {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)

    # Handle HTTP/HTTPS URLs
    elif image_url.startswith(("http://", "https://")):
        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=10)
                async with session.get(image_url, timeout=timeout) as response:
                    if response.status == 200:
                        data = await response.read()
                        base64_data = base64.b64encode(data).decode("utf-8")
                        content_type = response.headers.get("Content-Type", "image/png")
                        logger.debug(
                            f"✅ Image downloaded: size={len(data)} bytes, type={content_type}"
                        )
                        return f"data:{content_type};base64,{base64_data}"
                    else:
                        error_msg = (
                            f"Failed to download image: status={response.status}"
                        )
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
    user_id: str = None,
    guild_id: str = None,
    temperature: float = None,
    max_tokens: int = None,
    **kwargs,
) -> Dict[str, Any]:
    """Attempt VL inference using EnhancedRetryManager vision ladder with fallback."""
    config = load_config()

    # Handle model_override from kwargs (for backward compatibility with see_infer)
    model_override = kwargs.pop("model_override", None)

    base_url = str(config.get("OPENAI_API_BASE", "https://api.openai.com/v1") or "")
    stream_flag = bool(kwargs.get("stream", False))

    import httpx

    vl_prompt_file_path = config.get("VL_PROMPT_FILE")
    if not vl_prompt_file_path:
        raise APIError("VL_PROMPT_FILE not configured in environment variables")

    try:
        with open(vl_prompt_file_path, "r", encoding="utf-8") as vpf:
            vl_system_prompt = vpf.read().strip()
    except FileNotFoundError as exc:
        raise APIError(f"VL prompt file not found: {vl_prompt_file_path}") from exc
    except Exception as exc:
        raise APIError(
            f"Error reading VL prompt file {vl_prompt_file_path}: {exc}"
        ) from exc

    if user_id:
        profile = get_profile(str(user_id))
        user_prefs = profile.get("preferences", {}) if profile else {}
        if temperature is None:
            temperature = user_prefs.get("temperature", config.get("TEMPERATURE", 0.7))
    else:
        temperature = temperature or config.get("TEMPERATURE", 0.7)

    system_prompt = vl_system_prompt
    user_message_text = user_prompt if user_prompt else "Analyze this image."

    if isinstance(image_url, str) and image_url.startswith("data:"):
        image_content = image_url
    else:
        try:
            image_content = await get_base64_image(image_url)
        except Exception as exc:
            raise APIError(f"Image processing failed: {exc}") from exc

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
    use_openrouter_fallback = "openrouter" in base_url.lower() and not model_override

    if use_openrouter_fallback:
        logger.info(
            "[VL] Using EnhancedRetryManager vision fallback ladder (OpenRouter)"
        )
        retry_mgr = get_retry_manager()

        def _coro_factory(provider_config):
            selected_model = provider_config.model

            async def _run():
                # Create client with per-provider timeout from ladder config
                client = openai.AsyncOpenAI(
                    api_key=config.get("OPENAI_API_KEY"),
                    base_url=base_url or "https://api.openai.com/v1",
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
                api_params = {
                    k: v
                    for k, v in api_params.items()
                    if k in OPENROUTER_ALLOWED_PARAMS
                }

                t0 = time.monotonic()
                try:
                    logger.debug(
                        f"[VL] 🔄 Sending request to API with model: {selected_model}"
                    )
                    response = await client.chat.completions.create(**api_params)

                    if response is None:
                        raise APIError("VL API returned None response")
                    if not (
                        hasattr(response, "choices")
                        and response.choices
                        and hasattr(response.choices[0], "message")
                        and response.choices[0].message
                    ):
                        raise APIError("Invalid VL API response structure")

                    logger.debug("[VL] ✅ Received response from API")

                    # Extract response content
                    try:
                        response_text = response.choices[0].message.content or ""
                    except Exception:
                        response_text = ""

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
                        "backend": "openai",
                        "telemetry": {"provider_base": base_url},
                    }

                except httpx.HTTPStatusError as he:
                    status_code = (
                        he.response.status_code
                        if getattr(he, "response", None) is not None
                        else "unknown"
                    )
                    retry_after = None
                    try:
                        if getattr(he, "response", None) is not None:
                            hdrs = he.response.headers
                            ra = hdrs.get("retry-after") or hdrs.get("Retry-After")
                            if ra is not None:
                                retry_after = float(ra)
                    except Exception:
                        retry_after = None
                    extra = (
                        f" (retry-after={retry_after}s)"
                        if retry_after is not None
                        else ""
                    )
                    logger.warning(
                        f"VL HTTP error during fallback attempt: {status_code} {he}{extra}"
                    )
                    err = APIError(f"HTTP {status_code}: {str(he)}{extra}")
                    try:
                        if retry_after is not None:
                            setattr(err, "retry_after_seconds", retry_after)
                    except Exception:
                        pass
                    raise err
                except Exception as e:
                    elapsed_ms = int((time.monotonic() - t0) * 1000)
                    etype = type(e).__name__
                    if "timeout" in str(e).lower() or "timeout" in etype.lower():
                        logger.warning(
                            f"VL timeout (model {selected_model}): {elapsed_ms}ms"
                        )
                        raise APIError(
                            f"Request timeout after ~{elapsed_ms}ms for model {selected_model}"
                        )
                    raise

            return _run

        # Use default per-item budget of 45s for vision (consistent with EnhancedRetryManager default)
        per_item_budget = 45.0
        try:
            logger.info(f"[VL] vision.budget seconds={per_item_budget}")
        except Exception:
            pass

        rr = await retry_mgr.run_with_fallback(
            "vision", _coro_factory, per_item_budget=per_item_budget
        )

        if not rr.success:
            # Enrich error with ladder context for better observability [REH]
            if rr.error:
                base_err = rr.error
                try:
                    prov = (
                        getattr(rr, "provider_used", None)
                        or getattr(base_err, "provider_key", None)
                        or "unknown"
                    )
                    attempts = getattr(rr, "attempts", 0)
                    total_time = getattr(rr, "total_time", 0.0)
                    err_str = str(base_err).lower()

                    if "no endpoints found" in err_str or (
                        "404" in err_str and "endpoint" in err_str
                    ):
                        msg = (
                            "Vision providers unavailable via OpenRouter (404 / no endpoints) "
                            f"after {attempts} attempt(s) in {total_time:.2f}s "
                            f"(last_provider={prov}). Last error: {type(base_err).__name__}: {base_err}"
                        )
                    elif "timeout" in err_str:
                        msg = (
                            f"Vision generation timeout after {total_time:.2f}s across "
                            f"{attempts} attempt(s) (last_provider={prov}). "
                            f"Last error: {type(base_err).__name__}: {base_err}"
                        )
                    else:
                        msg = (
                            f"Vision fallback ladder failed after {attempts} attempt(s) in "
                            f"{total_time:.2f}s (last_provider={prov}): {type(base_err).__name__}: {base_err}"
                        )

                    api_err = APIError(msg)
                    api_err.vl_exhausted = True
                    api_err.vl_ladder_summary = (
                        f"attempts={attempts},time={total_time:.2f}s,last={prov}"
                    )
                    api_err.vl_attempts = attempts
                    api_err.vl_provider_base = base_url
                    try:
                        if "no endpoints found" in err_str or (
                            "404" in err_str and "endpoint" in err_str
                        ):
                            setattr(api_err, "retryable", False)
                    except Exception:
                        pass
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
            raise APIError("No VL models configured")
        model_name = ladder[0]

    try:
        timeout_seconds = float(config.get("VL_REQUEST_TIMEOUT", "30"))
    except Exception:
        timeout_seconds = 30.0

    client = openai.AsyncOpenAI(
        api_key=config.get("OPENAI_API_KEY"),
        base_url=base_url or "https://api.openai.com/v1",
        timeout=httpx.Timeout(timeout_seconds),
        max_retries=0,
    )

    logger.info(
        "vl.attempt model=%s provider_base=%s stream=%s (single-provider mode)",
        model_name,
        base_url,
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
        response = await client.chat.completions.create(**api_params)
        if response is None:
            raise APIError("VL API returned None response")
        if not (
            hasattr(response, "choices")
            and response.choices
            and hasattr(response.choices[0], "message")
            and response.choices[0].message
        ):
            raise APIError("Invalid VL API response structure")
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
        exhaustion_error.vl_provider_base = base_url
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

    usage_info = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": total_tokens or 0,
    }

    return {
        "text": response_text,
        "model": model_name,
        "usage": usage_info,
        "backend": "openai",
        "telemetry": {"provider_base": base_url, "ladder_attempts": 1},
    }


async def generate_vl_response(
    image_url: str,
    user_prompt: str = "",
    user_id: str = None,
    guild_id: str = None,
    temperature: float = None,
    max_tokens: int = None,
    **kwargs,
) -> Dict[str, Any]:
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
