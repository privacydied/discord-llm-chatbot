"""
OpenAI/OpenRouter Backend - Handles OpenAI API calls including OpenRouter.
"""

from typing import Any, AsyncGenerator, Dict, Union
import base64
import os
import time

import aiohttp
import openai

from .config import load_config
from .exceptions import APIError
from .memory import get_profile, get_server_profile
from .retry_utils import (
    API_RETRY_CONFIG,
    VISION_RETRY_CONFIG,
    is_retryable_error,
    with_retry,
)
from .utils.logging import get_logger
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
                                            retry_after = val - now if val > now + 1 else val
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

            per_item_budget = float(config.get("TEXT_PER_ITEM_BUDGET", 45.0))
            rr = await retry_mgr.run_with_fallback(
                "text", _coro_factory, per_item_budget=per_item_budget
            )
            if not rr.success:
                # Enrich error with ladder context for better observability [REH]
                if rr.error:
                    try:
                        prov = (
                            getattr(rr, "provider_used", None)
                            or getattr(rr.error, "provider_key", None)
                            or "unknown"
                        )
                        attempts = getattr(rr, "attempts", 0)
                        total_time = getattr(rr, "total_time", 0.0)
                        msg = f"Text fallback ladder failed after {attempts} attempt(s) in {total_time:.2f}s (last_provider={prov}): {type(rr.error).__name__}: {rr.error}"
                        raise APIError(msg)
                    except Exception:
                        # Fallback to original error if wrapping fails
                        raise rr.error
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

        logger.error(
            f"Unexpected error in generate_openai_response: {error_details}",
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


@with_retry(VISION_RETRY_CONFIG)
async def _generate_vl_response_with_retry(
    image_url: str,
    user_prompt: str = "",
    user_id: str = None,
    guild_id: str = None,
    temperature: float = None,
    max_tokens: int = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate a vision-language response using the VL model and VL_PROMPT_FILE.
    CHANGE: Enhanced VL model support with comprehensive debug logging for hybrid multimodal inference.

    Args:
        image_url: URL or path to the image
        user_prompt: Optional additional user prompt
        user_id: Optional user ID for personalization
        guild_id: Optional guild ID for server-specific context
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        **kwargs: Additional parameters to pass to the API

    Returns:
        Dictionary with the generated VL analysis and metadata
    """
    try:
        # Load configuration
        config = load_config()

        # Extract custom kwargs that should NOT be forwarded to OpenAI client
        model_override = kwargs.pop("model_override", None)

        # Get VL model configuration (honor override) - MUST be defined before logging
        vl_model = model_override or config.get("VL_MODEL")
        if not vl_model:
            logger.error("❌ VL_MODEL not configured in environment variables")
            raise APIError("VL_MODEL not configured in environment variables")

        logger.info("🎨 === VL RESPONSE GENERATION STARTED ===")
        logger.info(f"🎨 Processing with model: {vl_model}")

        if model_override:
            logger.info(f"🎨 Using VL model OVERRIDE from ladder: {vl_model}")
        else:
            logger.info(f"🎨 Using VL model: {vl_model}")

        logger.debug("🎨 Configuring OpenAI client for VL")

        # Configure timeout to prevent hanging requests
        import httpx

        timeout_seconds = float(config.get("VL_REQUEST_TIMEOUT", "30"))

        client = openai.AsyncOpenAI(
            api_key=config.get("OPENAI_API_KEY"),
            base_url=config.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            timeout=httpx.Timeout(timeout_seconds),
            max_retries=0,  # Let our retry system handle retries
        )
        logger.debug(
            f"🎨 Using API base: {config.get('OPENAI_API_BASE', 'https://api.openai.com/v1')}"
        )

        # CHANGE: Load VL prompt from VL_PROMPT_FILE with enhanced logging
        vl_prompt_file_path = config.get("VL_PROMPT_FILE")
        if not vl_prompt_file_path:
            logger.error("❌ VL_PROMPT_FILE not configured in environment variables")
            raise APIError("VL_PROMPT_FILE not configured in environment variables")

        logger.debug(f"🎨 Loading VL prompt from: {vl_prompt_file_path}")
        try:
            with open(vl_prompt_file_path, "r", encoding="utf-8") as vpf:
                vl_system_prompt = vpf.read().strip()
                logger.info(f"✅ VL prompt loaded: {len(vl_system_prompt)} characters")
                logger.debug(
                    f"🎨 VL prompt preview: {vl_system_prompt[:100]}{'...' if len(vl_system_prompt) > 100 else ''}"
                )
        except FileNotFoundError:
            logger.error(f"❌ VL prompt file not found: {vl_prompt_file_path}")
            raise APIError(f"VL prompt file not found: {vl_prompt_file_path}")
        except Exception as e:
            logger.error(f"❌ Error reading VL prompt file {vl_prompt_file_path}: {e}")
            raise APIError(f"Error reading VL prompt file {vl_prompt_file_path}: {e}")

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

        logger.debug(f"🎨 Temperature: {temperature}")

        # Move VL rules to system role, keep user message clean
        system_prompt = vl_system_prompt
        user_message_text = user_prompt if user_prompt else "Analyze this image."

        if user_prompt:
            logger.debug(
                f"🎨 Enhanced prompt with user context: +{len(user_prompt)} chars"
            )

        # Handle data URLs directly, otherwise try to download and encode
        if isinstance(image_url, str) and image_url.startswith("data:"):
            logger.info("✅ Using provided data URL for image")
            image_content = image_url
        else:
            try:
                image_content = await get_base64_image(image_url)
                logger.info("✅ Image downloaded and converted to base64 data URI")
            except Exception as img_error:
                logger.error(f"❌ Failed to process image: {img_error}")
                raise APIError(f"Image processing failed: {str(img_error)}")

        logger.debug(f"🎨 Base64 data length: {len(image_content)} chars")
        logger.debug(f"🎨 Base64 preview: {image_content[:100]}...")

        # Prepare the messages for vision model with system/user role separation
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

        # Set default max_tokens if not provided
        if max_tokens is None:
            max_tokens = config.get("MAX_RESPONSE_TOKENS", 1000)

        logger.debug(f"🎨 Max tokens: {max_tokens}")
        logger.info(f"🎨 Calling OpenAI VL API with model: {vl_model}")

        # Build initial params with all potential values (filter later for OpenRouter)
        raw_params = {
            "model": vl_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        # OpenRouter allow-list: only supported parameters for /chat/completions
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

        # Filter to only allowed parameters for OpenRouter compatibility
        api_params = {
            k: v for k, v in raw_params.items() if k in OPENROUTER_ALLOWED_PARAMS
        }

        # Log filtered parameters for debugging
        if config.get("VL_DEBUG_FLOW", "0") == "1":
            logger.debug(f"🎨 VL payload keys: {list(api_params.keys())}")
            filtered_out = set(raw_params.keys()) - set(api_params.keys())
            if filtered_out:
                logger.debug(f"🎨 Filtered out unsupported params: {filtered_out}")

        # Generate the VL response
        logger.debug(f"🎨 Sending request with messages: {len(messages)} messages")

        try:
            response = await client.chat.completions.create(**api_params)
        except Exception as e:
            # Try VL model fallbacks on parameter or image errors
            if (
                "unexpected keyword" in str(e).lower()
                or "not supported" in str(e).lower()
            ):
                logger.warning(
                    f"🎨 VL call failed with param error, trying fallbacks: {e}"
                )
                response = await _try_vl_fallback_models(
                    client, api_params, vl_model, str(e)
                )
            else:
                raise

        # CHANGE: Enhanced error handling with detailed response logging
        if response is None:
            logger.error("❌ VL API returned None response")
            raise APIError("VL API returned None response")

        # Fast validation with minimal logging
        if not (
            hasattr(response, "choices")
            and response.choices
            and hasattr(response.choices[0], "message")
            and response.choices[0].message
        ):
            logger.error("❌ Invalid VL API response structure")
            raise APIError("Invalid VL API response structure")

        # Removed redundant log message for performance

        # Extract the response text
        response_text = response.choices[0].message.content

        if not response_text:
            logger.warning("⚠️  VL model returned empty response")
        else:
            logger.info(f"✅ VL response: {len(response_text)} chars")

        usage_info = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens
            if response.usage
            else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        logger.info("✅ VL response completed")

        return {
            "text": response_text,
            "model": vl_model,
            "usage": usage_info,
            "backend": "openai",
        }

    except Exception as e:
        logger.error(
            f"❌ Error in _generate_vl_response_with_retry: {e}", exc_info=True
        )
        # Re-raise as APIError to ensure proper retry handling
        if not isinstance(e, APIError):
            raise APIError(f"Failed to generate VL response: {str(e)}")
        raise e


async def _try_vl_fallback_models(client, api_params, failed_model, error_msg):
    """Try VL model fallbacks when primary model fails with param/image errors."""
    from .config import load_config

    config = load_config()

    fallback_models = config.get(
        "VL_MODEL_FALLBACKS", "openai/gpt-4o-mini,anthropic/claude-3.5-sonnet"
    ).split(",")
    fallback_models = [
        m.strip() for m in fallback_models if m.strip() and m.strip() != failed_model
    ]

    if not fallback_models:
        logger.error(
            f"❌ No VL fallback models configured, original error: {error_msg}"
        )
        raise APIError(
            f"VL model {failed_model} failed and no fallbacks available: {error_msg}"
        )

    for fallback_model in fallback_models:
        try:
            logger.info(f"🎨 Trying VL fallback model: {fallback_model}")
            fallback_params = api_params.copy()
            fallback_params["model"] = fallback_model

            response = await client.chat.completions.create(**fallback_params)
            logger.info(f"✅ VL fallback successful with: {fallback_model}")
            return response

        except Exception as fallback_error:
            logger.warning(
                f"🎨 VL fallback {fallback_model} also failed: {fallback_error}"
            )
            continue

    # All fallbacks failed
    logger.error(f"❌ All VL fallbacks exhausted, original error: {error_msg}")
    raise APIError(f"VL model {failed_model} and all fallbacks failed: {error_msg}")


async def generate_vl_response(
    image_url: str,
    user_prompt: str = "",
    user_id: str = None,
    guild_id: str = None,
    temperature: float = None,
    max_tokens: int = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate a vision-language response with robust error handling and retry logic.

    This function wraps the core VL generation with retry logic for transient errors
    and provides fallback mechanisms for better user experience.

    Args:
        image_url: URL or file path to the image
        user_prompt: Text prompt for the vision model
        user_id: Discord user ID for logging
        guild_id: Discord guild ID for logging
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
        **kwargs: Additional arguments for the API call

    Returns:
        Dict containing the response text, model info, and usage stats

    Raises:
        APIError: For API-related errors after all retries are exhausted
        InferenceError: For inference-specific errors
    """
    try:
        logger.info("🎨 Starting VL response generation with retry logic")
        logger.debug(f"🎨 User: {user_id}, Guild: {guild_id}")

        # Call the retry-wrapped function
        result = await _generate_vl_response_with_retry(
            image_url=image_url,
            user_prompt=user_prompt,
            user_id=user_id,
            guild_id=guild_id,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        logger.info("✅ VL response generation completed successfully")
        return result

    except Exception as e:
        logger.error(f"❌ VL response generation failed after all retries: {e}")

        # Check if this was a retryable error that exhausted retries
        if is_retryable_error(e, VISION_RETRY_CONFIG):
            logger.warning(
                "⚠️ Provider appears to be experiencing issues. This may be temporary."
            )
            # Provide a more user-friendly error message for transient issues
            raise APIError(
                "The vision service is temporarily unavailable due to provider issues. "
                "Please try again in a few minutes. If the problem persists, the provider "
                "may be experiencing extended downtime."
            )

        # For non-retryable errors, re-raise as-is
        raise e
