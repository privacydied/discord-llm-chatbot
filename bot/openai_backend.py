"""
OpenAI/OpenRouter Backend - Handles OpenAI API calls including OpenRouter.
"""
from typing import Dict, Any, Union, AsyncGenerator
import openai
import aiohttp
import httpx
import base64
import os

try:
    from .config import load_config
    from .memory import get_profile, get_server_profile
    from .exceptions import APIError
    from .util.logging import get_logger
    from .retry_utils import with_retry, VISION_RETRY_CONFIG, API_RETRY_CONFIG
    from .action import BotAction
    # Optional: provider/model fallback for OpenRouter
    from .enhanced_retry import get_retry_manager
except Exception:
    # Standalone import fallback for smoke tests: define minimal shims
    import logging as _logging

    def get_logger(name: str):
        logger = _logging.getLogger(name)
        if not _logging.getLogger().handlers:
            _logging.basicConfig(level=_logging.INFO)
        return logger

    class APIError(Exception):
        pass

    def load_config():
        # Minimal stub; real config load not needed for smoke test
        return {}

    def get_profile(user_id: str):
        return {}

    def get_server_profile(guild_id: str):
        return {}

    def with_retry(config):
        def decorator(fn):
            return fn
        return decorator

    VISION_RETRY_CONFIG = {}
    API_RETRY_CONFIG = {}

    # Minimal stub so references don't break in smoke tests
    def get_retry_manager():
        class _Dummy:
            async def run_with_fallback(self, *args, **kwargs):
                raise NotImplementedError("Fallback manager not available in smoke mode")
        return _Dummy()

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
    **kwargs
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
        timeout_seconds = float(config.get('TEXT_REQUEST_TIMEOUT', '30'))
        
        client = openai.AsyncOpenAI(
            api_key=config.get('OPENAI_API_KEY'),
            base_url=config.get('OPENAI_API_BASE', 'https://api.openai.com/v1'),
            timeout=httpx.Timeout(timeout_seconds),
            max_retries=0  # Let our retry system handle retries
        )
        
        # Get model configuration
        model = config.get('OPENAI_TEXT_MODEL', 'gpt-4')
        
        # Get user preferences if user_id is provided
        if user_id:
            profile = get_profile(str(user_id))
            user_prefs = profile.get('preferences', {}) if profile else {}
            
            # Apply user preferences if not overridden
            if temperature is None:
                temperature = user_prefs.get('temperature', config.get('TEMPERATURE', 0.7))
        else:
            temperature = temperature or config.get('TEMPERATURE', 0.7)
        
        # Get server context if guild_id is provided
        server_context = ""
        if guild_id:
            server_profile = get_server_profile(str(guild_id))
            if server_profile:
                server_context = server_profile.get('context_notes', '')
        
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
            messages.append({"role": "system", "content": f"PREVIOUS_CONVERSATION_HISTORY:\n{context}"})

        # Finally, add the user's actual prompt
        messages.append({"role": "user", "content": prompt})
        
        # Set default max_tokens if not provided
        if max_tokens is None:
            max_tokens = config.get('MAX_RESPONSE_TOKENS', 1000)
        
        logger.info(f"Generating OpenAI response with model: {model}")
        logger.debug(f"[OpenAI] Request params: temp={temperature}, max_tokens={max_tokens}, stream={stream}")
        
        # Helper to normalize a completion response into our result dict
        def _normalize_nonstream_response(response_obj, used_model: str) -> Dict[str, Any]:
            try:
                logger.debug(f"[OpenAI] Normalizing response object: {type(response_obj)}")
                
                if not hasattr(response_obj, 'choices'):
                    logger.error(f"[OpenAI] ❌ Response object missing 'choices' attribute: {response_obj}")
                    raise APIError(f"Invalid OpenAI response structure - no choices attribute")
                    
                if not response_obj.choices:
                    logger.warning("[OpenAI] ⚠️ No choices in response - empty choices array")
                    logger.debug(f"[OpenAI] Response object: {response_obj}")
                    raise APIError("No choices returned in OpenAI response")
                    
                if not hasattr(response_obj.choices[0], 'message'):
                    logger.error(f"[OpenAI] ❌ First choice missing 'message' attribute: {response_obj.choices[0]}")
                    raise APIError("No message in OpenAI response choice")
                    
                if not response_obj.choices[0].message:
                    logger.error("[OpenAI] ❌ No message in first choice")
                    logger.debug(f"[OpenAI] First choice: {response_obj.choices[0]}")
                    raise APIError("No message in OpenAI response choice")
                    
                if not hasattr(response_obj.choices[0].message, 'content'):
                    logger.error(f"[OpenAI] ❌ Message missing 'content' attribute: {response_obj.choices[0].message}")
                    raise APIError("Message has no content attribute")
                    
                if not response_obj.choices[0].message.content:
                    logger.warning("[OpenAI] ⚠️ Empty message content returned")
                    response_text_local = ""
                else:
                    response_text_local = response_obj.choices[0].message.content
                    
                usage_info_local = {
                    'prompt_tokens': response_obj.usage.prompt_tokens if response_obj.usage else 0,
                    'completion_tokens': response_obj.usage.completion_tokens if response_obj.usage else 0,
                    'total_tokens': response_obj.usage.total_tokens if response_obj.usage else 0
                }
                return {
                    'text': response_text_local,
                    'model': used_model,
                    'usage': usage_info_local,
                    'backend': 'openai'
                }
            except Exception as norm_error:
                logger.error(f"[OpenAI] ❌ Error normalizing response: {type(norm_error).__name__}: {norm_error}")
                logger.debug(f"[OpenAI] Raw response object: {response_obj}")
                raise APIError(f"Response normalization failed: {type(norm_error).__name__}: {norm_error}")

        # If streaming, do a single attempt with standard retry; fallback ladder is not supported for streams
        if stream:
            logger.debug("[OpenAI] 🔄 Sending request to API (streaming)...")
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            logger.debug("[OpenAI] ✅ Received response from API (streaming)")
            logger.debug("[OpenAI] 🌊 Processing streaming response...")
            async def stream_generator():
                chunk_count = 0
                async for chunk in response:
                    chunk_count += 1
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield {
                            'text': chunk.choices[0].delta.content,
                            'finished': False
                        }
                logger.debug(f"[OpenAI] ✅ Streaming complete, processed {chunk_count} chunks")
                yield {'text': '', 'finished': True}
            return stream_generator()

        # Non-streaming: if using OpenRouter base, engage provider/model fallback ladder
        base_url = str(config.get('OPENAI_API_BASE', 'https://api.openai.com/v1') or '')
        use_openrouter_fallback = 'openrouter' in base_url.lower()

        if use_openrouter_fallback:
            logger.info("[OpenAI] Using EnhancedRetryManager text fallback ladder (OpenRouter)")
            retry_mgr = get_retry_manager()

            def _coro_factory(provider_config):
                selected_model = provider_config.model
                async def _run():
                    try:
                        logger.debug(f"[OpenAI] 🔄 Sending request to API with model: {selected_model}")
                        resp = await client.chat.completions.create(
                            model=selected_model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=False,
                            **kwargs
                        )
                        logger.debug("[OpenAI] ✅ Received response from API")
                        return _normalize_nonstream_response(resp, selected_model)
                    except httpx.HTTPStatusError as he:
                        status_code = he.response.status_code if getattr(he, 'response', None) is not None else 'unknown'
                        retry_after = None
                        try:
                            if getattr(he, 'response', None) is not None:
                                ra = he.response.headers.get('retry-after') or he.response.headers.get('Retry-After')
                                if ra is not None:
                                    retry_after = float(ra)
                        except Exception:
                            retry_after = None
                        extra = f" (retry-after={retry_after}s)" if retry_after is not None else ""
                        logger.warning(f"OpenAI HTTP error during fallback attempt: {status_code} {he}{extra}")
                        err = APIError(f"HTTP {status_code}: {str(he)}{extra}")
                        # Propagate Retry-After to outer retry harness
                        try:
                            if retry_after is not None:
                                setattr(err, 'retry_after_seconds', retry_after)
                        except Exception:
                            pass
                        raise err
                return _run

            per_item_budget = float(config.get('TEXT_PER_ITEM_BUDGET', 45.0))
            rr = await retry_mgr.run_with_fallback('text', _coro_factory, per_item_budget=per_item_budget)
            if not rr.success:
                # Re-raise last error (if present) to flow into with_retry() logic
                if rr.error:
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
                **kwargs
            )
        except Exception as e:
            if "timeout" in str(e).lower() or "TimeoutError" in str(type(e).__name__):
                logger.warning(f"[OpenAI] ⏰ Request timeout after {config.get('TEXT_REQUEST_TIMEOUT', 30)}s: {e}")
                raise APIError(f"Request timeout: {str(e)}")
            else:
                logger.error(f"[OpenAI] ❌ API request failed: {e}")
                raise
        logger.debug("[OpenAI] ✅ Received response from API")
        
        # Handle non-streaming response
        logger.debug("[OpenAI] 📝 Processing non-streaming response...")
        logger.debug(f"[OpenAI] Response object type: {type(response)}")
        result = _normalize_nonstream_response(response, model)
        logger.debug("[OpenAI] ✅ Response processing complete")
        return result
    
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI authentication failed: {e}")
        raise APIError(f"OpenAI authentication failed - check API key: {str(e)}")
    except openai.RateLimitError as e:
        logger.warning(f"OpenAI rate limit exceeded: {e}")
        retry_after = None
        try:
            # Try common headers if available on the error
            resp = getattr(e, 'response', None)
            if resp is not None and getattr(resp, 'headers', None) is not None:
                ra = resp.headers.get('retry-after') or resp.headers.get('Retry-After')
                if ra is not None:
                    retry_after = float(ra)
        except Exception:
            retry_after = None
        err = APIError(f"OpenAI rate limit exceeded: {str(e)}" + (f" (retry-after={retry_after}s)" if retry_after is not None else ""))
        try:
            if retry_after is not None:
                setattr(err, 'retry_after_seconds', retry_after)
        except Exception:
            pass
        raise err
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise APIError(f"OpenAI API error: {str(e)}")
    except httpx.HTTPStatusError as e:
        # Surface HTTP errors (e.g., 429 Too Many Requests from OpenRouter) as retriable APIError
        status = e.response.status_code if e.response is not None else 'unknown'
        retry_after = None
        try:
            if getattr(e, 'response', None) is not None:
                ra = e.response.headers.get('retry-after') or e.response.headers.get('Retry-After')
                if ra is not None:
                    retry_after = float(ra)
        except Exception:
            retry_after = None
        extra = f" (retry-after={retry_after}s)" if retry_after is not None else ""
        logger.warning(f"OpenAI HTTP error: {status} {e}{extra}")
        err = APIError(f"HTTP {status}: {str(e)}{extra}")
        try:
            if retry_after is not None:
                setattr(err, 'retry_after_seconds', retry_after)
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
        
        logger.error(f"Unexpected error in generate_openai_response: {error_details}", exc_info=True)
        raise APIError(f"Failed to generate OpenAI response: {error_details}")


async def get_base64_image(image_url: str) -> str:
    """
    Process image from URL or file path and convert to base64 data URI.
    CHANGE: Enhanced to handle both URLs and file paths.
    """
    logger.debug(f"📥 Processing image from: {image_url}")
    
    # Handle file paths (file:// protocol or direct file path)
    if image_url.startswith('file://') or os.path.exists(image_url):
        try:
            # Extract actual path from file:// URL if needed
            file_path = image_url[7:] if image_url.startswith('file://') else image_url
            
            # Verify file exists
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                logger.error(error_msg)
                raise APIError(error_msg)
                
            # Read file and encode to base64
            with open(file_path, 'rb') as f:
                data = f.read()
                
            # Determine content type from file extension
            ext = os.path.splitext(file_path)[1].lower()
            content_type = "image/jpeg" if ext in ('.jpg', '.jpeg') else \
                          "image/png" if ext == '.png' else \
                          "image/webp" if ext == '.webp' else \
                          "image/gif" if ext == '.gif' else \
                          "image/png"  # Default to PNG
                          
            base64_data = base64.b64encode(data).decode('utf-8')
            logger.debug(f"✅ Image loaded from file: size={len(data)} bytes, type={content_type}")
            return f"data:{content_type};base64,{base64_data}"
            
        except Exception as e:
            error_msg = f"Failed to process image file: {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)
    
    # Handle HTTP/HTTPS URLs
    elif image_url.startswith(('http://', 'https://')):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        data = await response.read()
                        base64_data = base64.b64encode(data).decode('utf-8')
                        content_type = response.headers.get('Content-Type', 'image/png')
                        logger.debug(f"✅ Image downloaded: size={len(data)} bytes, type={content_type}")
                        return f"data:{content_type};base64,{base64_data}"
                    else:
                        error_msg = f"Failed to download image: status={response.status}"
                        logger.error(error_msg)
                        raise APIError(error_msg)
        except Exception as e:
            error_msg = f"Failed to download image from URL: {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)
    
    # Already a data URL
    elif image_url.startswith('data:'):
        logger.debug("✅ Image already in data URL format")
        return image_url
    
    # Unsupported format
    else:
        error_msg = f"Unsupported image source format: {image_url[:30]}..."
        logger.error(error_msg)
        raise APIError(error_msg)


@with_retry(VISION_RETRY_CONFIG)
async def generate_vl_response(
    image_url: str,
    user_prompt: str = "",
    user_id: str = None,
    guild_id: str = None,
    temperature: float = None,
    max_tokens: int = None,
    **kwargs
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
        model_override = kwargs.pop('model_override', None)
        
        # Get VL model configuration (honor override) - MUST be defined before logging
        vl_model = (model_override or config.get('VL_MODEL'))
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
        timeout_seconds = float(config.get('VL_REQUEST_TIMEOUT', '30'))
        
        client = openai.AsyncOpenAI(
            api_key=config.get('OPENAI_API_KEY'),
            base_url=config.get('OPENAI_API_BASE', 'https://api.openai.com/v1'),
            timeout=httpx.Timeout(timeout_seconds),
            max_retries=0  # Let our retry system handle retries
        )
        logger.debug(f"🎨 Using API base: {config.get('OPENAI_API_BASE', 'https://api.openai.com/v1')}")
        
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
                logger.debug(f"🎨 VL prompt preview: {vl_system_prompt[:100]}{'...' if len(vl_system_prompt) > 100 else ''}")
        except FileNotFoundError:
            logger.error(f"❌ VL prompt file not found: {vl_prompt_file_path}")
            raise APIError(f"VL prompt file not found: {vl_prompt_file_path}")
        except Exception as e:
            logger.error(f"❌ Error reading VL prompt file {vl_prompt_file_path}: {e}")
            raise APIError(f"Error reading VL prompt file {vl_prompt_file_path}: {e}")
        
        # Get user preferences if user_id is provided
        if user_id:
            profile = get_profile(str(user_id))
            user_prefs = profile.get('preferences', {}) if profile else {}
            
            # Apply user preferences if not overridden
            if temperature is None:
                temperature = user_prefs.get('temperature', config.get('TEMPERATURE', 0.7))
        else:
            temperature = temperature or config.get('TEMPERATURE', 0.7)
        
        logger.debug(f"🎨 Temperature: {temperature}")
        
        # Move VL rules to system role, keep user message clean
        system_prompt = vl_system_prompt
        user_message_text = user_prompt if user_prompt else "Analyze this image."
        
        if user_prompt:
            logger.debug(f"🎨 Enhanced prompt with user context: +{len(user_prompt)} chars")
        
        # Handle data URLs directly, otherwise try to download and encode
        if isinstance(image_url, str) and image_url.startswith('data:'):
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
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": user_message_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_content
                        }
                    }
                ]
            }
        ]
        
        # Set default max_tokens if not provided
        if max_tokens is None:
            max_tokens = config.get('MAX_RESPONSE_TOKENS', 1000)
        
        logger.debug(f"🎨 Max tokens: {max_tokens}")
        logger.info(f"🎨 Calling OpenAI VL API with model: {vl_model}")
        
        # Build initial params with all potential values
        raw_params = {
            "model": vl_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # OpenRouter allow-list: only supported parameters for /chat/completions
        OPENROUTER_ALLOWED_PARAMS = {
            "model", "messages", "temperature", "max_tokens", 
            "top_p", "presence_penalty", "frequency_penalty", 
            "stop", "stream"
        }
        
        # Filter to only allowed parameters for OpenRouter compatibility
        api_params = {k: v for k, v in raw_params.items() if k in OPENROUTER_ALLOWED_PARAMS}
        
        # Log filtered parameters for debugging
        if config.get('VL_DEBUG_FLOW', '0') == '1':
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
            if "unexpected keyword" in str(e).lower() or "not supported" in str(e).lower():
                logger.warning(f"🎨 VL call failed with param error, trying fallbacks: {e}")
                response = await _try_vl_fallback_models(client, api_params, vl_model, str(e))
            else:
                raise
        
        # CHANGE: Enhanced error handling with detailed response logging
        if response is None:
            logger.error("❌ VL API returned None response")
            raise APIError("VL API returned None response")
        
        # Fast validation with minimal logging
        if not (hasattr(response, 'choices') and response.choices and 
                hasattr(response.choices[0], 'message') and response.choices[0].message):
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
            'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
            'completion_tokens': response.usage.completion_tokens if response.usage else 0,
            'total_tokens': response.usage.total_tokens if response.usage else 0
        }
        
        logger.info("✅ VL response completed")
        
        return BotAction(
            content=response_text,
            meta={'usage': usage_info}
        )
    
    except Exception as e:
        logger.error(f"❌ Error in generate_vl_response: {e}")
        raise APIError(f"Failed to generate VL response: {str(e)}")


async def _try_vl_fallback_models(client, api_params, failed_model, error_msg):
    """Try VL model fallbacks when primary model fails with param/image errors."""
    from .config import load_config
    config = load_config()
    
    fallback_models = config.get('VL_MODEL_FALLBACKS', 'openai/gpt-4o-mini,anthropic/claude-3.5-sonnet').split(',')
    fallback_models = [m.strip() for m in fallback_models if m.strip() and m.strip() != failed_model]
    
    if not fallback_models:
        logger.error(f"❌ No VL fallback models configured, original error: {error_msg}")
        raise APIError(f"VL model {failed_model} failed and no fallbacks available: {error_msg}")
    
    for fallback_model in fallback_models:
        try:
            logger.info(f"🎨 Trying VL fallback model: {fallback_model}")
            fallback_params = api_params.copy()
            fallback_params["model"] = fallback_model
            
            response = await client.chat.completions.create(**fallback_params)
            logger.info(f"✅ VL fallback successful with: {fallback_model}")
            return response
            
        except Exception as fallback_error:
            logger.warning(f"🎨 VL fallback {fallback_model} also failed: {fallback_error}")
            continue
    
    # All fallbacks failed
    logger.error(f"❌ All VL fallbacks exhausted, original error: {error_msg}")
    raise APIError(f"VL model {failed_model} and all fallbacks failed: {error_msg}")
