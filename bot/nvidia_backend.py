"""
NVIDIA NIM Backend - OpenAI-compatible endpoint for NVIDIA NIM models.

This module provides integration with NVIDIA NIM by reusing the existing OpenAI
backend infrastructure. NVIDIA NIM uses an OpenAI-compatible API, so we can
leverage all the existing OpenAI backend features (streaming, retries, fallback, etc.)
by simply pointing to NVIDIA's endpoint.

Configuration:
When TEXT_BACKEND=nvidia, users should set:
- OPENAI_API_KEY=<NVIDIA API key>
- OPENAI_API_BASE=https://integrate.api.nvidia.com/v1
- OPENAI_TEXT_MODEL=meta/llama3-70b-instruct

Or use NVIDIA-specific overrides:
- NVIDIA_NIM_API_KEY=<NVIDIA API key>
- NVIDIA_NIM_API_BASE=https://integrate.api.nvidia.com/v1
- NVIDIA_NIM_TEXT_MODEL=meta/llama3-70b-instruct

References:
- NVIDIA NIM Documentation: https://docs.nvidia.com/nim/
- API Reference: https://docs.api.nvidia.com/
"""

from typing import Any, AsyncGenerator, Dict, Union
import os

from bot.config import load_config
from bot.exceptions import APIError
from bot.utils.logging import get_logger

logger = get_logger(__name__)


async def generate_nvidia_response(
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
    Generate a response using NVIDIA NIM via OpenAI-compatible API.

    This function delegates to the OpenAI backend, which automatically uses
    NVIDIA configuration when TEXT_BACKEND=nvidia.

    Args:
        prompt: The user's input prompt
        context: Optional context to include in the prompt
        system_prompt: Optional system prompt override
        user_id: Optional user ID for personalization
        guild_id: Optional guild ID for server-specific context
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        stream: Whether to stream the response
        **kwargs: Additional parameters to pass to the API

    Returns:
        Dictionary with the generated text and metadata
    """
    # Import here to avoid circular imports
    from bot.openai_backend import generate_openai_response

    config = load_config()

    # Determine which model name to display before the OpenAI-compatible backend
    # applies the authoritative TEXT_FALLBACK_MODELS ladder.  Prefer the ladder
    # head so logs match the first actual attempted model when a ladder is set.
    ladder_head = None
    raw_ladder = os.getenv("TEXT_FALLBACK_MODELS")
    if raw_ladder:
        first_entry = str(raw_ladder).strip().strip('"').split(",", 1)[0].strip()
        ladder_head = first_entry.split("|", 1)[1].strip() if "|" in first_entry else first_entry

    model = ladder_head or config.get("NVIDIA_NIM_TEXT_MODEL") or config.get("OPENAI_TEXT_MODEL") or "meta/llama3-70b-instruct"

    logger.info(f"🚀 Using NVIDIA NIM backend with configured model/ladder head: {model}")

    try:
        # Delegate to OpenAI backend - it will use NVIDIA configuration
        result = await generate_openai_response(
            prompt=prompt,
            context=context,
            system_prompt=system_prompt,
            user_id=user_id,
            guild_id=guild_id,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )

        # Mark result as from NVIDIA backend without overwriting the actual model
        # returned by the OpenAI-compatible fallback ladder.
        if isinstance(result, dict):
            result["backend"] = "nvidia_nim"
            result.setdefault("configured_model", model)

        logger.info("✅ NVIDIA NIM response generated successfully")
        return result

    except Exception as e:
        logger.error(f"❌ NVIDIA NIM backend failed: {e}")
        raise


async def generate_nvidia_vl_response(
    image_url: str,
    user_prompt: str = "",
    user_id: str = None,
    guild_id: str = None,
    temperature: float = None,
    max_tokens: int = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate a vision-language response using NVIDIA NIM.

    Note: NVIDIA NIM currently focuses on text models. Vision tasks should
    use the default OpenAI/OpenRouter backend.
    """
    logger.warning("NVIDIA NIM vision-language not supported. NVIDIA NIM currently focuses on text models. Vision tasks will use the default backend.")
    raise APIError("NVIDIA NIM vision-language processing not available. Please use the default vision backend for image processing.")
