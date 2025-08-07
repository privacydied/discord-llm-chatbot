"""
AI Backend Router - Routes requests to appropriate AI service based on configuration.
"""
from typing import Dict, Any, Union, AsyncGenerator, Optional

from .config import load_config
from .util.logging import get_logger

logger = get_logger(__name__)


async def generate_response(
    prompt: str,
    context: str = "",
    system_prompt: Optional[str] = None,
    user_id: str = None,
    guild_id: str = None,
    temperature: float = None,
    max_tokens: int = None,
    stream: bool = False,
    **kwargs
) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
    """
    Generate a response using the configured AI backend.
    
    Routes to either OpenAI/OpenRouter or Ollama based on TEXT_BACKEND configuration.
    """
    try:
        logger.info("🤖 === TEXT RESPONSE GENERATION STARTED ===")
        logger.debug(f"🤖 Prompt: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
        logger.debug(f"🤖 Context length: {len(context)} chars")
        logger.debug(f"🤖 User ID: {user_id}, Guild ID: {guild_id}")
        
        config = load_config()
        backend = config.get('TEXT_BACKEND', 'openai')
        
        logger.info(f"🤖 Using AI backend: {backend}")
        logger.debug(f"🤖 Temperature: {temperature}, Max tokens: {max_tokens}, Stream: {stream}")
        
        if backend == 'openai':
            # Use OpenAI/OpenRouter backend
            logger.debug("🤖 Routing to OpenAI/OpenRouter backend")
            from .openai_backend import generate_openai_response
            result = await generate_openai_response(
                prompt=prompt,
                context=context,
                system_prompt=system_prompt,
                user_id=user_id,
                guild_id=guild_id,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
            logger.info("✅ OpenAI backend completed successfully")
            return result
        elif backend == 'ollama':
            # Use Ollama backend
            logger.debug("🤖 Routing to Ollama backend")
            from .ollama import generate_response as ollama_generate_response
            result = await ollama_generate_response(
                prompt=prompt,
                context=context,
                system_prompt=system_prompt,
                user_id=user_id,
                guild_id=guild_id,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
            logger.info("✅ Ollama backend completed successfully")
            return result
        else:
            logger.error(f"❌ Unknown backend: {backend}")
            raise ValueError(f"Unknown backend: {backend}")
            
    except Exception as e:
        logger.error(f"❌ Error in generate_response: {e}", exc_info=True)
        raise Exception(f"Failed to generate response: {str(e)}")


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
    Generate a vision-language response using the VL model.
    CHANGE: Enhanced VL routing with comprehensive debug logging for hybrid multimodal inference.
    
    Currently only supports OpenAI/OpenRouter VL models.
    """
    try:
        logger.info("🎨 === VL BACKEND ROUTING STARTED ===")
        image_url_str = str(image_url)
        logger.debug(f"🎨 Image URL: {image_url_str[:100]}{'...' if len(image_url_str) > 100 else ''}")
        logger.debug(f"🎨 User prompt: '{user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}'")
        logger.debug(f"🎨 User ID: {user_id}, Guild ID: {guild_id}")
        
        config = load_config()
        backend = config.get('TEXT_BACKEND', 'openai')
        
        logger.info(f"🎨 Using VL backend: {backend}")
        logger.debug(f"🎨 Temperature: {temperature}, Max tokens: {max_tokens}")
        
        if backend == 'openai':
            # Use OpenAI/OpenRouter VL backend
            logger.debug("🎨 Routing to OpenAI/OpenRouter VL backend")
            from .openai_backend import generate_vl_response as openai_generate_vl_response
            result = await openai_generate_vl_response(
                image_url=image_url,
                user_prompt=user_prompt,
                user_id=user_id,
                guild_id=guild_id,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            logger.info("✅ OpenAI VL backend completed successfully")
            return result
        else:
            # For now, Ollama VL is not implemented, fallback to OpenAI
            logger.warning(f"⚠️  VL not supported for backend {backend}, falling back to OpenAI")
            from .openai_backend import generate_vl_response as openai_generate_vl_response
            result = await openai_generate_vl_response(
                image_url=image_url,
                user_prompt=user_prompt,
                user_id=user_id,
                guild_id=guild_id,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            logger.info("✅ OpenAI VL fallback completed successfully")
            return result
            
    except Exception as e:
        logger.error(f"❌ Error in generate_vl_response: {e}")
        # Re-raise the original exception to preserve retry logic
        raise e
