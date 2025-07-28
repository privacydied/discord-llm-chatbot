"""
Centralized vision-language inference module (see)
"""
import logging
import os
from pathlib import Path
from .ai_backend import generate_vl_response
from .exceptions import InferenceError
from .config import load_config

logger = logging.getLogger(__name__)

async def see_infer(image_path: str, prompt: str = None) -> str:
    """Generate response from image path and prompt
    
    Args:
        image_path: Path to image file
        prompt: Text prompt to guide VL model's interpretation (optional, will use VL_PROMPT_FILE if not provided)
        
    Returns:
        Text description of the image
    """
    try:
        logger.info("👁️ Vision-language inference started.")
        logger.debug(f"Processing image at path: {image_path}")
        
        # Verify image file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            raise InferenceError(f"Image file not found: {image_path}")
            
        # Get file extension to determine MIME type
        mime_type = "image/jpeg" if image_path.lower().endswith(('.jpg', '.jpeg')) \
                   else "image/png" if image_path.lower().endswith('.png') \
                   else "image/webp" if image_path.lower().endswith('.webp') \
                   else "image/unknown"
        
        logger.debug(f"Detected MIME type: {mime_type}")
        
        # Load config to get VL_PROMPT_FILE if prompt not provided
        if prompt is None:
            config = load_config()
            vl_prompt_file = config.get('VL_PROMPT_FILE')
            
            if not vl_prompt_file:
                logger.warning("VL_PROMPT_FILE not set in config, using default prompt")
                prompt = "What's in this image? Describe it in detail."
            else:
                logger.debug(f"Loading VL prompt from file: {vl_prompt_file}")
                try:
                    with open(vl_prompt_file, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                    logger.debug(f"Loaded VL prompt: {len(prompt)} chars")
                except Exception as e:
                    logger.error(f"Failed to load VL prompt file: {e}")
                    prompt = "What's in this image? Describe it in detail."
        
        # Call the VL backend with file path
        logger.debug(f"Calling VL backend with prompt length: {len(prompt)} chars and image: {image_path}")
        response = await generate_vl_response(
            image_url=image_path,  # Pass the raw file path directly
            user_prompt=prompt
        )
        
        # Handle the response format from generate_vl_response
        if 'text' in response:
            vl_result = response['text']
            logger.info(f"VL model returned: {len(vl_result)} chars")
            logger.debug(f"VL result preview: '{vl_result[:100]}...'")
            return vl_result
        else:
            logger.error(f"Unexpected VL response format: {response}")
            raise InferenceError("Unexpected response format from vision model")
            
    except Exception as e:
        logger.error(f"👁️ Vision inference failed: {str(e)}", exc_info=True)
        raise InferenceError(f"Vision processing failed: {str(e)}")