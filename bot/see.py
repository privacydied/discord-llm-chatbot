"""
Centralized vision-language inference module (see)
"""
import logging
from pathlib import Path
from .ai_backend import generate_vl_response
from .exceptions import InferenceError

logger = logging.getLogger(__name__)

async def see_infer(image_data: bytes, prompt: str = "What's in this image?", mime_type: str = "image/jpeg") -> str:
    """Generate response from image(s) and prompt
    
    Args:
        image_paths: List of paths to image files
        prompt: Text prompt to guide VL model's interpretation
        
    Returns:
        Text description of the image(s)
    """
    try:
        logger.info("👁️ Vision-language inference started.")
        
        # Create a data URL with the image data
        import base64

        
        data_url = f"data:{mime_type};base64,{base64.b64encode(image_data).decode()}"
        
        # Call the VL backend
        logger.debug(f"Calling VL backend with prompt: '{prompt}'")
        response = await generate_vl_response(
            image_url=data_url,
            user_prompt=prompt
        )
        
        # Handle the response format from generate_vl_response
        if 'text' in response:
            vl_result = response['text']
            logger.debug(f"VL model returned: '{vl_result[:50]}...'")
            return vl_result
        else:
            logger.error(f"Unexpected VL response format: {response}")
            raise InferenceError("Unexpected response format from vision model")
            
    except Exception as e:
        logger.error(f"👁️ Vision inference failed: {str(e)}", exc_info=True)
        raise InferenceError(f"Vision processing failed: {str(e)}")