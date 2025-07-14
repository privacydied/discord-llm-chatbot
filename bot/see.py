"""
Centralized vision-language inference module (see)
"""
import logging
from pathlib import Path
from .ai_backend import generate_vl_response
from .exceptions import InferenceError

logger = logging.getLogger(__name__)

async def see_infer(image_paths: list[Path], prompt: str = "What's in this image?") -> str:
    """Generate response from image(s) and prompt
    
    Args:
        image_paths: List of paths to image files
        prompt: Text prompt to guide VL model's interpretation
        
    Returns:
        Text description of the image(s)
    """
    try:
        # Convert to list if a single Path was provided
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        
        logger.info(f"👁️ Vision-language inference started for {len(image_paths)} image(s)")
        
        # Process the first image only for now (VL model may not support multiple images)
        image_path = image_paths[0]
        if len(image_paths) > 1:
            logger.warning(f"Multiple images provided ({len(image_paths)}), but only using the first one")
        
        # Read the image file directly and convert to base64 data URI
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Create a data URL with the image data
        import base64
        from mimetypes import guess_type
        
        mime_type, _ = guess_type(str(image_path))
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/png'  # Default to png if mime type can't be determined
        
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