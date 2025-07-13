"""
Centralized vision-language inference module (see)
"""
import logging
from pathlib import Path
from .ai_backend import generate_vl_response
from .exceptions import InferenceError

logger = logging.getLogger(__name__)

async def see_infer(image_path: Path, prompt: str) -> str:
    """Generate response from image and prompt"""
    try:
        logger.info(f"👁️ Vision-language inference started for {image_path}")
        
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
        response = await generate_vl_response(
            image_url=data_url,
            user_prompt=prompt
        )
        
        # Handle the response format from generate_vl_response
        if 'text' in response:
            return response['text']
        else:
            logger.error(f"Unexpected VL response format: {response}")
            raise InferenceError("Unexpected response format from vision model")
            
    except Exception as e:
        logger.error(f"👁️ Vision inference failed: {str(e)}", exc_info=True)
        raise InferenceError(f"Vision processing failed: {str(e)}")