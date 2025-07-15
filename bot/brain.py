"""
Centralized text inference module (brain)
"""
import logging
from .ai_backend import generate_response
from .exceptions import InferenceError

logger = logging.getLogger(__name__)

async def brain_infer(prompt: str) -> str:
    """Generate text response using configured AI backend"""
    try:
        logger.info("🧠 Brain inference started")
        response = await generate_response(
            prompt=prompt,
            stream=False
        )
        # The response is already processed in openai_backend.py
        # It should have a 'text' key with the generated content
        if 'text' in response:
            return response['text']
        else:
            logger.error(f"Unexpected response format: {response}")
            raise InferenceError("Unexpected response format from AI backend")
    except Exception as e:
        logger.error(f"🧠 Brain inference failed: {str(e)}")
        raise InferenceError(f"Text generation failed: {str(e)}")