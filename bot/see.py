"""
Centralized vision-language inference module (see)
"""

import logging
import os
from .action import BotAction
from .ai_backend import generate_vl_response
from .exceptions import InferenceError
from .config import load_config
from .retry_utils import is_retryable_error, VISION_RETRY_CONFIG

logger = logging.getLogger(__name__)


async def see_infer(
    image_path: str, prompt: str = None, model_override: str | None = None
) -> BotAction:
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
        image_path_str = str(image_path)
        mime_type = (
            "image/jpeg"
            if image_path_str.lower().endswith((".jpg", ".jpeg"))
            else "image/png"
            if image_path_str.lower().endswith(".png")
            else "image/webp"
            if image_path_str.lower().endswith(".webp")
            else "image/unknown"
        )

        logger.debug(f"Detected MIME type: {mime_type}")

        # Load config to get VL_PROMPT_FILE if prompt not provided
        if prompt is None:
            config = load_config()
            vl_prompt_file = config.get("VL_PROMPT_FILE")

            if not vl_prompt_file:
                logger.warning("VL_PROMPT_FILE not set in config, using default prompt")
                prompt = "What's in this image? Describe it in detail."
            else:
                logger.debug(f"Loading VL prompt from file: {vl_prompt_file}")
                try:
                    with open(vl_prompt_file, "r", encoding="utf-8") as f:
                        prompt = f.read().strip()
                    logger.debug(f"Loaded VL prompt: {len(prompt)} chars")
                except Exception as e:
                    logger.error(f"Failed to load VL prompt file: {e}")
                    prompt = "What's in this image? Describe it in detail."

        # Call the VL backend with file path
        logger.debug(
            f"Calling VL backend with prompt length: {len(prompt)} chars and image: {image_path}"
        )
        response = await generate_vl_response(
            image_url=image_path,  # Pass the raw file path directly
            user_prompt=prompt,
            model_override=model_override if model_override else None,
        )

        # Handle the response format (dict from backend)
        if isinstance(response, dict) and response.get("text"):
            vl_text = response["text"]
            logger.info(f"VL model returned: {len(vl_text)} chars")
            logger.debug(f"VL result preview: '{vl_text[:100]}...'")
            return BotAction(content=vl_text)
        else:
            logger.error(f"Unexpected VL response format: {response}")
            raise InferenceError("Unexpected response format from vision model")

    except Exception as e:
        logger.error(f"👁️ Vision inference failed: {str(e)}", exc_info=True)

        # Provide user-friendly error messages based on error type
        error_str = str(e).lower()

        if is_retryable_error(e, VISION_RETRY_CONFIG):
            logger.warning("⚠️ Detected transient provider error in vision inference")
            user_message = (
                "🔧 The vision service is temporarily unavailable due to provider issues. "
                "This typically resolves within a few minutes. Please try uploading the image again shortly."
            )
        elif "file not found" in error_str or "no such file" in error_str:
            user_message = "📁 The uploaded image could not be found. Please try uploading the image again."
        elif "mime type" in error_str or "format" in error_str:
            user_message = "🖼️ The image format is not supported. Please try uploading a JPEG, PNG, or WebP image."
        elif "size" in error_str or "too large" in error_str:
            user_message = "📏 The image is too large. Please try uploading a smaller image (under 10MB)."
        else:
            user_message = (
                "❌ Vision processing failed. This could be due to a temporary service issue. "
                "Please try again, and if the problem persists, the image may not be processable."
            )

        logger.info(f"🎯 Providing user-friendly error message: {user_message}")
        raise InferenceError(user_message)
