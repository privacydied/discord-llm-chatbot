"""
Centralized vision-language inference module (see)
"""

import logging
import os
from pathlib import Path
from .action import BotAction
from .ai_backend import generate_vl_response
from .config import load_config
from .retry_utils import is_retryable_error, VISION_RETRY_CONFIG

logger = logging.getLogger(__name__)

# Phase 15: VL resource caps
VL_MAX_IMAGES = 5  # overridden by config VL_MAX_IMAGES
VL_MAX_TOTAL_BYTES = 20 * 1024 * 1024  # 20 MB default; overridden by MULTIMODAL_MAX_TOTAL_BYTES
IMAGE_MAX_DIMENSION = 2048  # overridden by config IMAGE_MAX_DIMENSION


def _load_vl_caps():
    """Load VL caps from config (cached per call, cheap)."""
    config = load_config()
    return {
        "vl_max_images": config.get("VL_MAX_IMAGES", VL_MAX_IMAGES),
        "vl_max_total_bytes": config.get("MULTIMODAL_MAX_TOTAL_BYTES", VL_MAX_TOTAL_BYTES),
        "image_max_dimension": config.get("IMAGE_MAX_DIMENSION", IMAGE_MAX_DIMENSION),
    }


def _downsample_image(image_path: str, max_dimension: int) -> str:
    """Downsample an image so its longest side <= max_dimension.

    Returns the path to the downsampled image (a temp JPEG), or the original
    path if downsizing was not needed.
    """
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            w, h = img.size
            if w <= max_dimension and h <= max_dimension:
                return image_path

            ratio = max_dimension / max(w, h)
            new_w = max(1, int(w * ratio))
            new_h = max(1, int(h * ratio))
            img = img.resize((new_w, new_h), Image.LANCZOS)

            tmp_dir = Path("temp")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / f"vl_downsample_{os.path.basename(image_path)}.jpg"
            img.convert("RGB").save(str(tmp_path), "JPEG", quality=85)
            logger.info(
                "vl.downsample original=%dx%d resized=%dx%d path=%s",
                w, h, new_w, new_h, tmp_path,
            )
            return str(tmp_path)
    except Exception:
        logger.warning("vl.downsample failed, using original image %s", image_path, exc_info=True)
        return image_path


async def see_infer(
    image_path: str, prompt: str = None, model_override: str | None = None
) -> BotAction:
    """Generate response from image path and prompt"""
    logger.debug(f"Processing image at path: {image_path}")

    caps = _load_vl_caps()
    max_dimension = caps["image_max_dimension"]
    max_total_bytes = caps["vl_max_total_bytes"]

    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        logger.info(
            "vl.final status=error exhausted=true reason=file_missing path=%s",
            image_path,
        )
        return BotAction(
            content="📁 The uploaded image could not be found. Please try uploading the image again."
        )

    # Phase 15: Check file size before processing
    try:
        file_size = os.path.getsize(image_path)
        if file_size > max_total_bytes:
            logger.warning(
                "vl.reject reason=too_large size=%d max=%d path=%s",
                file_size, max_total_bytes, image_path,
            )
            logger.info(
                "vl.final status=error exhausted=true reason=too_large path=%s",
                image_path,
            )
            return BotAction(
                content="📏 The image is too large. Please try uploading a smaller image."
            )
    except OSError:
        pass

    # Phase 15: Downsample large images before VL
    image_path_str = str(image_path)
    image_path_str = _downsample_image(image_path_str, max_dimension)

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
            except Exception as exc:
                logger.error(f"Failed to load VL prompt file: {exc}")
                prompt = "What's in this image? Describe it in detail."

    try:
        logger.debug(
            f"Calling VL backend with prompt length: {len(prompt)} chars and image: {image_path_str}"
        )
        response = await generate_vl_response(
            image_url=image_path_str,
            user_prompt=prompt,
            model_override=model_override if model_override else None,
        )

        telemetry = {}
        if isinstance(response, dict):
            telemetry = response.get("telemetry") or {}
            if response.get("ladder_exhausted"):
                ladder_summary = telemetry.get("ladder_summary")
                attempts = telemetry.get("ladder_attempts")
                provider_base = telemetry.get("provider_base")
                logger.info(
                    "vl.final status=error exhausted=true ladder=%s attempts=%s provider_base=%s scope=see",
                    ladder_summary,
                    attempts,
                    provider_base,
                )
                friendly_text = response.get("text") or (
                    "🔧 The vision service is temporarily unavailable. Please try again in a few minutes."
                )
                return BotAction(content=friendly_text, error=True)

            # Check for non-empty text content [REH][CA]
            vl_text = response.get("text", "")
            if vl_text and vl_text.strip():
                logger.info(f"VL model returned: {len(vl_text)} chars")
                logger.debug(f"VL result preview: '{vl_text[:100]}...'")
                logger.info(
                    "vl.final status=ok model=%s attempts=%s scope=see",
                    response.get("model"),
                    telemetry.get("ladder_attempts"),
                )
                return BotAction(content=vl_text)

            # Empty completion is a soft failure - model returned but produced no output [REH]
            logger.warning(
                "vl.final status=error reason=empty_completion model=%s attempts=%s scope=see",
                response.get("model"),
                telemetry.get("ladder_attempts"),
            )
            return BotAction(
                content=(
                    "🔧 The vision model returned an empty response. "
                    "Please try again with a clearer image or different prompt."
                )
            )

        if isinstance(response, str):
            # String response is typically an error message from the backend
            if response.strip():
                logger.info("vl.final status=ok type=string_response scope=see")
                return BotAction(content=response)
            logger.info(
                "vl.final status=error exhausted=true ladder=na attempts=na provider_base=na scope=see"
            )
            return BotAction(
                content="🔧 Vision processing returned an empty result. Please try again."
            )

        # Truly unexpected format - log for debugging but don't expose internals to user [REH]
        logger.error(
            "vl.final status=error reason=unexpected_format type=%s scope=see",
            type(response).__name__,
        )
        return BotAction(
            content=(
                "❌ Vision processing failed. This could be due to a temporary service issue. "
                "Please try again, and if the problem persists, the image may not be processable."
            )
        )

    except Exception as exc:
        logger.error(f"👁️ Vision inference failed: {str(exc)}", exc_info=True)

        error_str = str(exc).lower()
        reason = "generic_failure"

        if is_retryable_error(exc, VISION_RETRY_CONFIG):
            logger.warning("⚠️ Detected transient provider error in vision inference")
            user_message = (
                "🔧 The vision service is temporarily unavailable due to provider issues. "
                "This typically resolves within a few minutes. Please try uploading the image again shortly."
            )
            reason = "transient"
        elif "file not found" in error_str or "no such file" in error_str:
            user_message = "📁 The uploaded image could not be found. Please try uploading the image again."
            reason = "file_missing"
        elif "mime type" in error_str or "format" in error_str:
            user_message = "🖼️ The image format is not supported. Please try uploading a JPEG, PNG, or WebP image."
            reason = "unsupported_format"
        elif "size" in error_str or "too large" in error_str:
            user_message = "📏 The image is too large. Please try uploading a smaller image (under 10MB)."
            reason = "too_large"
        else:
            user_message = (
                "❌ Vision processing failed. This could be due to a temporary service issue. "
                "Please try again, and if the problem persists, the image may not be processable."
            )

        logger.info(
            "vl.final status=error exhausted=true reason=%s scope=see",
            reason,
        )
        return BotAction(content=user_message)
