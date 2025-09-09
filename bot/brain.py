"""
Centralized text inference module (brain)
"""

from typing import Optional
from .action import BotAction
from .ai_backend import generate_response
from .exceptions import InferenceError
from .util.logging import get_logger

logger = get_logger(__name__)


async def brain_infer(
    prompt: str, context: str = "", system_prompt: Optional[str] = None
) -> BotAction:
    """Generate text response using configured AI backend with graceful error handling [REH]"""
    try:
        logger.info("🧠 Brain inference started")
        response = await generate_response(
            prompt=prompt, context=context, system_prompt=system_prompt, stream=False
        )
        # The response is already processed in openai_backend.py
        # It should have a 'text' key with the generated content
        if "text" in response:
            logger.info(
                f"✅ Brain inference completed: {len(response['text'])} characters"
            )
            return BotAction(content=response["text"])
        else:
            logger.error(f"Unexpected response format: {response}")
            raise InferenceError("Unexpected response format from AI backend")
    except Exception as e:
        logger.error(f"🧠 Brain inference failed: {str(e)}")

        # Provide user-friendly error message based on error type [REH]
        error_str = str(e).lower()
        if "no choices returned" in error_str:
            user_message = "🤖 I'm experiencing an issue with the AI service. This might be due to content filtering, API limits, or a temporary service issue. Please try rephrasing your message or try again in a moment."
        elif "authentication" in error_str or "api key" in error_str:
            user_message = "🔐 There's an authentication issue with the AI service. Please contact an administrator."
        elif "rate limit" in error_str or "quota" in error_str:
            user_message = "⏱️ The AI service is currently rate-limited. Please wait a moment and try again."
        elif "timeout" in error_str:
            user_message = (
                "⏰ The AI service timed out. Please try again with a shorter message."
            )
        else:
            user_message = "🤖 I'm experiencing a temporary issue generating a response. Please try again in a moment."

        logger.info(f"📢 Providing user-friendly error message: {user_message}")
        return BotAction(content=user_message)
