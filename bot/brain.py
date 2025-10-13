"""
Centralized text inference module (brain)
"""

from typing import Optional
from .action import BotAction
from .ai_backend import generate_response
from .exceptions import InferenceError
from .utils.logging import get_logger
from .vl.postprocess import sanitize_model_output

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
            raw_text = response.get("text") or ""
            # Apply sanitization before measuring length (removes CoT/leakage) [SFT]
            try:
                safe_text = sanitize_model_output(raw_text)
            except Exception:
                safe_text = raw_text

            if safe_text and safe_text.strip():
                logger.info(
                    f"✅ Brain inference completed: {len(safe_text)} characters"
                )
                return BotAction(content=safe_text.strip())

            # Empty/whitespace-only model output → fallback reply to avoid silence [REH]
            try:
                # Build a concise, helpful fallback using user prompt as anchor
                src = (prompt or "").strip()
                snippet = (src[:180] + ("…" if len(src) > 180 else "")) if src else "your message"
                fallback = (
                    f"I didn’t receive a usable response from the model just now. "
                    f"Could you rephrase or add a bit more detail about “{snippet}”?"
                )
                logger.info(
                    "text.empty_fallback",
                    extra={
                        "event": "text.empty_fallback",
                        "detail": {"prompt_len": len(prompt or ""), "context_len": len(context or "")},
                    },
                )
                return BotAction(content=fallback)
            except Exception:
                # Last-resort minimal fallback
                return BotAction(
                    content="I didn’t get a usable reply. Could you rephrase or add more detail?"
                )
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
