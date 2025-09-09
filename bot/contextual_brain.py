"""Contextual brain inference with enhanced context manager integration."""

from typing import TYPE_CHECKING, Any, Dict, Optional

import discord

from bot.brain import brain_infer
from bot.util.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from bot.core.bot import LLMBot

logger = get_logger(__name__)


async def contextual_brain_infer(
    message: discord.Message,
    prompt: str,
    bot: Optional["LLMBot"] = None,
    include_cross_user: bool = True,
    return_json_envelope: bool = False,
    *,
    perception_notes: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Enhanced brain inference with contextual conversation awareness.

    Args:
        message: Discord message being responded to
        prompt: The input prompt/content
        bot: Bot instance (for accessing enhanced context manager)
        include_cross_user: Whether to include cross-user context in shared channels
        return_json_envelope: Whether to return JSON envelope format

    Returns:
        Dict containing response_text, used_history, and fallback status
    """
    logger.debug(
        f"🧠 Contextual brain inference starting [msg_id={message.id}, cross_user={include_cross_user}]"
    )

    # Default response structure
    response_data = {"response_text": "", "used_history": [], "fallback": False}

    try:
        # Get bot instance from message if not provided
        if not bot:
            bot = (
                message.guild.get_member(message.guild.me.id) if message.guild else None
            )
            if not hasattr(bot, "enhanced_context_manager"):
                logger.warning(
                    "Enhanced context manager not available, falling back to basic inference"
                )
                response_data["fallback"] = True
                response_data["response_text"] = await brain_infer(prompt)
                return response_data

        # Check if enhanced context manager is available
        if (
            not hasattr(bot, "enhanced_context_manager")
            or not bot.enhanced_context_manager
        ):
            logger.warning(
                "Enhanced context manager not available, falling back to basic inference"
            )
            response_data["fallback"] = True
            response_data["response_text"] = await brain_infer(prompt)
            return response_data

        # Get conversation context
        context_entries = bot.enhanced_context_manager.get_context_for_user(
            message, include_cross_user=include_cross_user
        )

        # Build contextual prompt with optional perception notes and extra context
        contextual_prompt = prompt
        history_block = None
        try:
            if context_entries:
                context_str = bot.enhanced_context_manager.format_context_string(
                    context_entries
                )
                if context_str:
                    history_block = f"Conversation history:\n{context_str}"
                    logger.debug(
                        f"✔ Context added to prompt [entries={len(context_entries)}, tokens≈{len(context_str) // 4}]"
                    )
        except Exception as _e:
            logger.debug(f"Context history build failed: {_e}")

        perception_block = None
        if perception_notes:
            perception_block = (
                "Perception (from the image the user replied to):\n"
                + perception_notes.strip()
            )
            # Breadcrumb: show that we are injecting, but do not log full notes
            try:
                logger.info(
                    f"🧩 Injecting perception into text prompt | chars={len(perception_notes)}"
                )
                logger.info(
                    "Prompt preview includes: 'Perception (from the image...' section"
                )
            except Exception:
                pass

        blocks = []
        if history_block:
            blocks.append(history_block)
        if extra_context:
            blocks.append(extra_context.strip())
        if perception_block:
            blocks.append(perception_block)

        if blocks:
            contextual_prompt = "\n\n".join(blocks) + f"\n\nCurrent message: {prompt}"

        # Generate AI response
        ai_response = await brain_infer(contextual_prompt)

        # Get contextual response with metadata
        context_response = await bot.enhanced_context_manager.get_contextual_response(
            message=message,
            response_text=ai_response.content
            if hasattr(ai_response, "content")
            else str(ai_response),
            include_cross_user=include_cross_user,
        )

        # Build response data
        response_data = {
            "response_text": context_response.response_text,
            "used_history": context_response.used_history,
            "fallback": context_response.fallback,
        }

        logger.debug(
            f"✔ Contextual brain inference complete [fallback={context_response.fallback}]"
        )

    except Exception as e:
        logger.error(f"❌ Contextual brain inference failed: {e}", exc_info=True)

        # Fallback to basic inference with enhanced error handling [REH]
        try:
            logger.info("🔄 Attempting fallback to basic brain inference")
            basic_response = await brain_infer(prompt)
            response_text = (
                basic_response.content
                if hasattr(basic_response, "content")
                else str(basic_response)
            )

            # Check if the basic response is already an error message from brain_infer
            if any(emoji in response_text for emoji in ["🤖", "🔐", "⏱️", "⏰"]):
                # brain_infer already provided a user-friendly error message
                response_data = {
                    "response_text": response_text,
                    "used_history": [],
                    "fallback": True,
                }
            else:
                # Normal fallback response
                response_data = {
                    "response_text": f"Got you — y'all wild. {response_text}",
                    "used_history": [],
                    "fallback": True,
                }
            logger.info("✅ Fallback to basic inference successful")

        except Exception as fallback_error:
            logger.error(f"❌ Fallback inference also failed: {fallback_error}")

            # Provide user-friendly error message based on error type [REH]
            error_str = str(fallback_error).lower()
            if "no choices returned" in error_str:
                error_message = "🤖 I'm experiencing an issue with the AI service. This might be due to content filtering, API limits, or a temporary service issue. Please try rephrasing your message or try again in a moment."
            elif "authentication" in error_str or "api key" in error_str:
                error_message = "🔐 There's an authentication issue with the AI service. Please contact an administrator."
            elif "rate limit" in error_str or "quota" in error_str:
                error_message = "⏱️ The AI service is currently rate-limited. Please wait a moment and try again."
            elif "timeout" in error_str:
                error_message = "⏰ The AI service timed out. Please try again with a shorter message."
            else:
                error_message = "🤖 I'm experiencing a temporary issue generating a response. Please try again in a moment."

            response_data = {
                "response_text": error_message,
                "used_history": [],
                "fallback": True,
            }
            logger.info(f"📢 Providing user-friendly error message: {error_message}")

    return response_data


async def contextual_brain_infer_simple(
    message: discord.Message,
    prompt: str,
    bot: Optional["LLMBot"] = None,
    *,
    perception_notes: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> str:
    """
    Simplified contextual brain inference that returns just the response text.
    For backward compatibility with existing code.

    Args:
        message: Discord message being responded to
        prompt: The input prompt/content
        bot: Bot instance

    Returns:
        Response text string
    """
    result = await contextual_brain_infer(
        message,
        prompt,
        bot,
        return_json_envelope=False,
        perception_notes=perception_notes,
        extra_context=extra_context,
    )
    return result["response_text"]


def create_context_command_handler(bot: "LLMBot"):
    """
    Create command handlers for context management.

    Args:
        bot: Bot instance

    Returns:
        Dict of command handlers
    """

    async def handle_context_reset(message: discord.Message) -> str:
        """Reset conversation context."""
        if bot.enhanced_context_manager:
            bot.enhanced_context_manager.reset_context(message)
            return "✔ Conversation context has been reset."
        return "❌ Enhanced context manager not available."

    async def handle_context_stats(message: discord.Message) -> str:
        """Get context manager statistics."""
        if bot.enhanced_context_manager:
            stats = bot.enhanced_context_manager.get_stats()
            return (
                f"📊 Context Stats:\n"
                f"• Total contexts: {stats['total_contexts']}\n"
                f"• Total messages: {stats['total_messages']}\n"
                f"• Privacy opt-outs: {stats['privacy_opt_outs']}\n"
                f"• History window: {stats['history_window']}\n"
                f"• Encryption: {'✔' if stats['encryption_enabled'] else '❌'}"
            )
        return "❌ Enhanced context manager not available."

    async def handle_privacy_optout(
        message: discord.Message, opt_out: bool = True
    ) -> str:
        """Handle privacy opt-out/opt-in."""
        if bot.enhanced_context_manager:
            bot.enhanced_context_manager.set_privacy_opt_out(message.author.id, opt_out)
            action = "opted out of" if opt_out else "opted into"
            return f"✔ You have been {action} conversation context tracking."
        return "❌ Enhanced context manager not available."

    return {
        "context_reset": handle_context_reset,
        "context_stats": handle_context_stats,
        "privacy_optout": lambda msg: handle_privacy_optout(msg, True),
        "privacy_optin": lambda msg: handle_privacy_optout(msg, False),
    }
