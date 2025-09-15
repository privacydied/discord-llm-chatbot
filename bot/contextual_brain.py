"""Contextual brain inference with enhanced context manager integration."""

from typing import TYPE_CHECKING, Any, Dict, Optional

import discord
import re

from bot.brain import brain_infer
from bot.utils.logging import get_logger
from bot.decision_helpers import resolve_scope

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
    system_prompt: Optional[str] = None,
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
                response_data["response_text"] = await brain_infer(
                    prompt, system_prompt=system_prompt
                )
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
            response_data["response_text"] = await brain_infer(
                prompt, system_prompt=system_prompt
            )
            return response_data

        # Derive include_cross_user when mention/reply context is injected (extra_context set) in non-thread replies
        include_cross_user_eff = include_cross_user
        try:
            is_thread = isinstance(message.channel, discord.Thread)
            is_reply = getattr(message, "reference", None) is not None
            if extra_context and (not is_thread) and is_reply:
                include_cross_user_eff = False
        except Exception:
            include_cross_user_eff = include_cross_user

        # Determine basic case/scope for logging and locality policy
        _scope = resolve_scope(message)
        case = _scope.case.upper() if _scope.case else "LONE"
        scope_id = _scope.scope_id

        # Locality-first policies
        try:
            mentioned_me = bot.user in (getattr(message, "mentions", None) or [])
        except Exception:
            mentioned_me = False
        try:
            content_has_signal = bool(re.search(r"[A-Za-z0-9]", prompt or ""))
        except Exception:
            content_has_signal = bool(prompt and prompt.strip())

        # Skip history if:
        # - We have an explicit local block (extra_context) for THREAD/REPLY/LONE
        # - Or LONE @mention has substantive content
        skip_history = bool(extra_context)
        if not skip_history:
            try:
                if mentioned_me and (case == "LONE") and content_has_signal:
                    skip_history = True
            except Exception:
                pass

        if skip_history:
            # Optionally compute how many entries would have been dropped for visibility
            dropped = 0
            try:
                _entries = bot.enhanced_context_manager.get_context_for_user(
                    message, include_cross_user=False
                )
                dropped = len(_entries or [])
            except Exception:
                dropped = 0
            # Minimal, same-schema breadcrumbs
            try:
                logger.info(
                    "scope_resolved",
                    extra={
                        "subsys": "mem.ctx",
                        "event": "scope_resolved",
                        "guild_id": getattr(
                            getattr(message, "guild", None), "id", None
                        ),
                        "user_id": getattr(
                            getattr(message, "author", None), "id", None
                        ),
                        "msg_id": getattr(message, "id", None),
                        "detail": {"case": case, "scope": scope_id},
                    },
                )
                logger.info(
                    "local_block",
                    extra={
                        "subsys": "mem.ctx",
                        "event": "local_block",
                        "guild_id": getattr(
                            getattr(message, "guild", None), "id", None
                        ),
                        "user_id": getattr(
                            getattr(message, "author", None), "id", None
                        ),
                        "msg_id": getattr(message, "id", None),
                        "detail": {"msgs": 0, "order": "local_first"},
                    },
                )
                if dropped:
                    logger.info(
                        "drop_stale",
                        extra={
                            "subsys": "mem.ctx",
                            "event": "drop_stale",
                            "guild_id": getattr(
                                getattr(message, "guild", None), "id", None
                            ),
                            "user_id": getattr(
                                getattr(message, "author", None), "id", None
                            ),
                            "msg_id": getattr(message, "id", None),
                            "detail": {"reason": "scope_mismatch", "dropped": dropped},
                        },
                    )
            except Exception:
                pass
            context_entries = []
        else:
            # Get conversation context normally
            context_entries = bot.enhanced_context_manager.get_context_for_user(
                message, include_cross_user=include_cross_user_eff
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
        # Prepend mention/reply context before historical memory
        if extra_context:
            blocks.append(extra_context.strip())
        if history_block:
            blocks.append(history_block)
        if perception_block:
            blocks.append(perception_block)

        if blocks:
            contextual_prompt = "\n\n".join(blocks) + f"\n\nCurrent message: {prompt}"

        # Generate AI response
        ai_response = await brain_infer(contextual_prompt, system_prompt=system_prompt)

        # Get contextual response with metadata
        context_response = await bot.enhanced_context_manager.get_contextual_response(
            message=message,
            response_text=(
                ai_response.content
                if hasattr(ai_response, "content")
                else str(ai_response)
            ),
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
            basic_response = await brain_infer(prompt, system_prompt=system_prompt)
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
    system_prompt: Optional[str] = None,
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
        system_prompt=system_prompt,
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
